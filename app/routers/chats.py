import asyncio
from datetime import datetime, timezone
import io
import json
import os
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from functools import partial
from sqlalchemy import text
from app.clases.ChatModel import ChatModel
from app.clases.MessageInfo import MessageInfoV2
from app.clases.Enums import TipoMensaje, Estados
from typing import Optional
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_chroma_collection, get_session
from app.clases.ImageModel import ImageModel
from uuid import UUID
from app.clases.Responses import ChatListResponse, CompleteChatResponse
from app.core.auth import get_current_user_id
from app.core import config
from app.core.model_loader import embedder
from app.services.message_processing import (
    analizar_imagen_con_qwen_requests,
    buscar_imagenes_similares,
    procesar_mensaje_complejo_recibido,
    procesar_mensaje_simple_recibido,
)
from app.services.chat_processing import (
    add_created_msg,
    add_received_msg,
    create_chat_with_image,
    get_chat_with_id,
    get_internal_chat_id,
)
from app.services.sse_queues import get_queue
from app.services.image_processing import (
    save_image_and_get_data,
    get_image_with_id,
)

from PIL import Image

chatRouter = APIRouter()
client = config.client
qwenModel = "Qwen/Qwen3-VL-8B-Instruct-FP8"


@chatRouter.get("/me/chats/{chat_id:uuid}")
async def get_chat_details(chat_id: UUID, user: dict = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)):
    query_chat = text("""
        SELECT C.*, i.local_route, i.name AS image_name
        FROM chats AS C 
        INNER JOIN images AS i ON C.image_id = i.id
        WHERE C.id = :chat_id AND C.user_id = :user_id
    """)
    result_chat = await session.execute(query_chat, {"chat_id": chat_id, "user_id": user})
    chat = result_chat.mappings().one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado o no pertenece al usuario")
    
    chat_dict = dict(chat)
    chat_details = ChatModel(
        id=str(chat_dict['id']), 
        user_id=str(chat_dict['user_id']), 
        image_id=str(chat_dict['image_id']), 
        collection_id="",
        created_at=chat_dict['created_at'],
        status=chat_dict['status']
    )
    
    return chat_details

@chatRouter.post("/analyze")
async def search_similar_art_and_analyze(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    image_id: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    save_to_db: bool = Form(False),
    user: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
    is_new_chat: Optional[bool] = Form(False),
    question: Optional[str] = Form(None)):
    print(
        f"Parametros recibidos - file: {file}, image_id: {image_id}, chat_id: {chat_id}, "
        f"save_to_db: {save_to_db}, is_new_chat: {is_new_chat}"
    )
    collection = get_chroma_collection(use_http=True)
    if not file and not image_id and not chat_id:
        raise HTTPException(status_code=400, detail="Debe proporcionar un archivo o una ruta.")

    message_json = json.loads(question) if question else None
    mes_info = MessageInfoV2(**message_json) if message_json else None
    print(f"Pregunta parseada con Pydantic: {mes_info}\n")

    if save_to_db:
        chat_internal = None
        image = None
        if is_new_chat:
            print(
                f"Nuevo chat, creando chat y asociando imagen. is_new_chat: {is_new_chat}, "
                f"image_id: {image_id}, file: {file}"
            )
            if image_id and not file:
                print(f"Creando nuevo chat con imagen existente. image_id: {image_id}")
                chat_data = await create_chat_with_image(
                    session=session, image_id=image_id, user=user, external_id=chat_id
                )
                chat_internal = chat_data.id
                image_data = await get_image_with_id(session=session, image_id=image_id, user=user)
                image_path = os.path.join(config.CARPETA_IMAGENES, image_data.image_url)
                if os.path.exists(image_path):
                    with open(image_path, "rb") as img_file:
                        current_image = img_file.read()
                        image = Image.open(io.BytesIO(current_image))
                else:
                    raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {image_data.image_url}")
            elif image_id and file:
                current_image = await file.read()
                image_data = await save_image_and_get_data(
                    session=session, contents=current_image, user=user, commit=False
                )
                saved_image_id = image_data.id
                chat_data = await create_chat_with_image(
                    session=session, image_id=saved_image_id, user=user, external_id=chat_id
                )
                chat_internal = chat_data.id
                image = Image.open(io.BytesIO(current_image))
        else:
            print(f"Chat no nuevo, buscando imagen asociada al chat {chat_id} (ID interno: {chat_internal})")
            if chat_id:
                chat_internal = chat_id
                chat_data = await get_chat_with_id(session=session, chat_id=chat_internal, user=user)
                if not chat_data:
                    print(
                        f"No se encontró un chat con ID interno {chat_internal} para el usuario {user}. "
                        f"Intentando buscar equivalencia con ID externo {chat_id}."
                    )
                    chat_data = await get_internal_chat_id(session=session, external_id=chat_id, user_id=user)
                if not chat_data:
                    raise HTTPException(status_code=404, detail=f"Chat no encontrado para el ID proporcionado: {chat_id}")

                image_data = await get_image_with_id(session=session, image_id=chat_data.image_id, user=user)
                image_path = os.path.join(config.CARPETA_IMAGENES, image_data.image_url)
                if os.path.exists(image_path):
                    with open(image_path, "rb") as img_file:
                        current_image = img_file.read()
                        image = Image.open(io.BytesIO(current_image))
                else:
                    raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {image_data.image_url}")

        message_data = await add_received_msg(
            session=session, chat_id=chat_internal, message=mes_info, should_commit=False, question_id=None
        )

        error = False
        returned_message = await add_created_msg(
            session=session,
            chat_id=chat_internal,
            response=True,
            content="Estamos procesando tu solicitud...",
            should_commit=False,
            question_id=message_data.id,
            status=Estados.ERROR if error else Estados.WAITING,
            type=message_data.type,
        )
        await session.commit()

        response = CompleteChatResponse(
            chat_info=chat_data,
            request_info=message_data,
            message_info=returned_message,
            image_info=image_data,
        )

        print(f"RESPUESTA FINAL QUE SE ENVIA AL FRONT: {response}")

        if message_data.type == TipoMensaje.SIMPLE:
            print("Lanzando tarea en background para pregunta simple...")
            background_tasks.add_task(
                procesar_mensaje_simple_recibido,
                message_data,
                returned_message,
                chat_data,
                image_data,
                qwen_client=client,
                qwen_model=qwenModel,
                collection=collection,
                embedder=embedder,
                get_queue=get_queue,
            )
        elif message_data.type == TipoMensaje.COLORS:
            print("Lanzando tarea en background para extracción de colores...")
        elif message_data.type == TipoMensaje.COMPLETE:
            background_tasks.add_task(
                procesar_mensaje_complejo_recibido,
                message_data,
                returned_message,
                chat_data,
                image_data,
                qwen_client=client,
                qwen_model=qwenModel,
                collection=collection,
                embedder=embedder,
                get_queue=get_queue,
            )

        return response

    try:
        if image_id:
            raise HTTPException(
                status_code=400,
                detail="El flujo save_to_db=false con image_id aún no está implementado. Envia file o activa save_to_db.",
            )
        if file:
            current_image = await file.read()
            image = Image.open(io.BytesIO(current_image))
        else:
            raise HTTPException(status_code=400, detail="Debes proporcionar file o image_id.")

        results = await buscar_imagenes_similares(
            image,
            n_results=3,
            session=session,
            user=user,
            collection=collection,
            embedder=embedder,
        )

        loop = asyncio.get_event_loop()
        qwen_description = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(
                    analizar_imagen_con_qwen_requests,
                    current_image,
                    "Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, Comparacion rapida con las similares en caso de haber.",
                    results,
                    qwen_client=client,
                    qwen_model=qwenModel,
                ),
            ),
            timeout=120,
        )
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        print("Timeout al llamar a QWEN en flujo save_to_db=false")
        qwen_description = "Tiempo de espera agotado al analizar la imagen con QWEN."
        results = []
    except Exception as e:
        print(f"Error al llamar a QWEN: {e}")
        qwen_description = "Error al analizar la imagen con QWEN."
        results = []

    print(f"Descripción obtenida de QWEN: {qwen_description}")

    response = CompleteChatResponse(
        chat_info=None,
        image_info=None,
        message_info=MessageInfoV2(
            id=str(uuid.uuid4()),
            chat_id=chat_id if chat_id else "no_chat",
            response=True,
            content=qwen_description,
            created_at=datetime.now(timezone.utc),
            related_images=results,
        ),
    )

    return response

@chatRouter.post("/retry")
async def retry_endpoint(
    background_tasks: BackgroundTasks,
    response_id: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    user: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    print(f"Parametros recibidos para retry - response_id: {response_id}, chat_id: {chat_id}")
    collection = get_chroma_collection(use_http=True)
    sql_chat = text("SELECT * FROM chats WHERE id = :chat_id AND user_id = :user_id")
    result = await session.execute(sql_chat, {"chat_id": chat_id, "user_id": user})
    chat_db = result.mappings().one_or_none()
    if not chat_db:
        raise HTTPException(status_code=404, detail="Chat no encontrado para el ID proporcionado")
    sql_resp = text("SELECT * FROM messages WHERE id = :response_id AND chat_id = :chat_id")
    result = await session.execute(sql_resp, {"response_id": response_id, "chat_id": chat_id})
    resp_db = result.mappings().one_or_none()
    if not resp_db:
        raise HTTPException(status_code=404, detail="Respuesta no encontrada para el ID proporcionado")
    sql_preg = text("SELECT * FROM messages WHERE id = :question_id AND chat_id = :chat_id")
    result = await session.execute(sql_preg, {"question_id": resp_db["question_id"], "chat_id": chat_id})
    preg_db = result.mappings().one_or_none()
    if not preg_db:
        raise HTTPException(status_code=404, detail="Pregunta no encontrada para el ID proporcionado")
    image = await get_image_with_id(session=session, image_id=chat_db["image_id"], user=user)
    update_resp = text("UPDATE messages SET status = 'WAITING' WHERE id = :response_id")
    await session.execute(update_resp, {"response_id": response_id})
    await session.commit()
    chat = ChatModel(
        id=str(chat_db["id"]),
        user_id=str(chat_db["user_id"]),
        image_id=str(chat_db["image_id"]),
        created_at=chat_db["created_at"],
    )
    question_message = MessageInfoV2(
        id=str(preg_db["id"]),
        chat_id=str(preg_db["chat_id"]),
        response=preg_db["response"],
        content=preg_db["content"],
        created_at=preg_db["created_at"],
        status=preg_db["status"],
        type=preg_db["type"],
    )
    receive_message = MessageInfoV2(
        id=str(resp_db["id"]),
        chat_id=str(resp_db["chat_id"]),
        response=resp_db["response"],
        content=resp_db["content"],
        created_at=resp_db["created_at"],
        status=Estados.WAITING,
        question_id=str(resp_db["question_id"]) if resp_db["question_id"] else None,
        type=resp_db["type"],
    )

    if receive_message.type == TipoMensaje.SIMPLE:
        print("Lanzando tarea en background para pregunta simple...")
        background_tasks.add_task(
            procesar_mensaje_simple_recibido,
            question_message,
            receive_message,
            chat,
            image,
            qwen_client=client,
            qwen_model=qwenModel,
            collection=collection,
            embedder=embedder,
            get_queue=get_queue,
        )
    elif receive_message.type == TipoMensaje.COLORS:
        print("Lanzando tarea en background para extracción de colores...")
    elif receive_message.type == TipoMensaje.COMPLETE:
        background_tasks.add_task(
            procesar_mensaje_complejo_recibido,
            question_message,
            receive_message,
            chat,
            image,
            qwen_client=client,
            qwen_model=qwenModel,
            collection=collection,
            embedder=embedder,
            get_queue=get_queue,
        )

    return receive_message

@chatRouter.get("/me/chats")
async def get_user_chats(last_date : Optional[str] = None, user: dict = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)) -> ChatListResponse:
    query_chats = text("""
        SELECT C.*
        FROM chats AS C 
        WHERE C.user_id = :user_id
        ORDER BY C.created_at DESC
    """)
    result_chats = await session.execute(query_chats, {"user_id": user, "last_date": last_date})
    chats = result_chats.mappings().all()
    
    chatsRes = []
    for chat in chats:
        chat_dict = dict(chat)
        new_chat = ChatModel(
            id=str(chat_dict['id']), 
            user_id=str(chat_dict['user_id']), 
            image_id=str(chat_dict['image_id']), 
            created_at=chat_dict['created_at'],
            external_id=str(chat_dict['external_id']),
            status=chat_dict['status']
        )
        chatsRes.append(new_chat)
    
    print(f"Chats encontrados para el usuario {user}: {chatsRes}")
    return ChatListResponse(chats=chatsRes)

@chatRouter.get("/chats/messages")
async def get_chat_messages(
    chat_id: Optional[str] = None,
    user: dict = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session)
):
    """Obtiene todos los mensajes de un chat específico"""
    # Verificar que el chat pertenece al usuario
    chat_details = await get_chat_with_id(session=session, chat_id=chat_id, user=user)
    if not chat_details:
        chat_details = await get_internal_chat_id(session=session, external_id=chat_id, user_id=user)
    if not chat_details:
        raise HTTPException(status_code=404, detail="Chat no encontrado o no pertenece al usuario")
    
    # Obtener mensajes ordenados por fecha de creación
    query_msgs = text("""
        SELECT *
        FROM messages 
        WHERE chat_id = :chat_id 
        ORDER BY created_at DESC
    """)
    result_msgs = await session.execute(query_msgs, {"chat_id": chat_details.id})
    messages = result_msgs.mappings().all()

    messages_res = []
    for msg in messages:
        msgInfo = MessageInfoV2(
            id=str(msg['id']),
            chat_id=str(msg['chat_id']),
            response=msg['response'],
            content=msg['content'],
            created_at=msg['created_at'],
            status=msg["status"],
            related_images=[],
        )
        
        print(f"La fecha es: {msg['created_at']}")
        
        msg_dict = dict(msg)
        msg_dict['id'] = str(msg_dict['id'])
        msg_dict['chat_id'] = str(msg_dict['chat_id'])
        msg_dict['created_at'] = msg_dict['created_at']
        
        # Obtener imágenes relacionadas con este mensaje
        query_related = text("""
            SELECT
                ri.id as relation_id,
                ri.similarity,
                i.id,
                i.local_route,
                i.name,
                i.year,
                a.name AS artist,
                s.name AS style,
                g.name AS genre
            FROM related_images ri
            JOIN images i ON ri.image_id = i.id
            LEFT JOIN artists AS a ON i.artist_id = a.id
            LEFT JOIN styles AS s ON i.style_id = s.id
            LEFT JOIN genres AS g ON i.genre_id = g.id
            WHERE ri.message_id = :message_id
            ORDER BY ri.similarity DESC
        """)
        result_related = await session.execute(query_related, {"message_id": msg['id']})
        related_images = result_related.mappings().all()
        
        # Formatear imágenes relacionadas
        related_images_list = []
        for img in related_images:
            img_dict = {
                'id': str(img['id']),
                'name': img['name'],
                'artist': img['artist'],
                'style': img['style'],
                'genre': img['genre'],
                'year': img['year'],
                'similarity_score': float(img['similarity']),
                'image_url': f"/art/{img['local_route']}"
            }
            imgData = ImageModel(
                id=str(img_dict['id']),
                name=img_dict['name'],
                artist=img_dict['artist'],
                style=img_dict['style'],
                genre=img_dict['genre'],
                year=img_dict['year'],
                similarity_score=img_dict['similarity_score'],
                image_url=img_dict['image_url']
            )
            related_images_list.append(imgData)
            msgInfo.related_images.append(imgData)
        
        msg_dict['related_images'] = related_images_list
        messages_res.append(msgInfo)
    
    for msg in messages_res:
        print(f"Mensaje ID: {msg.id}, timestamp: {msg.created_at}")
    
    image_info = await get_image_with_id(session=session, image_id=chat_details.image_id, user=user)
    print(f"Información de la imagen asociada al chat: {image_info}")

    return {"messages": messages_res,
            "chat_details": chat_details,
            "image_info": image_info}


