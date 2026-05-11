from datetime import datetime, timezone
import uuid
from fastapi import APIRouter
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import text
from clases.ChatModel import ChatModel
from clases.MessageInfo import MessageInfoV2
from clases.Enums import TipoMensaje, Estados
from typing import Optional
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import config
from database import get_session
from clases.ImageModel import ImageModel
from uuid import UUID
from clases.ChatModel import ChatListResponse
from images import get_image_with_id

chatRouter = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        now = int(datetime.now().timestamp())
        print("DEBUG now/exp delta:", now, payload.get("exp"), payload.get("exp") - now)
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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


async def add_created_msg(session, chat_id, response, content, should_commit=False, question_id=None, status: Estados = Estados.SUCCESS, type=TipoMensaje.SIMPLE) -> MessageInfoV2:
    # Insertar mensaje con timestamp explícito (sin commit aún)
    query_msg = text("""INSERT INTO messages (chat_id, response, content, created_at, question_id, id, status,type) VALUES (:chat_id, :response, :content, :created_at, :question_id, :response_id, :status, :type) RETURNING id""")
    created_at = datetime.now(timezone.utc)
    response_id = uuid.uuid4()
    result_msg = await session.execute(query_msg, {
        "chat_id": chat_id,
        "response": response,
        "content": content,
        "created_at": created_at,
        "question_id": question_id,
        "response_id": response_id,
        "status": status.value,
        "type": type
    })
    message_id = result_msg.fetchone()[0]

    # Si sabemos que no habra errores, podemos commitear tranquilamente, pero generalmente no lo haremos
    if should_commit:
        await session.commit()

    returned_message = MessageInfoV2(
        id=str(message_id),
        chat_id=str(chat_id),
        response=response,
        content=content,
        created_at=created_at,
        status=status,

    )

    return returned_message

async def add_received_msg(session, chat_id, message : MessageInfoV2, should_commit=False, question_id=None):
    # Insertar mensaje con timestamp explícito (sin commit aún)
    print(f"Guardando mensaje en BD - Chat ID: {chat_id}, Response: {message.response}, Content: {message.content}, Question ID: {question_id}")
    query_msg = text("""INSERT INTO messages (id, chat_id, response, content, created_at, question_id, type) VALUES (:id, :chat_id, :response, :content, :created_at, :question_id, :type) RETURNING id""")
    created_at = datetime.now(timezone.utc)
    result_msg = await session.execute(query_msg, {
        "id": message.id,
        "chat_id": chat_id,
        "response": message.response,
        "content": message.content,
        "created_at": created_at,
        "question_id": question_id,
        "type": message.type
    })
    message_id = result_msg.fetchone()[0]

    # Si sabemos que no habra errores, podemos commitear tranquilamente, pero generalmente no lo haremos
    if should_commit:
        await session.commit()

    returned_message = MessageInfoV2(
        id=str(message_id),
        chat_id=str(chat_id),
        response=message.response,
        content=message.content,
        created_at=created_at,
        status=Estados.SUCCESS,
        type=message.type
    )

    return returned_message

async def get_chat_by_image_and_user(image_id, user, session) -> Optional[dict]:
    query_chat = text("SELECT * FROM chats WHERE image_id = :image_id and user_id = :user_id")
    result_chat = await session.execute(query_chat, {"image_id": image_id, "user_id": user})
    chat_db = result_chat.mappings().one_or_none()
    if not chat_db:
        return None
    print(f"Chat encontrado para la imagen ID {image_id} y usuario {user}: {chat_db}")
    return chat_db

async def create_chat_with_image(image_id, user, session, external_id=None) -> ChatModel | None:
    chat_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    query_chat = text("""INSERT INTO chats (id, user_id, image_id, created_at, external_id) VALUES (:id, :user_id, :image_id, :created_at, :external_id) ON CONFLICT (user_id, image_id) DO UPDATE SET id=chats.id RETURNING id""")
    result_chat = await session.execute(query_chat, {
    "id": chat_id,
    "user_id": user,
    "image_id": image_id,
    "created_at": created_at,
    "external_id": external_id
    })
    
    chat_id = result_chat.fetchone()[0]
    
    chat_db = ChatModel(
        id=str(chat_id),
        user_id=user,
        image_id=image_id,
        created_at=created_at,
        topic="",
        external_id=external_id,
        status="SUCCESS"
    )
    print(f"Chat creado para la imagen ID {image_id} y usuario {user}: {chat_db}")
    return chat_db

async def get_chat_with_id(chat_id, session, user) -> ChatModel | None:
    query_chat = text("SELECT * FROM chats WHERE id = :id and user_id = :user_id")
    result_chat = await session.execute(query_chat, {"id": chat_id, "user_id": user})
    chat_data = result_chat.mappings().one_or_none()
    print(f"Resultado de la consulta de chat con ID {chat_id}: {chat_data}")
    if not chat_data:
        return None
    else:
        return ChatModel(
            id=str(chat_data['id']),
            user_id=str(chat_data['user_id']),
            image_id=str(chat_data['image_id']),
            created_at=chat_data['created_at'],
            topic="",
            status=chat_data['status'],
            external_id=str(chat_data['external_id'])
        )

async def add_related_images_2_db(message_id, related_images, session, should_commit=False):
    final_related_images = []
    for img in related_images:
        print(f"Guardando imagen relacionada en la BD: {img}")
        query = text("INSERT INTO related_images (message_id, image_id, similarity) VALUES (:message_id, :image_id, :similarity) returning id")
        result = await session.execute(query, {"message_id": message_id, "image_id": img['id'], "similarity": img['similarity_score']})
        img_id = result.scalar_one()
        print(f"Inserted related image with ID: {img_id}")

    if should_commit:
        await session.commit()

async def get_internal_chat_id(session, external_id, user_id) -> ChatModel | None:
    query = text("SELECT * FROM chats WHERE external_id = :external_id and user_id = :user_id")
    result = await session.execute(query, {"external_id": external_id, "user_id": user_id})
    internal_id = result.mappings().one_or_none()
    print(f"Resultado de la consulta de equivalencia con external_id {external_id} y user_id {user_id}: {internal_id}")
    if not internal_id:
        return None
    else:
        return ChatModel(
            id=str(internal_id['id']),
            user_id=str(internal_id['user_id']),
            image_id=str(internal_id['image_id']),
            created_at=internal_id['created_at'],
            topic=internal_id['topic'],
            external_id=str(internal_id['external_id']),
            status=internal_id['status']
        )
