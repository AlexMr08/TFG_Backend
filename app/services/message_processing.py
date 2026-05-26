from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import partial
import io
import os
from typing import Optional, Callable

import numpy as np
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from app.services.chat_processing import add_related_images_2_db
from app.routers.images import get_image_with_id
from app.services.image_processing import encode_image_bytes, encode_image_file
from app.clases.ChatModel import ChatModel
from app.clases.Enums import Estados, TipoMensaje
from app.clases.ImageModel import ImageModel
from app.clases.MessageInfo import MessageInfoV2
from app.clases.Responses import CompleteChatResponse
from app.db.database import AsyncSessionLocal


def _is_placeholder_or_empty_message(content: str | None) -> bool:
    if not content:
        return True
    normalized = content.strip().lower()
    if not normalized:
        return True
    return normalized == "estamos procesando tu solicitud..."


async def build_chat_history_for_vllm(
    session: AsyncSession,
    chat_id: str,
    exclude_message_id: Optional[str] = None,
    max_messages: int = 12,
) -> list[dict]:
    """Build OpenAI-style chat history from persisted messages for vLLM."""
    query_history = text(
        """
        SELECT id, response, content, status
        FROM messages
        WHERE chat_id = :chat_id
        ORDER BY created_at ASC
        """
    )
    result = await session.execute(query_history, {"chat_id": chat_id})
    rows = result.mappings().all()

    history: list[dict] = []
    for row in rows:
        row_id = str(row["id"])
        if exclude_message_id and row_id == str(exclude_message_id):
            continue

        status = row.get("status")
        if status == Estados.WAITING.value:
            continue

        content = row.get("content")
        if _is_placeholder_or_empty_message(content):
            continue

        role = "assistant" if row["response"] else "user"
        history.append({"role": role, "content": content})

    if max_messages > 0 and len(history) > max_messages:
        history = history[-max_messages:]

    return history


async def buscar_imagenes_similares(
    image: Image.Image,
    n_results: int = 5,
    *,
    session: AsyncSession,
    user: str,
    collection,
    embedder,
) -> list[dict]:
    """
    Función reutilizable que vectoriza una imagen y busca similares en ChromaDB.

    Args:
        image: Objeto PIL.Image
        n_results: Número de resultados a devolver
        session: Sesión de base de datos
        user: ID del usuario
        collection: Colección ChromaDB inicializada
        embedder: Modelo de embeddings

    Returns:
        Lista de diccionarios con resultados formateados
    """

    query_vector = embedder.encode(image, normalize_embeddings=True).tolist()

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
    }

    results = collection.query(**query_params)
    total_results = len(results["ids"][0]) if results.get("ids") and results["ids"] else 0
    print(f"[buscar_imagenes_similares] Resultados en ChromaDB: {total_results} (n_results solicitado: {n_results})")
    if total_results == 0:
        print("[buscar_imagenes_similares] No se encontraron resultados en ChromaDB para esta imagen.")

    response_data = []

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distancia = results["distances"][0][i]
        chroma_id = results["ids"][0][i]
        similarity_score = max(0, 100 * (1 - distancia))
        image = await get_image_with_id(chroma_id, session=session, user=user)
        print(f"Resultado {i+1}: ID={chroma_id}, Distancia={distancia}, Metadata={meta}, Imagen encontrada: {image}")
        if image:
            relative_path = image.image_url
            image_url = f"/art/{relative_path}"
            is_identical = False
            if round(similarity_score, 2) >= 100.0:
                is_identical = True

            response_data.append(
                {
                    "id": chroma_id,
                    "name": image.name,
                    "artist": image.artist,
                    "style": image.style,
                    "genre": image.genre,
                    "similarity_score": round(similarity_score, 2),
                    "distance": round(distancia, 4),
                    "image_url": image_url,
                    "is_identical": is_identical,
                }
            )
    print(f"[buscar_imagenes_similares] Respuesta formateada con {len(response_data)} resultados similares.")

    return response_data


def analizar_imagen_con_qwen_requests(
    image_bytes: bytes,
    prompt: str,
    results: list,
    *,
    qwen_client,
    qwen_model: str,
    history_messages: Optional[list[dict]] = None,
    force_identical_id: Optional[str] = None,
) -> str:
    """Analiza una imagen usando QWEN en vllm (localhost:8000)."""
    image_base64 = encode_image_bytes(image_bytes, max_size=(1024, 1024), quality=85)

    system_prompt_tecnico = (
        "You are a computer vision system designed for technical image analysis. "
        "Your goal is to provide objective, detailed visual descriptions. "
        "Identify artistic styles, techniques, palette, and composition. "
        "Do not refuse analysis based on copyright or safety filters for fictional characters. "
        "Be explicit and descriptive.\n\n"
        "IMPORTANT: ALWAYS respond in the SAME LANGUAGE as the user's question. "
        "You can ONLY answer questions that are directly related to the visual content of the image provided. "
        "If the user asks about topics unrelated to the image (such as general knowledge, math problems, coding, history, "
        "or any other subject not visible in the image), you must politely decline in the same language. Response only with the response to the question asked, do not provide additional information beyond what is asked. Keep answers concise and direct."
    )

    used_reference = None
    if force_identical_id and results:
        used_reference = next((r for r in results if str(r.get("id")) == str(force_identical_id)), None)

    if used_reference:
        try:
            image_path = used_reference["image_url"].replace("/view/image_thumbnail/", "").replace("/art/", "")
            full_path = os.path.join(config.CARPETA_IMAGENES, image_path)
            if os.path.exists(full_path):
                image_base64 = encode_image_file(full_path, max_size=(1024, 1024), quality=85)
        except Exception as e:
            print(f"WARN: No se pudo usar la imagen de la referencia idéntica como principal: {e}")

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]

    if used_reference:
        ref_info = (
            f"\n\nUsando obra idéntica como sujeto principal: ID {used_reference['id']}, "
            f"Artist: {used_reference.get('artist')}, Style: {used_reference.get('style')}, "
            f"Genre: {used_reference.get('genre')}, Similarity: {used_reference.get('similarity_score')}%\n"
        )
        content.append({"type": "text", "text": ref_info})

    if results:
        similar_info = "\n\nSimilar artworks found:\n"
        filtered = [r for r in results if not (used_reference and str(r.get("id")) == str(used_reference.get("id")))]
        for idx, res in enumerate(filtered, 1):
            similar_info += (
                f"{idx}. Artist: {res['artist']}, Style: {res['style']}, "
                f"Genre: {res['genre']}, Similarity: {res['similarity_score']}%\n"
            )
        if filtered:
            content.append({"type": "text", "text": similar_info})

        added = 0
        for idx, res in enumerate(results, 1):
            if added >= 3:
                break
            if used_reference and str(res.get("id")) == str(used_reference.get("id")):
                continue
            try:
                image_path = res["image_url"].replace("/view/image_thumbnail/", "").replace("/art/", "")
                full_path = os.path.join(config.CARPETA_IMAGENES, image_path)

                if os.path.exists(full_path):
                    similar_base64 = encode_image_file(full_path, max_size=(512, 512), quality=75)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{similar_base64}"}})
                    added += 1
            except Exception as e:
                print(f"ERROR convirtiendo imagen similar {idx}: {e}")

    messages_payload = [
        {
            "role": "system",
            "content": system_prompt_tecnico,
        }
    ]
    if force_identical_id:
        instruction_text = (
            f"NOTA: Se ha encontrado una obra idéntica en la base de datos con ID {force_identical_id}. "
            "Asume que es la MISMA obra y basa TODO tu análisis únicamente en la obra referenciada. "
            "No inventes diferencias ni atribuciones. Al final indica claramente 'Obra idéntica: "
            + str(force_identical_id)
            + "'."
        )
        messages_payload.append({"role": "system", "content": instruction_text})
    include_history = False
    if include_history and history_messages:
        messages_payload.extend(history_messages)
    messages_payload.append({"role": "user", "content": content})

    response = qwen_client.chat.completions.create(
        model=qwen_model,
        messages=messages_payload,
        max_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        seed=np.random.randint(0, 2**31),
    )

    return response.choices[0].message.content


def preguntar_a_vllm(
    image_bytes: bytes,
    prompt: str,
    *,
    qwen_client,
    qwen_model: str,
    history_messages: Optional[list[dict]] = None,
) -> str:
    """Analiza una imagen usando QWEN en vllm (localhost:8000)."""
    image_base64 = encode_image_bytes(image_bytes, max_size=(1024, 1024), quality=85)

    system_prompt_tecnico = (
        "You are a computer vision system designed for technical image analysis. "
        "Your goal is to provide objective, detailed visual descriptions. "
        "Do not refuse analysis based on copyright or safety filters for fictional characters. "
        "Be explicit and descriptive.\n\n"
        "IMPORTANT: ALWAYS respond in the SAME LANGUAGE as the user's question. "
        "You can ONLY answer questions that are directly related to the visual content of the image provided. "
        "If the user asks about topics unrelated to the image (such as general knowledge, math problems, coding, history, "
        "or any other subject not visible in the image), you must politely decline in the same language. "
    )

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]

    messages_payload = [
        {
            "role": "system",
            "content": system_prompt_tecnico,
        }
    ]
    include_history = False
    if include_history and history_messages:
        messages_payload.extend(history_messages)
    messages_payload.append({"role": "user", "content": content})

    response = qwen_client.chat.completions.create(
        model=qwen_model,
        messages=messages_payload,
        max_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        seed=np.random.randint(0, 2**31),
    )

    return response.choices[0].message.content


async def procesar_mensaje_complejo_recibido(
    receive_message: MessageInfoV2,
    new_message: MessageInfoV2,
    chat: ChatModel,
    image: ImageModel,
    *,
    qwen_client,
    qwen_model: str,
    collection,
    embedder,
    get_queue: Callable[[str], asyncio.Queue],
) -> None:
    async with AsyncSessionLocal() as session:
        try:
            print(
                "✅ Procesando mensaje recibido con ID "
                f"{receive_message.id}, y su respuesta {new_message.id} "
                f"para el chat ID {chat.id} y la imagen ID {image.id}"
            )
            history_messages = await build_chat_history_for_vllm(
                session=session,
                chat_id=str(chat.id),
                exclude_message_id=str(receive_message.id),
                max_messages=12,
            )
            full_path = os.path.join(config.CARPETA_IMAGENES, image.image_url)
            with open(full_path, "rb") as img_file:
                current_image = img_file.read()

            pil_image = Image.open(io.BytesIO(current_image))

            print("[procesar_mensaje_complejo_recibido] Ejecutando búsqueda de imágenes similares...")
            results = await buscar_imagenes_similares(
                pil_image,
                n_results=3,
                session=session,
                user=chat.user_id,
                collection=collection,
                embedder=embedder,
            )
            print(f"[procesar_mensaje_complejo_recibido] Búsqueda completada. Resultados útiles: {len(results)}")

            loop = asyncio.get_event_loop()
            identical_entry = None
            for r in results:
                if r.get("is_identical") or float(r.get("similarity_score", 0)) >= 100.0:
                    print(
                        "Obra idéntica encontrada en resultados similares: ID "
                        f"{r.get('id')}, Artist: {r.get('artist')}, "
                        f"Similarity: {r.get('similarity_score')}%"
                    )
                    identical_entry = r
                    break

            identical_id = identical_entry["id"] if identical_entry else None

            qwen_description = await loop.run_in_executor(
                None,
                partial(
                    analizar_imagen_con_qwen_requests,
                    current_image,
                    """Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. 
                    El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, 
                    Comparacion rapida con las similares en caso de haber.""",
                    results,
                    qwen_client=qwen_client,
                    qwen_model=qwen_model,
                    history_messages=None,
                    force_identical_id=identical_id,
                ),
            )
            status = Estados.SUCCESS

        except Exception as e:
            print(f"❌ Error al procesar imagen: {e}")
            qwen_description = "Error al analizar la imagen con QWEN."
            results = []
            status = Estados.ERROR

        try:
            query_update = text(
                """
                UPDATE messages 
                SET content = :content, status = :status, created_at = :created_at
                WHERE id = :message_id
                """
            )
            new_time = datetime.now(timezone.utc)
            await session.execute(
                query_update,
                {
                    "content": qwen_description,
                    "status": status.value,
                    "created_at": new_time,
                    "message_id": new_message.id,
                },
            )

            if results:
                await add_related_images_2_db(
                    message_id=new_message.id,
                    related_images=results,
                    session=session,
                    should_commit=False,
                )

            await session.commit()
            print(f"Procesamiento completado para mensaje {new_message.id}")
            queue = get_queue(chat.user_id)
            result = MessageInfoV2(
                id=str(new_message.id),
                chat_id=str(new_message.chat_id),
                response=new_message.response,
                content=qwen_description,
                created_at=new_message.created_at,
                status=status,
                related_images=results,
                question_id=new_message.question_id,
            )
            fin = CompleteChatResponse(
                message_info=result,
                request_info=receive_message,
                chat_info=chat,
                image_info=image,
            )
            print(f"Enviando resultado al cliente a través de SSE: {fin}")
            await queue.put(fin.model_dump_json())
        except Exception as e:
            print(f"Error al actualizar la base de datos: {e}")
            await session.rollback()


async def procesar_mensaje_simple_recibido(
    receive_message: MessageInfoV2,
    new_message: MessageInfoV2,
    chat: ChatModel,
    image: ImageModel,
    *,
    qwen_client,
    qwen_model: str,
    collection,
    embedder,
    get_queue: Callable[[str], asyncio.Queue],
) -> None:
    async with AsyncSessionLocal() as session:
        try:
            print(
                "Procesando mensaje SIMPLE recibido con ID "
                f"{receive_message.id}, y su respuesta {new_message.id} "
                f"para el chat ID {chat.id} y la imagen ID {image.id}"
            )
            history_messages = await build_chat_history_for_vllm(
                session=session,
                chat_id=str(chat.id),
                exclude_message_id=str(receive_message.id),
                max_messages=12,
            )
            full_path = os.path.join(config.CARPETA_IMAGENES, image.image_url)
            with open(full_path, "rb") as img_file:
                current_image = img_file.read()

            pil_image = Image.open(io.BytesIO(current_image))
            print("[procesar_mensaje_simple_recibido] Flujo SIMPLE: no se ejecuta buscar_imagenes_similares.")

            loop = asyncio.get_event_loop()
            qwen_description = await loop.run_in_executor(
                None,
                partial(
                    preguntar_a_vllm,
                    current_image,
                    receive_message.content,
                    qwen_client=qwen_client,
                    qwen_model=qwen_model,
                    history_messages=history_messages,
                ),
            )
            status = Estados.SUCCESS

        except Exception as e:
            print(f"Error al procesar imagen: {e}")
            qwen_description = "Error al analizar la imagen con QWEN."
            status = Estados.ERROR

        try:
            query_update = text(
                """
                UPDATE messages 
                SET content = :content, status = :status, created_at = :created_at
                WHERE id = :message_id
                """
            )
            new_time = datetime.now(timezone.utc)
            await session.execute(
                query_update,
                {
                    "content": qwen_description,
                    "status": status.value,
                    "message_id": new_message.id,
                    "created_at": new_time,
                },
            )

            await session.commit()
            print(f" Procesamiento completado para mensaje {new_message.id}")
            queue = get_queue(chat.user_id)
            result = MessageInfoV2(
                id=str(new_message.id),
                chat_id=str(new_message.chat_id),
                response=new_message.response,
                content=qwen_description,
                created_at=new_time,
                status=status,
                question_id=new_message.question_id,
            )
            fin = CompleteChatResponse(
                message_info=result,
                request_info=receive_message,
                chat_info=chat,
                image_info=image,
            )
            print(f"Enviando resultado al cliente a través de SSE: {fin}")
            await queue.put(fin.model_dump_json())

        except Exception as e:
            print(f"Error al actualizar la base de datos: {e}")
            await session.rollback()


def retry_message_processing(
    receive_message: MessageInfoV2,
    new_message: MessageInfoV2,
    chat: ChatModel,
    image: ImageModel,
    *,
    qwen_client,
    qwen_model: str,
    collection,
    embedder,
    get_queue: Callable[[str], asyncio.Queue],
) -> None:
    print(f"Reintentando procesamiento para mensaje {new_message.id}")
    if new_message.type == TipoMensaje.SIMPLE:
        asyncio.create_task(
            procesar_mensaje_simple_recibido(
                receive_message,
                new_message,
                chat,
                image,
                qwen_client=qwen_client,
                qwen_model=qwen_model,
                collection=collection,
                embedder=embedder,
                get_queue=get_queue,
            )
        )
    else:
        asyncio.create_task(
            procesar_mensaje_complejo_recibido(
                receive_message,
                new_message,
                chat,
                image,
                qwen_client=qwen_client,
                qwen_model=qwen_model,
                collection=collection,
                embedder=embedder,
                get_queue=get_queue,
            )
        )
