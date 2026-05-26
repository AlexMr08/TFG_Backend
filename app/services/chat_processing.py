from datetime import datetime, timezone
import uuid
from typing import Optional

from sqlalchemy import text

from app.clases.ChatModel import ChatModel
from app.clases.Enums import Estados, TipoMensaje
from app.clases.MessageInfo import MessageInfoV2


async def add_created_msg(
    session,
    chat_id,
    response,
    content,
    should_commit: bool = False,
    question_id=None,
    status: Estados = Estados.SUCCESS,
    type: TipoMensaje = TipoMensaje.SIMPLE,
) -> MessageInfoV2:
    query_msg = text(
        """INSERT INTO messages (chat_id, response, content, created_at, question_id, id, status,type)
        VALUES (:chat_id, :response, :content, :created_at, :question_id, :response_id, :status, :type)
        RETURNING id"""
    )
    created_at = datetime.now(timezone.utc)
    response_id = uuid.uuid4()
    result_msg = await session.execute(
        query_msg,
        {
            "chat_id": chat_id,
            "response": response,
            "content": content,
            "created_at": created_at,
            "question_id": question_id,
            "response_id": response_id,
            "status": status.value,
            "type": type,
        },
    )
    message_id = result_msg.fetchone()[0]

    if should_commit:
        await session.commit()

    return MessageInfoV2(
        id=str(message_id),
        chat_id=str(chat_id),
        response=response,
        content=content,
        created_at=created_at,
        status=status,
    )


async def add_received_msg(session, chat_id, message: MessageInfoV2, should_commit: bool = False, question_id=None):
    print(
        f"Guardando mensaje en BD - Chat ID: {chat_id}, Response: {message.response}, "
        f"Content: {message.content}, Question ID: {question_id}"
    )
    query_msg = text(
        """INSERT INTO messages (id, chat_id, response, content, created_at, question_id, type)
        VALUES (:id, :chat_id, :response, :content, :created_at, :question_id, :type)
        RETURNING id"""
    )
    created_at = datetime.now(timezone.utc)
    result_msg = await session.execute(
        query_msg,
        {
            "id": message.id,
            "chat_id": chat_id,
            "response": message.response,
            "content": message.content,
            "created_at": created_at,
            "question_id": question_id,
            "type": message.type,
        },
    )
    message_id = result_msg.fetchone()[0]

    if should_commit:
        await session.commit()

    return MessageInfoV2(
        id=str(message_id),
        chat_id=str(chat_id),
        response=message.response,
        content=message.content,
        created_at=created_at,
        status=Estados.SUCCESS,
        type=message.type,
    )


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
    query_chat = text(
        """INSERT INTO chats (id, user_id, image_id, created_at, external_id)
        VALUES (:id, :user_id, :image_id, :created_at, :external_id)
        ON CONFLICT (user_id, image_id) DO UPDATE SET id=chats.id
        RETURNING id"""
    )
    result_chat = await session.execute(
        query_chat,
        {
            "id": chat_id,
            "user_id": user,
            "image_id": image_id,
            "created_at": created_at,
            "external_id": external_id,
        },
    )

    chat_id = result_chat.fetchone()[0]

    chat_db = ChatModel(
        id=str(chat_id),
        user_id=user,
        image_id=image_id,
        created_at=created_at,
        topic="",
        external_id=external_id,
        status="SUCCESS",
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
    return ChatModel(
        id=str(chat_data["id"]),
        user_id=str(chat_data["user_id"]),
        image_id=str(chat_data["image_id"]),
        created_at=chat_data["created_at"],
        topic="",
        status=chat_data["status"],
        external_id=str(chat_data["external_id"]),
    )


async def add_related_images_2_db(message_id, related_images, session, should_commit: bool = False) -> None:
    for img in related_images:
        print(f"Guardando imagen relacionada en la BD: {img}")
        query = text(
            "INSERT INTO related_images (message_id, image_id, similarity) "
            "VALUES (:message_id, :image_id, :similarity) returning id"
        )
        result = await session.execute(
            query,
            {
                "message_id": message_id,
                "image_id": img["id"],
                "similarity": img["similarity_score"],
            },
        )
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
    return ChatModel(
        id=str(internal_id["id"]),
        user_id=str(internal_id["user_id"]),
        image_id=str(internal_id["image_id"]),
        created_at=internal_id["created_at"],
        topic=internal_id["topic"],
        external_id=str(internal_id["external_id"]),
        status=internal_id["status"],
    )
