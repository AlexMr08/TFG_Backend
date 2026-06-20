import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.chat_processing import (
    get_chat_by_image_and_user,
    add_created_msg,
    create_chat_with_image,
    add_received_msg,
    get_chat_with_id,
    add_related_images_2_db,
    get_internal_chat_id
)
from app.clases.Enums import Estados, TipoMensaje
from app.clases.message_info import MessageInfoV2


# Un "fixture" de Pytest para darnos una sesión de base de datos falsa (mock) limpia en cada test
@pytest.fixture
def mock_session():
    session = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_get_chat_by_image_and_user_found(mock_session):
    # 1. Preparación (Arrange)
    mock_result = MagicMock()
    mock_data = {"id": "chat_123", "user_id": "user_1", "image_id": "img_1"}
    mock_result.mappings().one_or_none.return_value = mock_data
    
    mock_session.execute.return_value = mock_result

    # 2. Ejecución (Act)
    result = await get_chat_by_image_and_user("img_1", "user_1", mock_session)

    # 3. Verificación (Assert)
    assert result == mock_data
    # Verificamos que se ejecutó la query una vez
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_chat_by_image_and_user_not_found(mock_session):
    # 1. Preparación
    mock_result = MagicMock()
    mock_result.mappings().one_or_none.return_value = None  # Simulamos que no hay resultados
    mock_session.execute.return_value = mock_result

    # 2. Ejecución
    result = await get_chat_by_image_and_user("img_1", "user_1", mock_session)

    # 3. Verificación
    assert result is None


@pytest.mark.asyncio
async def test_add_created_msg_without_commit(mock_session):
    # 1. Preparación
    mock_result = MagicMock()
    # fetchone()[0] devuelve el ID retornado por la query
    mock_result.fetchone.return_value = ["mocked_message_id"]
    mock_session.execute.return_value = mock_result

    chat_id = "chat_123"
    response = True
    content = "Contenido original"

    # 2. Ejecución con should_commit=False (por defecto)
    msg_info = await add_created_msg(
        session=mock_session,
        chat_id=chat_id,
        response=response,
        content=content,
        status=Estados.SUCCESS,
        type=TipoMensaje.SIMPLE
    )

    # 3. Verificación
    assert msg_info.id == "mocked_message_id"
    assert msg_info.chat_id == chat_id
    assert msg_info.response == response
    assert msg_info.status == Estados.SUCCESS
    
    mock_session.execute.assert_called_once()
    # Comprobamos que el commit NO se ha llamado
    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_add_created_msg_with_commit(mock_session):
    # 1. Preparación
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ["mocked_message_id"]
    mock_session.execute.return_value = mock_result

    # 2. Ejecución con should_commit=True
    await add_created_msg(
        session=mock_session,
        chat_id="chat_123",
        response=True,
        content="Contenido",
        should_commit=True
    )

    # 3. Verificación
    # Comprobamos que el commit SÍ se ha llamado
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_add_received_msg(mock_session):
    # 1. Preparación
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ["mocked_received_msg_id"]
    mock_session.execute.return_value = mock_result

    input_msg = MessageInfoV2(
        id="orig_msg_1",
        chat_id="chat_123",
        response=False,
        content="Mensaje recibido del usuario",
        created_at=datetime.now(timezone.utc),
        status=Estados.SUCCESS,
        type=TipoMensaje.SIMPLE
    )

    # 2. Ejecución
    result = await add_received_msg(
        session=mock_session,
        chat_id="chat_123",
        message=input_msg,
        should_commit=True
    )

    # 3. Verificación
    assert result.id == "mocked_received_msg_id"
    assert result.content == "Mensaje recibido del usuario"
    assert result.response is False
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_create_chat_with_image(mock_session):
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ["new_chat_123"]
    mock_session.execute.return_value = mock_result

    chat = await create_chat_with_image(
        image_id="img_123",
        user="user_1",
        session=mock_session,
        external_id="ext_123"
    )

    assert chat.id == "new_chat_123"
    assert chat.user_id == "user_1"
    assert chat.image_id == "img_123"
    assert chat.external_id == "ext_123"
    assert chat.status == Estados.SUCCESS


@pytest.mark.asyncio
async def test_get_chat_with_id_found(mock_session):
    mock_result = MagicMock()
    mock_data = {
        "id": "chat_123",
        "user_id": "user_1",
        "image_id": "img_123",
        "created_at": datetime.now(timezone.utc),
        "status": "SUCCESS",
        "external_id": "ext_123"
    }
    mock_result.mappings().one_or_none.return_value = mock_data
    mock_session.execute.return_value = mock_result

    result = await get_chat_with_id("chat_123", mock_session, "user_1")

    assert result is not None
    assert result.id == "chat_123"
    assert result.image_id == "img_123"


@pytest.mark.asyncio
async def test_get_chat_with_id_not_found(mock_session):
    mock_result = MagicMock()
    mock_result.mappings().one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await get_chat_with_id("chat_invalid", mock_session, "user_1")
    assert result is None


@pytest.mark.asyncio
async def test_add_related_images_2_db(mock_session):
    mock_result = MagicMock()
    # En add_related_images_2_db se usa .scalar_one() dentro de un bucle.
    # .side_effect permite devolver un valor diferente por cada iteración.
    mock_result.scalar_one.side_effect = ["rel_1", "rel_2"]
    mock_session.execute.return_value = mock_result

    related_images = [
        {"id": "img_1", "similarity_score": 0.95},
        {"id": "img_2", "similarity_score": 0.88}
    ]

    await add_related_images_2_db(
        message_id="msg_123",
        related_images=related_images,
        session=mock_session,
        should_commit=True
    )

    # Verificamos que se ejecutó la query tantas veces como imágenes pasadas en el array
    assert mock_session.execute.call_count == 2
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_internal_chat_id_found(mock_session):
    mock_result = MagicMock()
    mock_data = {
        "id": "chat_internal_1",
        "user_id": "user_1",
        "image_id": "img_123",
        "created_at": datetime.now(timezone.utc),
        "topic": "Mi tema",
        "external_id": "ext_123",
        "status": "SUCCESS"
    }
    mock_result.mappings().one_or_none.return_value = mock_data
    mock_session.execute.return_value = mock_result

    result = await get_internal_chat_id(mock_session, "ext_123", "user_1")

    assert result is not None
    assert result.id == "chat_internal_1"
    assert result.external_id == "ext_123"


@pytest.mark.asyncio
async def test_get_internal_chat_id_not_found(mock_session):
    mock_result = MagicMock()
    mock_result.mappings().one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await get_internal_chat_id(mock_session, "ext_invalid", "user_1")
    assert result is None


#################################################################
# Additional edge-case and error-handling tests appended below
#################################################################


@pytest.mark.asyncio
async def test_add_created_msg_fetchone_none_raises(mock_session):
    """Si el resultado de fetchone() es None, se debe propagar el error (None[0] falla)."""
    mock_result = MagicMock()
    mock_result.fetchone.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(TypeError):
        await add_created_msg(
            session=mock_session,
            chat_id="chat_123",
            response=True,
            content="Contenido",
        )


@pytest.mark.asyncio
async def test_add_received_msg_execute_raises_no_commit(mock_session):
    """Si la ejecución en BD lanza una excepción, no debe hacer commit."""
    mock_session.execute.side_effect = Exception("DB error")

    input_msg = MessageInfoV2(
        id="orig_msg_2",
        chat_id="chat_123",
        response=False,
        content="Mensaje",
        created_at=datetime.now(timezone.utc),
        status=Estados.SUCCESS,
        type=TipoMensaje.SIMPLE,
    )

    with pytest.raises(Exception):
        await add_received_msg(
            session=mock_session,
            chat_id="chat_123",
            message=input_msg,
            should_commit=True,
        )

    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_add_related_images_empty_list_commit_called(mock_session):
    """Si la lista de imágenes relacionadas está vacía, no se ejecutan inserts, pero commit sí si se pide."""
    # Aseguramos que execute no hará nada
    mock_session.execute.return_value = MagicMock()

    await add_related_images_2_db(
        message_id="msg_empty",
        related_images=[],
        session=mock_session,
        should_commit=True,
    )

    mock_session.execute.assert_not_called()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_add_received_msg_returned_created_at_is_utc(mock_session):
    """Comprobar que el `created_at` que devuelve `add_received_msg` está en UTC."""
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ["mocked_received_msg_id_2"]
    mock_session.execute.return_value = mock_result

    input_msg = MessageInfoV2(
        id="orig_msg_3",
        chat_id="chat_123",
        response=False,
        content="Mensaje UTC",
        created_at=datetime.now(timezone.utc),
        status=Estados.SUCCESS,
        type=TipoMensaje.SIMPLE,
    )

    result = await add_received_msg(
        session=mock_session,
        chat_id="chat_123",
        message=input_msg,
        should_commit=False,
    )

    assert result.created_at.tzinfo is not None
    assert result.created_at.tzinfo == timezone.utc