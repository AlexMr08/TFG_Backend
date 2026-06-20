import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import io
import os
import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
from PIL import Image

from app.services.message_processing import (
    _is_placeholder_or_empty_message,
    build_chat_history_for_vllm,
    buscar_imagenes_similares,
    analizar_imagen_con_qwen_requests,
    preguntar_a_vllm,
    procesar_mensaje_complejo_recibido,
    procesar_mensaje_simple_recibido,
    retry_message_processing,
)
from app.clases.Enums import Estados, TipoMensaje
from app.clases.message_info import MessageInfoV2
from app.clases.chat_model import ChatModel
from app.clases.image_model import ImageModel
from datetime import datetime, timezone

def make_message_info(msg_id: str = "msg-1", content: str = "test content") -> MessageInfoV2:
    return MessageInfoV2(
        id=msg_id,
        chat_id="chat-1",
        response=False,
        content=content,
        created_at=datetime.now(timezone.utc),
        status=Estados.SUCCESS,
        type=TipoMensaje.COMPLETE,
    )

def make_chat() -> ChatModel:
    return ChatModel(
        id="chat-1",
        user_id="user-1",
        image_id="img-1",          # one of image_id OR collection_id required
        collection_id=None,
        created_at=datetime.now(timezone.utc),
    )

def make_image_model(image_url: str = "artist/obra.jpg") -> ImageModel:
    return ImageModel(
        id="img-1",
        image_url=image_url,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=100, height=100, color=(200, 100, 50)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def make_jpeg_bytes(width=100, height=100) -> bytes:
    img = make_pil_image(width, height)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_row(
    row_id="msg-1",
    response=False,
    content="Hola",
    status=Estados.SUCCESS.value,
):
    row = {
        "id": row_id,
        "response": response,
        "content": content,
        "status": status,
    }
    m = MagicMock()
    m.__getitem__ = lambda self, k: row[k]
    m.get = lambda k, default=None: row.get(k, default)
    return m


class AsyncMappingsAll:
    """Mock para result.mappings().all()"""
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


def make_session_with_rows(rows):
    session = AsyncMock()
    session.execute = AsyncMock(return_value=AsyncMappingsAll(rows))
    return session


def make_message_info(
    msg_id="msg-1",
    chat_id="chat-1",
    response=False,
    content="Hola",
    msg_type=TipoMensaje.SIMPLE,
):
    m = MagicMock()
    m.id = msg_id
    m.chat_id = chat_id
    m.response = response
    m.content = content
    m.type = msg_type

    m.created_at = datetime.now(timezone.utc)
    m.status = Estados.WAITING
    m.question_id = None
    return m


def make_chat(chat_id="chat-1", user_id="user-1"):
    c = MagicMock()
    c.id = chat_id
    c.user_id = user_id
    return c


def make_image_model(image_id="img-1", image_url="artist/obra.jpg"):
    i = MagicMock()
    i.id = image_id
    i.image_url = image_url
    return i


# ===========================================================================
# _is_placeholder_or_empty_message
# ===========================================================================

def test_placeholder_none():
    assert _is_placeholder_or_empty_message(None) is True


def test_placeholder_empty_string():
    assert _is_placeholder_or_empty_message("") is True


def test_placeholder_whitespace_only():
    assert _is_placeholder_or_empty_message("   ") is True


def test_placeholder_exact_string():
    assert _is_placeholder_or_empty_message("estamos procesando tu solicitud...") is True


def test_placeholder_case_insensitive():
    assert _is_placeholder_or_empty_message("ESTAMOS PROCESANDO TU SOLICITUD...") is True


def test_placeholder_normal_message():
    assert _is_placeholder_or_empty_message("¿De qué trata esta obra?") is False


def test_placeholder_partial_match_is_not_placeholder():
    assert _is_placeholder_or_empty_message("procesando") is False


# ===========================================================================
# build_chat_history_for_vllm
# ===========================================================================

@pytest.mark.asyncio
async def test_build_chat_history_user_and_assistant_roles():
    """Asigna rol 'user' o 'assistant' según el campo response."""
    rows = [
        make_row("1", response=False, content="Pregunta", status=Estados.SUCCESS.value),
        make_row("2", response=True, content="Respuesta", status=Estados.SUCCESS.value),
    ]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1")

    assert history[0] == {"role": "user", "content": "Pregunta"}
    assert history[1] == {"role": "assistant", "content": "Respuesta"}


@pytest.mark.asyncio
async def test_build_chat_history_excludes_message_id():
    """Excluye el mensaje con el ID indicado."""
    rows = [
        make_row("1", content="Mensaje A"),
        make_row("2", content="Mensaje B"),
    ]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1", exclude_message_id="1")

    assert len(history) == 1
    assert history[0]["content"] == "Mensaje B"


@pytest.mark.asyncio
async def test_build_chat_history_skips_waiting_status():
    """Omite mensajes con estado WAITING."""
    rows = [
        make_row("1", content="Visible", status=Estados.SUCCESS.value),
        make_row("2", content="Esperando", status=Estados.WAITING.value),
    ]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1")

    assert len(history) == 1
    assert history[0]["content"] == "Visible"


@pytest.mark.asyncio
async def test_build_chat_history_skips_placeholder_content():
    """Omite mensajes con contenido placeholder."""
    rows = [
        make_row("1", content="Estamos procesando tu solicitud..."),
        make_row("2", content="Mensaje real"),
    ]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1")

    assert len(history) == 1
    assert history[0]["content"] == "Mensaje real"


@pytest.mark.asyncio
async def test_build_chat_history_respects_max_messages():
    """Devuelve los últimos max_messages mensajes."""
    rows = [make_row(str(i), content=f"msg-{i}") for i in range(20)]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1", max_messages=5)

    assert len(history) == 5
    assert history[-1]["content"] == "msg-19"


@pytest.mark.asyncio
async def test_build_chat_history_max_messages_zero_returns_all():
    """max_messages=0 desactiva el límite."""
    rows = [make_row(str(i), content=f"msg-{i}") for i in range(15)]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1", max_messages=0)

    assert len(history) == 15


@pytest.mark.asyncio
async def test_build_chat_history_empty_db():
    """Sin mensajes devuelve lista vacía."""
    session = make_session_with_rows([])

    history = await build_chat_history_for_vllm(session, "chat-1")

    assert history == []


@pytest.mark.asyncio
async def test_build_chat_history_skips_none_content():
    """Omite filas con content=None."""
    rows = [
        make_row("1", content=None),
        make_row("2", content="Válido"),
    ]
    session = make_session_with_rows(rows)

    history = await build_chat_history_for_vllm(session, "chat-1")

    assert len(history) == 1
    assert history[0]["content"] == "Válido"


# ===========================================================================
# buscar_imagenes_similares
# ===========================================================================

@pytest.mark.asyncio
async def test_buscar_imagenes_similares_returns_formatted_list():
    """Devuelve lista con los campos esperados cuando hay resultados."""
    pil_img = make_pil_image()

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.1] * 512))

    image_mock = MagicMock()
    image_mock.image_url = "artist/obra.jpg"
    image_mock.name = "Obra"
    image_mock.artist = "Artista"
    image_mock.style = "Impresionismo"
    image_mock.genre = "Paisaje"

    collection = MagicMock()
    collection.query = MagicMock(return_value={
        "ids": [["id-1"]],
        "distances": [[0.1]],
        "metadatas": [[{}]],
    })

    session = AsyncMock()

    with patch("app.services.message_processing.get_image_with_id", new_callable=AsyncMock, return_value=image_mock):
        results = await buscar_imagenes_similares(
            pil_img,
            n_results=1,
            session=session,
            user="user-1",
            collection=collection,
            embedder=embedder,
        )

    assert len(results) == 1
    r = results[0]
    assert r["id"] == "id-1"
    assert r["name"] == "Obra"
    assert r["artist"] == "Artista"
    assert "similarity_score" in r
    assert "distance" in r
    assert r["image_url"] == "/art/artist/obra.jpg"


@pytest.mark.asyncio
async def test_buscar_imagenes_similares_no_results():
    """Sin resultados en ChromaDB devuelve lista vacía."""
    pil_img = make_pil_image()

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.1] * 512))

    collection = MagicMock()
    collection.query = MagicMock(return_value={"ids": [[]], "distances": [[]], "metadatas": [[]]})

    session = AsyncMock()

    results = await buscar_imagenes_similares(
        pil_img,
        n_results=5,
        session=session,
        user="user-1",
        collection=collection,
        embedder=embedder,
    )

    assert results == []


@pytest.mark.asyncio
async def test_buscar_imagenes_similares_identical_score_marks_flag():
    """Score >= 100 activa is_identical=True."""
    pil_img = make_pil_image()

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.0] * 512))

    image_mock = MagicMock()
    image_mock.image_url = "a/b.jpg"
    image_mock.name = "Obra"
    image_mock.artist = "A"
    image_mock.style = "S"
    image_mock.genre = "G"

    collection = MagicMock()
    collection.query = MagicMock(return_value={
        "ids": [["id-1"]],
        "distances": [[0.0]],   # distancia 0 → score 100
        "metadatas": [[{}]],
    })

    session = AsyncMock()

    with patch("app.services.message_processing.get_image_with_id", new_callable=AsyncMock, return_value=image_mock):
        results = await buscar_imagenes_similares(
            pil_img, n_results=1,
            session=session, user="u",
            collection=collection, embedder=embedder,
        )

    assert results[0]["is_identical"] is True


@pytest.mark.asyncio
async def test_buscar_imagenes_similares_image_not_found_in_db():
    """Si get_image_with_id devuelve None, el resultado se omite."""
    pil_img = make_pil_image()

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.1] * 512))

    collection = MagicMock()
    collection.query = MagicMock(return_value={
        "ids": [["id-1"]],
        "distances": [[0.2]],
        "metadatas": [[{}]],
    })

    session = AsyncMock()

    with patch("app.services.message_processing.get_image_with_id", new_callable=AsyncMock, return_value=None):
        results = await buscar_imagenes_similares(
            pil_img, n_results=1,
            session=session, user="u",
            collection=collection, embedder=embedder,
        )

    assert results == []


# ===========================================================================
# analizar_imagen_con_qwen_requests
# ===========================================================================

def make_qwen_client(response_text="Análisis de arte"):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create = MagicMock(return_value=MagicMock(choices=[choice]))
    return client


def test_analizar_imagen_con_qwen_returns_string():
    """Devuelve el texto de la respuesta del modelo."""
    client = make_qwen_client("Descripción detallada")
    result = analizar_imagen_con_qwen_requests(
        make_jpeg_bytes(),
        "Describe esta obra",
        [],
        qwen_client=client,
        qwen_model="qwen-vl",
    )
    assert result == "Descripción detallada"


def test_analizar_imagen_con_qwen_calls_model_once():
    """Solo hace una llamada al modelo."""
    client = make_qwen_client()
    analizar_imagen_con_qwen_requests(
        make_jpeg_bytes(), "prompt", [],
        qwen_client=client, qwen_model="qwen-vl",
    )
    client.chat.completions.create.assert_called_once()


def test_analizar_imagen_con_qwen_includes_similar_results_in_payload():
    """Los resultados similares se adjuntan al contenido del mensaje."""
    client = make_qwen_client()
    results = [
        {"id": "r1", "artist": "Goya", "style": "Romántico", "genre": "Historia",
         "similarity_score": 85.0, "image_url": "/art/goya/obra.jpg", "is_identical": False}
    ]
    analizar_imagen_con_qwen_requests(
        make_jpeg_bytes(), "Describe", results,
        qwen_client=client, qwen_model="qwen-vl",
    )
    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") or call_kwargs[0][0]
    payload_str = str(messages)
    assert "Goya" in payload_str


def test_analizar_imagen_con_qwen_force_identical_id_sets_instruction():
    """Con force_identical_id, agrega instrucción de obra idéntica al payload."""
    client = make_qwen_client()
    results = [
        {"id": "img-42", "artist": "Velázquez", "style": "Barroco", "genre": "Retrato",
         "similarity_score": 100.0, "image_url": "/art/v/obra.jpg", "is_identical": True}
    ]
    analizar_imagen_con_qwen_requests(
        make_jpeg_bytes(), "prompt", results,
        qwen_client=client, qwen_model="qwen-vl",
        force_identical_id="img-42",
    )
    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
    payload_str = str(messages)
    assert "img-42" in payload_str


def test_analizar_imagen_con_qwen_without_results():
    """Funciona correctamente con lista de resultados vacía."""
    client = make_qwen_client("OK")
    result = analizar_imagen_con_qwen_requests(
        make_jpeg_bytes(), "Describe", [],
        qwen_client=client, qwen_model="qwen-vl",
    )
    assert result == "OK"


# ===========================================================================
# preguntar_a_vllm
# ===========================================================================

def test_preguntar_a_vllm_returns_model_response():
    """Devuelve el texto generado por el modelo."""
    client = make_qwen_client("Respuesta simple")
    result = preguntar_a_vllm(
        make_jpeg_bytes(), "¿Qué colores predominan?",
        qwen_client=client, qwen_model="qwen-vl",
    )
    assert result == "Respuesta simple"


def test_preguntar_a_vllm_calls_model_once():
    """Solo hace una llamada al modelo."""
    client = make_qwen_client()
    preguntar_a_vllm(
        make_jpeg_bytes(), "prompt",
        qwen_client=client, qwen_model="qwen-vl",
    )
    client.chat.completions.create.assert_called_once()


def test_preguntar_a_vllm_sends_image_in_payload():
    """El payload enviado al modelo incluye la imagen en base64."""
    client = make_qwen_client()
    preguntar_a_vllm(
        make_jpeg_bytes(), "Describe",
        qwen_client=client, qwen_model="qwen-vl",
    )
    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
    payload_str = str(messages)
    assert "image_url" in payload_str
    assert "base64" in payload_str


def test_preguntar_a_vllm_with_history_does_not_include_by_default():
    """Con include_history=False (por defecto), el historial no se envía."""
    client = make_qwen_client()
    history = [{"role": "user", "content": "mensaje previo"}]
    preguntar_a_vllm(
        make_jpeg_bytes(), "Describe",
        qwen_client=client, qwen_model="qwen-vl",
        history_messages=history,
    )
    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
    # El historial no debe estar (include_history = False en el código fuente)
    payload_str = str(messages)
    assert "mensaje previo" not in payload_str


# ===========================================================================
# retry_message_processing
# ===========================================================================

def test_retry_message_processing_simple_creates_task():
    """Para mensajes SIMPLE crea la tarea simple."""
    receive_msg = make_message_info(msg_type=TipoMensaje.SIMPLE)
    new_msg = make_message_info(msg_type=TipoMensaje.SIMPLE)
    chat = make_chat()
    image = make_image_model()
    get_queue = MagicMock(return_value=asyncio.Queue())

    with patch("app.services.message_processing.asyncio.create_task") as mock_create_task, \
         patch("app.services.message_processing.procesar_mensaje_simple_recibido") as mock_simple:
        mock_simple.return_value = MagicMock()
        retry_message_processing(
            receive_msg, new_msg, chat, image,
            qwen_client=MagicMock(),
            qwen_model="qwen-vl",
            collection=MagicMock(),
            embedder=MagicMock(),
            get_queue=get_queue,
        )
        mock_create_task.assert_called_once()
        mock_simple.assert_called_once()


def test_retry_message_processing_complex_creates_task():
    """Para mensajes COMPLETE crea la tarea compleja."""
    receive_msg = make_message_info(msg_type=TipoMensaje.COMPLETE)
    new_msg = make_message_info(msg_type=TipoMensaje.COMPLETE)
    chat = make_chat()
    image = make_image_model()
    get_queue = MagicMock(return_value=asyncio.Queue())

    with patch("app.services.message_processing.asyncio.create_task") as mock_create_task, \
         patch("app.services.message_processing.procesar_mensaje_complejo_recibido") as mock_complejo:
        mock_complejo.return_value = MagicMock()
        retry_message_processing(
            receive_msg, new_msg, chat, image,
            qwen_client=MagicMock(),
            qwen_model="qwen-vl",
            collection=MagicMock(),
            embedder=MagicMock(),
            get_queue=get_queue,
        )
        mock_create_task.assert_called_once()
        mock_complejo.assert_called_once()
    
@pytest.mark.asyncio
async def test_procesar_mensaje_simple_sets_error_status_on_exception(tmp_path):
    """Cuando el modelo falla, el estado guardado en BD es ERROR."""
    receive_msg = make_message_info(content="Pregunta")
    new_msg = make_message_info(msg_id="resp-err")
    chat = make_chat()
    image_model = make_image_model(image_url="artist/obra.jpg")

    img_path = tmp_path / "artist"
    img_path.mkdir()
    (img_path / "obra.jpg").write_bytes(make_jpeg_bytes())

    get_queue = MagicMock(return_value=asyncio.Queue())

    with patch("app.services.message_processing.AsyncSessionLocal") as mock_session_cls, \
         patch("app.core.config.CARPETA_IMAGENES", str(tmp_path)), \
         patch("app.services.message_processing.build_chat_history_for_vllm", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.message_processing.preguntar_a_vllm", side_effect=RuntimeError("fallo GPU")):

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        execute_calls = []
        mock_session.execute = AsyncMock(side_effect=lambda q, p=None: execute_calls.append((q, p)))
        mock_session.commit = AsyncMock()
        mock_session_cls.return_value = mock_session

        await procesar_mensaje_simple_recibido(
            receive_msg, new_msg, chat, image_model,
            qwen_client=MagicMock(),
            qwen_model="qwen-vl",
            collection=MagicMock(),
            embedder=MagicMock(),
            get_queue=get_queue,
        )

    # El status pasado al UPDATE debe ser ERROR
    update_params = execute_calls[-1][1]
    assert update_params["status"] == Estados.ERROR.value

@pytest.mark.asyncio
async def test_procesar_mensaje_complejo_stores_related_images(tmp_path):
    """En flujo complejo, si hay resultados similares, se llama add_related_images_2_db."""
    receive_msg = make_message_info()
    new_msg = make_message_info(msg_id="resp-c")
    chat = make_chat()
    image_model = make_image_model(image_url="artist/obra.jpg")

    img_path = tmp_path / "artist"
    img_path.mkdir()
    (img_path / "obra.jpg").write_bytes(make_jpeg_bytes())

    similar_results = [
        {"id": "s1", "name": "S", "artist": "A", "style": "St", "genre": "G",
         "similarity_score": 80.0, "distance": 0.2, "image_url": "/art/a/b.jpg", "is_identical": False}
    ]

    get_queue = MagicMock(return_value=asyncio.Queue())

    with patch("app.services.message_processing.AsyncSessionLocal") as mock_session_cls, \
         patch("app.core.config.CARPETA_IMAGENES", str(tmp_path)), \
         patch("app.services.message_processing.build_chat_history_for_vllm", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.message_processing.buscar_imagenes_similares", new_callable=AsyncMock, return_value=similar_results), \
         patch("app.services.message_processing.analizar_imagen_con_qwen_requests", return_value="Descripción"), \
         patch("app.services.message_processing.add_related_images_2_db", new_callable=AsyncMock) as mock_add:

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session_cls.return_value = mock_session

        await procesar_mensaje_complejo_recibido(
            receive_msg, new_msg, chat, image_model,
            qwen_client=MagicMock(),
            qwen_model="qwen-vl",
            collection=MagicMock(),
            embedder=MagicMock(),
            get_queue=get_queue,
        )

    mock_add.assert_called_once()


@pytest.mark.asyncio
async def test_procesar_mensaje_complejo_detects_identical_entry(tmp_path):
    """Si algún resultado tiene score 100, se pasa force_identical_id al analizador."""
    receive_msg = make_message_info()
    new_msg = make_message_info(msg_id="resp-ident")
    chat = make_chat()
    image_model = make_image_model(image_url="artist/obra.jpg")

    img_path = tmp_path / "artist"
    img_path.mkdir()
    (img_path / "obra.jpg").write_bytes(make_jpeg_bytes())

    identical_result = [
        {"id": "ident-99", "name": "X", "artist": "Y", "style": "Z", "genre": "W",
         "similarity_score": 100.0, "distance": 0.0, "image_url": "/art/x/y.jpg", "is_identical": True}
    ]

    get_queue = MagicMock(return_value=asyncio.Queue())
    captured = {}

    def fake_analizar(image_bytes, prompt, results, *, qwen_client, qwen_model, history_messages, force_identical_id):
        captured["force_identical_id"] = force_identical_id
        return "Descripción idéntica"

    with patch("app.services.message_processing.AsyncSessionLocal") as mock_session_cls, \
         patch("app.core.config.CARPETA_IMAGENES", str(tmp_path)), \
         patch("app.services.message_processing.build_chat_history_for_vllm", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.message_processing.buscar_imagenes_similares", new_callable=AsyncMock, return_value=identical_result), \
         patch("app.services.message_processing.analizar_imagen_con_qwen_requests", side_effect=fake_analizar), \
         patch("app.services.message_processing.add_related_images_2_db", new_callable=AsyncMock):

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session_cls.return_value = mock_session

        await procesar_mensaje_complejo_recibido(
            receive_msg, new_msg, chat, image_model,
            qwen_client=MagicMock(),
            qwen_model="qwen-vl",
            collection=MagicMock(),
            embedder=MagicMock(),
            get_queue=get_queue,
        )

    assert captured.get("force_identical_id") == "ident-99"
