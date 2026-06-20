import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from PIL import Image

from app.services.image_processing import (
    encode_image_bytes,
    encode_image_file,
    encode_pil_image,
    get_image_with_id,
    save_image_and_get_data,
    _artist_characteristics_sql,
)
from app.core import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jpeg_bytes(width=100, height=100, color=(255, 0, 0)) -> bytes:
    """Genera bytes JPEG de una imagen sintética."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_rgba_bytes(width=100, height=100) -> bytes:
    """Genera bytes PNG con canal alpha."""
    img = Image.new("RGBA", (width, height), color=(0, 255, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def decode_base64_to_image(b64: str) -> Image.Image:
    """Decodifica base64 a PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class AsyncMappingOne:
    """Mock para result.mappings().one_or_none()"""
    def __init__(self, value):
        self._value = value
    def mappings(self):
        return self
    def one_or_none(self):
        return self._value


class AsyncScalarOne:
    """Mock para result.scalar_one_or_none()"""
    def __init__(self, value):
        self._value = value
    def scalar_one_or_none(self):
        return self._value


# ===========================================================================
# encode_pil_image
# ===========================================================================

def test_encode_pil_image_returns_valid_base64():
    """Devuelve una cadena base64 válida."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = encode_pil_image(img)
    assert isinstance(result, str)
    # Debe poder decodificarse sin error
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_encode_pil_image_result_is_valid_jpeg():
    """El base64 decodificado es una imagen JPEG válida."""
    img = Image.new("RGB", (200, 200), color=(0, 255, 0))
    result = encode_pil_image(img)
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.format == "JPEG"


def test_encode_pil_image_respects_max_size():
    """Imágenes mayores que max_size se reducen."""
    img = Image.new("RGB", (2000, 2000))
    result = encode_pil_image(img, max_size=(100, 100))
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.width <= 100
    assert decoded_img.height <= 100


def test_encode_pil_image_small_image_not_enlarged():
    """Imágenes menores que max_size no se amplían."""
    img = Image.new("RGB", (50, 50))
    result = encode_pil_image(img, max_size=(500, 500))
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.width <= 50
    assert decoded_img.height <= 50


def test_encode_pil_image_converts_rgba_to_rgb():
    """Imágenes RGBA se convierten a RGB antes de guardar como JPEG."""
    img = Image.new("RGBA", (100, 100), color=(0, 0, 255, 128))
    result = encode_pil_image(img)
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.mode == "RGB"


def test_encode_pil_image_converts_grayscale_to_rgb():
    """Imágenes en escala de grises se convierten a RGB."""
    img = Image.new("L", (100, 100), color=128)
    result = encode_pil_image(img)
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.mode == "RGB"


def test_encode_pil_image_does_not_modify_original():
    """La imagen original no se modifica."""
    img = Image.new("RGBA", (100, 100))
    original_mode = img.mode
    original_size = img.size
    encode_pil_image(img)
    assert img.mode == original_mode
    assert img.size == original_size


def test_encode_pil_image_custom_quality():
    """Calidades distintas producen tamaños distintos."""
    img = Image.new("RGB", (500, 500), color=(123, 45, 67))
    high = encode_pil_image(img, quality=95)
    low = encode_pil_image(img, quality=10)
    assert len(base64.b64decode(high)) > len(base64.b64decode(low))


# ===========================================================================
# encode_image_bytes
# ===========================================================================

def test_encode_image_bytes_jpeg():
    """Codifica bytes JPEG correctamente."""
    jpeg_bytes = make_jpeg_bytes()
    result = encode_image_bytes(jpeg_bytes)
    assert isinstance(result, str)
    decoded_img = decode_base64_to_image(result)
    assert decoded_img is not None


def test_encode_image_bytes_png_rgba():
    """Codifica bytes PNG con canal alpha — convierte a RGB."""
    png_bytes = make_png_rgba_bytes()
    result = encode_image_bytes(png_bytes)
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.mode == "RGB"


def test_encode_image_bytes_respects_max_size():
    """Respeta el max_size pasado."""
    jpeg_bytes = make_jpeg_bytes(width=800, height=800)
    result = encode_image_bytes(jpeg_bytes, max_size=(100, 100))
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.width <= 100
    assert decoded_img.height <= 100


def test_encode_image_bytes_invalid_raises():
    """Bytes inválidos lanzan excepción."""
    with pytest.raises(Exception):
        encode_image_bytes(b"not_an_image")


# ===========================================================================
# encode_image_file
# ===========================================================================

def test_encode_image_file_valid(tmp_path):
    """Codifica un archivo de imagen real."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color=(0, 0, 255)).save(img_path, format="JPEG")
    result = encode_image_file(str(img_path))
    assert isinstance(result, str)
    decoded_img = decode_base64_to_image(result)
    assert decoded_img is not None


def test_encode_image_file_not_found():
    """Archivo inexistente lanza FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        encode_image_file("/ruta/inexistente/imagen.jpg")


def test_encode_image_file_respects_max_size(tmp_path):
    """Respeta el max_size."""
    img_path = tmp_path / "large.jpg"
    Image.new("RGB", (1000, 1000)).save(img_path, format="JPEG")
    result = encode_image_file(str(img_path), max_size=(50, 50))
    decoded_img = decode_base64_to_image(result)
    assert decoded_img.width <= 50
    assert decoded_img.height <= 50


# ===========================================================================
# _artist_characteristics_sql
# ===========================================================================

def test_artist_characteristics_sql_contains_alias():
    """El SQL generado contiene el alias del artista."""
    sql = _artist_characteristics_sql("a")
    assert "a.id" in sql


def test_artist_characteristics_sql_contains_genre_and_style():
    """El SQL generado incluye los joins de género y estilo."""
    sql = _artist_characteristics_sql("a")
    assert "top_genre" in sql
    assert "top_style" in sql
    assert "genres" in sql
    assert "styles" in sql


def test_artist_characteristics_sql_different_aliases():
    """Alias diferentes generan SQL diferente."""
    sql_a = _artist_characteristics_sql("a")
    sql_b = _artist_characteristics_sql("b")
    assert "a.id" in sql_a
    assert "b.id" in sql_b
    assert sql_a != sql_b


# ===========================================================================
# get_image_with_id
# ===========================================================================

@pytest.mark.asyncio
async def test_get_image_with_id_found():
    """Devuelve ImageModel cuando la imagen existe."""
    from uuid import uuid4
    img_id = str(uuid4())
    artist_id = str(uuid4())

    row = {
        "id": img_id, "local_route": "artist/obra.jpg",
        "owner_id": None, "artist_id": artist_id,
        "style_id": None, "genre_id": None,
        "name": "Obra", "year": "1900",
        "artist": "Artista", "style": "Estilo", "genre": "Género"
    }
    row_mock = MagicMock()
    row_mock.__getitem__ = lambda self, k: row[k]

    session = AsyncMock()
    session.execute = AsyncMock(return_value=AsyncMappingOne(row_mock))

    result = await get_image_with_id(img_id, session, user="fakeuser")

    assert result.id == str(img_id)
    assert result.name == "Obra"
    assert result.image_url == "artist/obra.jpg"
    assert result.artist_id == str(artist_id)
    assert result.owner_id is None


@pytest.mark.asyncio
async def test_get_image_with_id_not_found():
    """Lanza 404 cuando la imagen no existe."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=AsyncMappingOne(None))

    with pytest.raises(HTTPException) as exc:
        await get_image_with_id("fake_id", session, user="fakeuser")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_image_with_id_owner_id_none():
    """owner_id None se mapea como None en el modelo."""
    from uuid import uuid4
    row = {
        "id": str(uuid4()), "local_route": "ruta.jpg",
        "owner_id": None, "artist_id": None,
        "style_id": None, "genre_id": None,
        "name": "Obra", "year": "1900",
        "artist": None, "style": None, "genre": None
    }
    row_mock = MagicMock()
    row_mock.__getitem__ = lambda self, k: row[k]

    session = AsyncMock()
    session.execute = AsyncMock(return_value=AsyncMappingOne(row_mock))

    result = await get_image_with_id(row["id"], session)
    assert result.owner_id is None
    assert result.artist_id is None


# ===========================================================================
# save_image_and_get_data
# ===========================================================================

@pytest.mark.asyncio
async def test_save_image_and_get_data_creates_file(tmp_path, monkeypatch):
    """Guarda el archivo en disco y devuelve ImageModel."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))

    session = AsyncMock()
    # artist, genre, style queries → scalar_one_or_none
    session.execute = AsyncMock(side_effect=[
        AsyncScalarOne("artist-id"),   # SELECT artist
        AsyncScalarOne("genre-id"),    # SELECT genre
        AsyncScalarOne("style-id"),    # SELECT style — existe, no inserta
        AsyncMock(),                   # INSERT image
    ])

    jpeg_bytes = make_jpeg_bytes()
    result = await save_image_and_get_data(
        contents=jpeg_bytes, user="user-123", commit=False, session=session
    )

    assert result.owner_id == "user-123"
    assert result.image_url.startswith("User/")
    assert result.image_url.endswith(".jpg")

    # Verificar que el archivo existe en disco
    full_path = tmp_path / result.image_url
    assert full_path.exists()


@pytest.mark.asyncio
async def test_save_image_and_get_data_inserts_unknown_style(tmp_path, monkeypatch):
    """Si Unknown Style no existe, lo inserta."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[
        AsyncScalarOne("artist-id"),   # SELECT artist
        AsyncScalarOne("genre-id"),    # SELECT genre
        AsyncScalarOne(None),          # SELECT style → no existe
        AsyncMock(),                   # INSERT style
        AsyncMock(),                   # INSERT image
    ])

    result = await save_image_and_get_data(
        contents=make_jpeg_bytes(), user="user-123", commit=False, session=session
    )

    assert result is not None
    # Debe haberse llamado 5 veces: artist + genre + style + insert_style + insert_image
    assert session.execute.call_count == 5


@pytest.mark.asyncio
async def test_save_image_and_get_data_commits_when_requested(tmp_path, monkeypatch):
    """Con commit=True llama a session.commit()."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[
        AsyncScalarOne("artist-id"),
        AsyncScalarOne("genre-id"),
        AsyncScalarOne("style-id"),
        AsyncMock(),
    ])

    await save_image_and_get_data(
        contents=make_jpeg_bytes(), user="user-123", commit=True, session=session
    )
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_save_image_and_get_data_no_commit_by_default(tmp_path, monkeypatch):
    """Con commit=False no llama a session.commit()."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[
        AsyncScalarOne("artist-id"),
        AsyncScalarOne("genre-id"),
        AsyncScalarOne("style-id"),
        AsyncMock(),
    ])

    await save_image_and_get_data(
        contents=make_jpeg_bytes(), user="user-123", commit=False, session=session
    )
    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_save_image_and_get_data_returns_correct_model(tmp_path, monkeypatch):
    """El ImageModel devuelto tiene los campos correctos."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))

    artist_id = "artist-uuid"
    style_id = "style-uuid"
    genre_id = "genre-uuid"

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[
        AsyncScalarOne(artist_id),
        AsyncScalarOne(genre_id),
        AsyncScalarOne(style_id),
        AsyncMock(),
    ])

    result = await save_image_and_get_data(
        contents=make_jpeg_bytes(), user="user-xyz", commit=False, session=session
    )

    assert result.artist_id == artist_id
    assert result.genre_id == genre_id
    assert result.style_id == style_id
    assert result.owner_id == "user-xyz"
    assert result.name == "Unknown"
    assert result.year == "Unknown"
