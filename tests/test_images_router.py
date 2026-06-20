import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import tempfile
import io
from uuid import uuid4
from fastapi import HTTPException
from fastapi.responses import Response
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from app.routers import images as images_router
from app.core import config


# ---------------------------------------------------------------------------
# Mock helpers para SQLAlchemy async
# ---------------------------------------------------------------------------

class AsyncAll:
    def __init__(self, value):
        self._value = value
    def all(self):
        return self._value

class AsyncMappings:
    def __init__(self, value):
        self._value = value
    def mappings(self):
        return AsyncAll(self._value)

class AsyncScalar:
    def __init__(self, value):
        self._value = value
    def scalar(self):
        return self._value


def make_row(**kwargs):
    """Crea un objeto fila simulado accesible por clave."""
    return MagicMock(**{"__getitem__.side_effect": lambda k: kwargs[k],
                        "get.side_effect": lambda k, d=None: kwargs.get(k, d),
                        **{k: v for k, v in kwargs.items()}})


# ===========================================================================
# search_art
# ===========================================================================

@pytest.mark.asyncio
async def test_search_art_empty_query_skips_db():
    """Con query vacía el endpoint devuelve vacío sin consultar la BD."""
    session = AsyncMock()
    session.execute = AsyncMock()
    result = await images_router.search_art("   ", session=session)
    session.execute.assert_not_called()
    assert result["artists"] == []
    assert result["artworks"] == []  # query vacía devuelve "artworks"

@pytest.mark.asyncio
async def test_search_art_returns_artists_and_artworks():
    """Con resultados reales mockeados devuelve los modelos correctamente."""
    artist_row = make_row(
        id=str(uuid4()), name="Van Gogh", image="/img/van.jpg",
        genre="Post-Impressionism", style="Impressionism"
    )
    artwork_row = make_row(
        id=str(uuid4()), name="Starry Night", local_route="van/starry.jpg",
        year="1889", artist="Van Gogh", style="Post-Impressionism",
        genre="Landscape", artist_id=str(uuid4()),
        style_id=str(uuid4()), genre_id=str(uuid4())
    )

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[
        AsyncMappings([artist_row]),
        AsyncMappings([artwork_row])
    ])

    result = await images_router.search_art("Van Gogh", session=session)

    assert len(result["artists"]) == 1
    assert result["artists"][0].name == "Van Gogh"
    assert len(result["art"]) == 1          # con resultados devuelve "art"
    assert result["art"][0].name == "Starry Night"
    assert result["art"][0].image_url == "/art/van/starry.jpg"
    assert result["total_artists"] == 1
    assert result["total_artworks"] == 1


@pytest.mark.asyncio
async def test_search_art_special_characters():
    """Caracteres especiales no deben lanzar excepción."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[AsyncMappings([]), AsyncMappings([])])
    result = await images_router.search_art("'; DROP TABLE artists;--", session=session)
    assert result["artists"] == []
    assert result["art"] == []

# ===========================================================================
# get_art_paginated
# ===========================================================================

@pytest.mark.asyncio
async def test_get_art_paginated_returns_results(tmp_path, monkeypatch):
    """Con filas mockeadas devuelve los modelos correctamente."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))

    row = make_row(
        id=str(uuid4()), local_route="artist/obra.jpg", name="Obra",
        artist_id=str(uuid4()), style_id=str(uuid4()), genre_id=str(uuid4()),
        owner_id=None, year="1900", artist="Artista", style="Estilo", genre="Género"
    )

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[AsyncMappings([row]), AsyncScalar(1)])

    result = await images_router.get_art_paginated(session=session, user="fakeuser")

    assert len(result["art"]) == 1
    assert result["art"][0].name == "Obra"
    assert result["art"][0].image_url == "/art/artist/obra.jpg"
    assert result["total_items"] == 1


@pytest.mark.asyncio
async def test_get_art_paginated_with_filters():
    """Con filtros artist_id/style_id/genre_id no lanza excepción."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[AsyncMappings([]), AsyncScalar(0)])
    result = await images_router.get_art_paginated(
        session=session,
        user="fakeuser",
        artist_id=str(uuid4()),
        style_id=str(uuid4()),
        genre_id=str(uuid4())
    )
    assert result["art"] == []
    assert result["total_items"] == 0

# ===========================================================================
# get_artists
# ===========================================================================

@pytest.mark.asyncio
async def test_get_artists_returns_results():
    """Con filas mockeadas devuelve los artistas correctamente."""
    row = make_row(
        id=str(uuid4()), name="Picasso", image="/img/picasso.jpg",
        genre="Cubism", style="Cubism"
    )
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[AsyncMappings([row]), AsyncScalar(1)])

    result = await images_router.get_artists(session=session, user="fakeuser")

    assert len(result["artists"]) == 1
    assert result["artists"][0].name == "Picasso"
    assert result["total_items"] == 1

# ===========================================================================
# get_recommended_artists
# ===========================================================================

@pytest.mark.asyncio
async def test_get_recommended_artists_returns_results():
    row = make_row(
        id=str(uuid4()), name="Monet", image="/img/monet.jpg",
        genre="Impressionism", style="Impressionism"
    )
    session = AsyncMock()
    session.execute = AsyncMock(return_value=AsyncMappings([row]))

    result = await images_router.get_recommended_artists(session=session, user="fakeuser")

    assert len(result["artists"]) == 1
    assert result["artists"][0].name == "Monet"
    
# ===========================================================================
# get_image_thumbnail
# ===========================================================================

@pytest.mark.asyncio
async def test_get_image_thumbnail_file_not_found(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(config, "CARPETA_IMAGENES", tmpdir)
        with pytest.raises(HTTPException) as exc:
            await images_router.get_image_thumbnail("nonexistent.jpg", size=200)
        assert exc.value.status_code == 404

@pytest.mark.asyncio
async def test_get_image_thumbnail_strip_prefixes(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    img_path = tmp_path / "foo.jpg"
    Image.new("RGB", (50, 50), color=(0, 255, 0)).save(img_path, format="JPEG")

    resp = await images_router.get_image_thumbnail("art/" + img_path.name, size=100)
    assert isinstance(resp, Response)

    resp2 = await images_router.get_image_thumbnail("/art/" + img_path.name, size=100)
    assert isinstance(resp2, Response)


@pytest.mark.asyncio
async def test_get_image_thumbnail_returns_valid_jpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    img_path = tmp_path / "large.jpg"
    Image.new("RGB", (2000, 2000), color=(123, 222, 64)).save(
        img_path, format="JPEG", quality=95
    )

    resp = await images_router.get_image_thumbnail(str(img_path.name), size=300)

    data = resp.body if hasattr(resp, "body") else resp.render()
    assert resp.media_type == "image/jpeg"
    Image.open(io.BytesIO(data)).verify()


@pytest.mark.asyncio
async def test_get_image_thumbnail_small_image_not_resized(monkeypatch, tmp_path):
    """Imagen pequeña (< 200KB) no se redimensiona pero sí se devuelve."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    img_path = tmp_path / "tiny.jpg"
    Image.new("RGB", (10, 10), color=(0, 0, 255)).save(img_path, format="JPEG")

    resp = await images_router.get_image_thumbnail(str(img_path.name), size=300)
    assert resp.media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_get_image_thumbnail_png_converted_to_jpeg(monkeypatch, tmp_path):
    """PNG con canal alpha se convierte a RGB/JPEG sin error."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    img_path = tmp_path / "image.png"
    # RGBA — canal alpha presente
    Image.new("RGBA", (100, 100), color=(255, 0, 0, 128)).save(img_path, format="PNG")

    resp = await images_router.get_image_thumbnail(str(img_path.name), size=50)
    assert resp.media_type == "image/jpeg"