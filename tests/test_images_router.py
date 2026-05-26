import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest
import tempfile
import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io

from app.routers import images as images_router
from app.core import config


@pytest.mark.asyncio
async def test_get_image_thumbnail_file_not_found(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(config, "CARPETA_IMAGENES", tmpdir)
        with pytest.raises(HTTPException) as exc:
            await images_router.get_image_thumbnail("nonexistent.jpg", size=200)
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_image_thumbnail_invalid_size(monkeypatch, tmp_path):
    # Creamos un archivo de imagen válido
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    img_path = tmp_path / "small.jpg"
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(img_path, format="JPEG")

    # size <= 0 debe devolver 400
    with pytest.raises(HTTPException) as exc:
        await images_router.get_image_thumbnail(str(img_path.name), size=0)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_image_thumbnail_strip_prefixes(monkeypatch, tmp_path):
    # Comprobamos que acepta prefijos 'art/' y '/art/'
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    img_path = tmp_path / "foo.jpg"
    img = Image.new("RGB", (50, 50), color=(0, 255, 0))
    img.save(img_path, format="JPEG")

    # Llamada con 'art/' prefix
    resp = await images_router.get_image_thumbnail("art/" + img_path.name, size=100)
    assert isinstance(resp, object)  # debe devolver una Response con bytes (no excepción)

    # Llamada con '/art/' prefix
    resp2 = await images_router.get_image_thumbnail("/art/" + img_path.name, size=100)
    assert isinstance(resp2, object)


@pytest.mark.asyncio
async def test_get_image_static_not_found(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(config, "CARPETA_IMAGENES", tmpdir)
        with pytest.raises(HTTPException) as exc:
            await images_router.get_image("noexist.jpg")
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_image_returns_file_response(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    file = tmp_path / "bar.jpg"
    img = Image.new("RGB", (20, 20), color=(0, 0, 255))
    img.save(file, format="JPEG")

    resp = await images_router.get_image(str(file.name))
    assert isinstance(resp, FileResponse)
    assert resp.media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_get_image_thumbnail_returns_valid_jpeg(monkeypatch, tmp_path):
    """Genera una imagen grande, pide thumbnail y comprueba que los bytes devueltos son JPEG válidos."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    # Creamos una imagen relativamente grande para forzar el path que usa thumbnail()
    img_path = tmp_path / "large.jpg"
    large_img = Image.new("RGB", (2000, 2000), color=(123, 222, 64))
    # Guardamos con calidad alta para aumentar tamaño
    large_img.save(img_path, format="JPEG", quality=95)

    resp = await images_router.get_image_thumbnail(str(img_path.name), size=300)

    # Obtener bytes de la Response de forma robusta
    if hasattr(resp, "body"):
        data = resp.body
    else:
        # starlette Response permite render()
        data = resp.render()

    assert resp.media_type == "image/jpeg"
    # Intentar abrir los bytes con PIL para validar que son una imagen válida
    img_bytes = io.BytesIO(data)
    opened = Image.open(img_bytes)
    opened.verify()  # lanzará excepción si no es JPEG/imagen válida


@pytest.mark.asyncio
async def test_get_image_static_file_response_path(monkeypatch, tmp_path):
    """Verifica que `get_image` devuelve FileResponse apuntando al fichero correcto."""
    monkeypatch.setattr(config, "CARPETA_IMAGENES", str(tmp_path))
    file = tmp_path / "baz.jpg"
    img = Image.new("RGB", (30, 30), color=(10, 20, 30))
    img.save(file, format="JPEG")

    resp = await images_router.get_image(str(file.name))
    assert isinstance(resp, FileResponse)
    # FileResponse expone la ruta del fichero en .path
    assert getattr(resp, "path", None) == str(file)
    assert resp.media_type == "image/jpeg"
