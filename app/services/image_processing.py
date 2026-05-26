from __future__ import annotations

import base64
import io
from typing import Tuple

from PIL import Image
import os
import uuid
from sqlalchemy import text
from fastapi import HTTPException
from typing import Optional

from app.clases.ImageModel import ImageModel
from app.core import config


def encode_image_bytes(
    image_bytes: bytes,
    max_size: Tuple[int, int] = (1024, 1024),
    quality: int = 85,
    image_format: str = "JPEG",
) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return encode_pil_image(image, max_size=max_size, quality=quality, image_format=image_format)


def encode_image_file(
    image_path: str,
    max_size: Tuple[int, int] = (1024, 1024),
    quality: int = 85,
    image_format: str = "JPEG",
) -> str:
    with open(image_path, "rb") as img_file:
        return encode_image_bytes(
            img_file.read(),
            max_size=max_size,
            quality=quality,
            image_format=image_format,
        )


def encode_pil_image(
    image: Image.Image,
    max_size: Tuple[int, int] = (1024, 1024),
    quality: int = 85,
    image_format: str = "JPEG",
) -> str:
    working_image = image.copy()
    working_image.thumbnail(max_size, Image.LANCZOS)
    if working_image.mode != "RGB":
        working_image = working_image.convert("RGB")

    img_byte_arr = io.BytesIO()
    working_image.save(img_byte_arr, format=image_format, quality=quality)
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


def _artist_characteristics_sql(artist_alias: str) -> str:
    return f"""
        LEFT JOIN LATERAL (
            SELECT g.name AS genre
            FROM images AS i
            INNER JOIN genres AS g ON i.genre_id = g.id
            WHERE i.artist_id = {artist_alias}.id
            GROUP BY g.name
            ORDER BY COUNT(*) DESC, g.name
            LIMIT 1
        ) AS top_genre ON TRUE
        LEFT JOIN LATERAL (
            SELECT s.name AS style
            FROM images AS i
            INNER JOIN styles AS s ON i.style_id = s.id
            WHERE i.artist_id = {artist_alias}.id
            GROUP BY s.name
            ORDER BY COUNT(*) DESC, s.name
            LIMIT 1
        ) AS top_style ON TRUE
    """


async def get_image_with_id(image_id, session, user: Optional[str] = None) -> ImageModel:
    query_img = text("""SELECT i.id, i.local_route, i.owner_id, i.artist_id, 
                     i.style_id, i.genre_id, i.name, i.year, 
                     a.name AS artist, s.name AS style, g.name AS genre 
                     FROM images AS i
                     LEFT JOIN artists a ON i.artist_id = a.id 
                     LEFT JOIN styles s ON i.style_id = s.id 
                     LEFT JOIN genres g ON i.genre_id = g.id 
                     WHERE i.id = :id and (i.owner_id = :user_id OR i.owner_id IS NULL)""")
    result_img = await session.execute(query_img, {"id": image_id, "user_id": user})
    image_db = result_img.mappings().one_or_none()
    if not image_db:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    image_model = ImageModel(
        id=str(image_db['id']),
        owner_id=str(image_db['owner_id']) if image_db['owner_id'] is not None else None,
        artist_id=str(image_db['artist_id']) if image_db['artist_id'] is not None else None,
        style_id=str(image_db['style_id']) if image_db['style_id'] is not None else None,
        genre_id=str(image_db['genre_id']) if image_db['genre_id'] is not None else None,
        name=image_db['name'],
        artist=image_db['artist'],
        style=image_db['style'],
        genre=image_db['genre'],
        year=image_db['year'],
        image_url=image_db['local_route'],
    )
    return image_model


async def save_image_and_get_data(contents: bytes, user: str, commit: bool, session) -> ImageModel:
    new_id = str(uuid.uuid4())
    save_dir = os.path.join(config.CARPETA_IMAGENES, "User")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{new_id}.jpg")
    with open(file_path, 'wb') as img_file:
        img_file.write(contents)
    local_route = f"User/{new_id}.jpg"

    query_artist = text("SELECT id FROM artists WHERE name = :name")
    artist_res = await session.execute(query_artist, {"name": "Unknown Artist"})
    artist_id = artist_res.scalar_one_or_none()

    query_genre = text("SELECT id FROM genres WHERE name = :name")
    genre_res = await session.execute(query_genre, {"name": "Unknown Genre"})
    genre_id = genre_res.scalar_one_or_none()

    query_style = text("SELECT id FROM styles WHERE name = :name")
    style_res = await session.execute(query_style, {"name": "Unknown Style"})
    style_id = style_res.scalar_one_or_none()
    if not style_id:
        style_id = str(uuid.uuid4())
        query_insert_style = text("INSERT INTO styles (id, name) VALUES (:id, :name)")
        await session.execute(query_insert_style, {"id": style_id, "name": "Unknown Style"})

    query = text("""INSERT INTO images (id, local_route, owner_id, artist_id, style_id, genre_id) 
                VALUES (:id, :local_route, :owner_id, :artist_id, :style_id, :genre_id) RETURNING id""")
    await session.execute(query, {
        "id": new_id,
        "local_route": local_route,
        "owner_id": user,
        "artist_id": artist_id,
        "style_id": style_id,
        "genre_id": genre_id,
    })
    if commit:
        await session.commit()
    return ImageModel(
        id=new_id,
        artist_id=str(artist_id) if artist_id is not None else None,
        style_id=str(style_id) if style_id is not None else None,
        genre_id=str(genre_id) if genre_id is not None else None,
        name="Unknown",
        year="Unknown",
        owner_id=user,
        image_url=local_route,
    )
