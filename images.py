from fastapi import APIRouter, Depends, Response, Form, HTTPException
from fastapi.responses import FileResponse
from jose import jwt, JWTError
from PIL import Image
import io
import os
import config
import uuid
from uuid import UUID
from database import get_chroma_collection
from PaginacionRequest import PaginacionRequest
from chromadb.api import Collection
from fastapi import Depends
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from model_loader import embedder
from clases.ImageModel import ImageModel, ArtistModel
from fastapi import HTTPException, Depends
from database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional

bearer_scheme = HTTPBearer(auto_error=False)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        print("DEBUG: Payload decodificado del token:", payload)
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

imagesRouter = APIRouter()
collection = get_chroma_collection()

@imagesRouter.get("/search")
async def search_art(
    query: str,
    artists_limit: int = 5,
    artworks_limit: int = 5,
    session: AsyncSession = Depends(get_session)
):
    """
    Busca artistas y obras de arte por texto.
    """
    print(f"Buscando arte con query: {query}")
    clean_query = query.strip()
    if not clean_query:
        return {"artists": [], "artworks": [], "total_artists": 0, "total_artworks": 0}

    artists_limit = max(1, min(artists_limit, 100))
    artworks_limit = max(1, min(artworks_limit, 100))
    search_value = f"%{clean_query}%"
    search_prefix_value = f"{clean_query}%"

    query_artists = text(
        """
        SELECT id, name, image
        FROM artists
        WHERE name ILIKE :query
        ORDER BY (name ILIKE :prefix) DESC, name
        LIMIT :limit
        """
    )

    query_artworks = text(
        """
        SELECT
            i.id,
            i.name,
            i.local_route,
            i.year,
            i.artist_id,
            i.style_id,
            i.genre_id,
            a.name AS artist,
            s.name AS style,
            g.name AS genre
        FROM images AS i
        LEFT JOIN artists AS a ON i.artist_id = a.id
        LEFT JOIN styles AS s ON i.style_id = s.id
        LEFT JOIN genres AS g ON i.genre_id = g.id
        WHERE i.owner_id IS NULL
          AND (
            i.name ILIKE :query
            OR a.name ILIKE :query
            OR s.name ILIKE :query
            OR g.name ILIKE :query
            OR CAST(i.year AS TEXT) ILIKE :query
          )
                ORDER BY
                        (i.name ILIKE :prefix) DESC,
                        i.name,
                        (a.name ILIKE :prefix) DESC,
                        (s.name ILIKE :prefix) DESC,
                        (g.name ILIKE :prefix) DESC
        LIMIT :limit
        """
    )

    artists_result = await session.execute(
        query_artists,
        {"query": search_value, "prefix": search_prefix_value, "limit": artists_limit},
    )
    artworks_result = await session.execute(
        query_artworks,
        {"query": search_value, "prefix": search_prefix_value, "limit": artworks_limit},
    )

    artists_rows = artists_result.mappings().all()
    artworks_rows = artworks_result.mappings().all()

    artists = []
    for row in artists_rows:
        artista = ArtistModel(id=str(row['id']), name=row['name'], image_url=row['image'])
        artists.append(artista)

    artworks = []
    for row in artworks_rows:
        relative_path = row["local_route"]
        artworks.append(ImageModel(
            id=str(row["id"]),
            name=row["name"],
            artist=row["artist"],
            style=row["style"],
            genre=row["genre"],
            image_url=f"/art/{relative_path}",
            year=row["year"],
            artist_id=str(row["artist_id"]) if row["artist_id"] else None,
            style_id=str(row["style_id"]) if row["style_id"] else None,
            genre_id=str(row["genre_id"]) if row["genre_id"] else None,
        ))

    print(f"Artistas encontrados: {len(artists)}")
    print(f"Obras encontradas: {len(artworks)}")
    return {
        "artists": artists,
        "art": artworks,
        "total_artists": len(artists),
        "total_artworks": len(artworks),
    }

@imagesRouter.get("/view")
async def get_art_paginated(
    page: int = 1,
    items_per_page: int = 20,
    artist_id: Optional[str] = None,
    style_id: Optional[str] = None,
    genre_id: Optional[str] = None,
    user: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    """
    Devuelve los elementos paginados.
    """

    page = max(1, page)
    limit = max(1, min(items_per_page, 100))
    offset = (page - 1) * limit
    
    print(f"Pidiendo página {page} con {limit} elementos.")
    print(f"Consulta DB: LIMIT {limit} OFFSET {offset}")
    print(f"Filtros recibidos: artist_id={artist_id}, style_id={style_id}, genre_id={genre_id}")
    
    art = []
    id = offset
    # Obtenemos solo imágenes públicas y aplicamos filtros opcionales.
    from_sql = """
        FROM images AS i
        LEFT JOIN artists AS a ON i.artist_id = a.id
        LEFT JOIN styles AS s ON i.style_id = s.id
        LEFT JOIN genres AS g ON i.genre_id = g.id
    """
    where_clauses = ["i.owner_id IS NULL"]
    params = {"limit": limit, "offset": offset}

    if artist_id:
        where_clauses.append("i.artist_id = :artist_id")
        params["artist_id"] = str(artist_id)

    if style_id:
        where_clauses.append("i.style_id = :style_id")
        params["style_id"] = str(style_id)

    if genre_id:
        where_clauses.append("i.genre_id = :genre_id")
        params["genre_id"] = str(genre_id)

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            i.id AS id,
            i.local_route AS local_route,
            i.name AS name,
            i.artist_id AS artist_id,
            i.style_id AS style_id,
            i.genre_id AS genre_id,
            i.owner_id AS owner_id,
            i.year AS year,
            a.name AS artist,
            s.name AS style,
            g.name AS genre
        {from_sql}
        WHERE {where_sql}
        ORDER BY i.name
        LIMIT :limit OFFSET :offset
    """)
    result = await session.execute(query, params)
    rows = result.mappings().all()

    count_query = text(f"""
        SELECT COUNT(*)
        {from_sql}
        WHERE {where_sql}
    """)
    count_params = {k: v for k, v in params.items() if k not in {"limit", "offset"}}
    total_result = await session.execute(count_query, count_params)
    collection_size = total_result.scalar()

    for row in rows:
        relative_path = row['local_route']
        full_path = os.path.join(config.CARPETA_IMAGENES, relative_path)
        
        if os.path.exists(full_path) and os.path.getsize(full_path) > 600*1024:
            print(f"Imagen {full_path} es mayor de 600KB, se recomienda usar miniatura.")
        
        image_url = f"/art/{relative_path}"
        art.append(
            ImageModel(
                id=str(row['id']),
                artist_id=str(row['artist_id']) if row['artist_id'] else None,
                style_id=str(row['style_id']) if row['style_id'] else None,
                genre_id=str(row['genre_id']) if row['genre_id'] else None,
                name=row['name'],
                artist=row['artist'],
                style=row['style'],
                genre=row['genre'],
                image_url=image_url,
                year=row['year'],
            )
        )
    return {"art": art, "total_items": collection_size}

@imagesRouter.get("/images/{image_id:uuid}")
async def get_art_by_id(image_id: UUID, user: str = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)):
    """
    Devuelve los detalles de una obra por su ID.
    """
    image_found = await get_image_with_id(image_id, session, user)
    
    image = ImageModel(
        id=str(image_found.id),
        name=image_found.name,
        artist=image_found.artist,
        style=image_found.style,
        genre=image_found.genre,
        image_url=image_found.image_url,
        owner_id=str(image_found.owner_id) if image_found.owner_id else None,
        artist_id=str(image_found.artist_id) if image_found.artist_id else None,
        style_id=str(image_found.style_id) if image_found.style_id else None,
        genre_id=str(image_found.genre_id) if image_found.genre_id else None,
        year=image_found.year
    )
    
    return image

@imagesRouter.post("/find_art")
async def find_art(query: str = Form(...), collection: Collection = Depends(get_chroma_collection)):
    """
    Busca arte por artista, estilo o género.
    """
    print(f"Buscando arte con query: {query}")
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_vector],  # CAMBIAR query_texts por query_embeddings
        include=['metadatas']
    )
    
    art = []
    for id, meta in zip(results['ids'][0], results['metadatas'][0]):
        relative_path = meta['local_route']
        image_url = f"/art/{relative_path}"
        art.append({
            "id": id,
            "artist": meta['artist'],
            "style": meta['style'],
            "genre": meta['genre'],
            "image_url": image_url
        })
    
    return {"art": art, "total_items": len(art)}

DeprecationWarning("El endpoint /find_art está en desuso. Se recomienda usar /search/art con parámetros de búsqueda más específicos.")
@imagesRouter.get("/art/{image_path:path}", deprecated=True)
async def get_image_static(image_path: str):
    """
    Devuelve la imagen original sin procesar.
    Para uso de VLLM y otros servicios que necesitan la imagen completa.
    """
    # Construir ruta completa
    full_path = os.path.join(config.CARPETA_IMAGENES, image_path)
    
    # Verificar que el archivo existe
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {image_path}")
    print(full_path)
    return FileResponse(full_path, media_type="image/jpeg")

@imagesRouter.get("/view/image_thumbnail/{image_path:path}")
async def get_image_thumbnail(image_path: str, size: int = 600):
    """
    Devuelve una miniatura de la imagen solicitada.
    Acepta rutas con o sin el prefijo '/art/'.
    """
    if image_path.startswith('art/'):
        image_path = image_path[4:]
    elif image_path.startswith('/art/'):
        image_path = image_path[5:]
    
    print(f"Generando miniatura para {image_path} con tamaño {size}")
    
    # Construir ruta completa
    full_path = os.path.join(config.CARPETA_IMAGENES, image_path)
    
    # Verificar que el archivo existe
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {image_path}")
    if size <= 0:
        raise HTTPException(status_code=400, detail="El parametro 'size' debe ser mayor que 0")
    
    try:
        with Image.open(full_path) as image:
            image.load()

            # JPEG no admite alpha/paleta: convertir para evitar errores 500 al guardar.
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            elif image.mode == "L":
                image = image.convert("RGB")

            size_ori = os.path.getsize(full_path)
            if size_ori > 200*1024:
                image.thumbnail((size, size), Image.LANCZOS)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr = img_byte_arr.getvalue()
            return Response(content=img_byte_arr, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
    
@imagesRouter.get("/imagen/{filename}")
async def get_image(filename: str):
    full_path = os.path.join(config.CARPETA_IMAGENES, filename)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(full_path, media_type="image/jpeg")

@imagesRouter.get("/artists")
async def get_artists(
    page: int = 1,
    items_per_page: int = 20,
    user: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    artists = []
    page = max(1, page)
    items_per_page = max(1, min(items_per_page, 100))
    offset = (page - 1) * items_per_page

    stmnt = text("""
        SELECT id, name, image
        FROM artists
        ORDER BY name
        LIMIT :limit OFFSET :offset
    """)
    result = await session.execute(stmnt, {"limit": items_per_page, "offset": offset})
    rows = result.mappings().all()
    for row in rows:
        artista = ArtistModel(id=str(row['id']), name=row['name'], image_url=row['image'])
        artists.append(artista)

    count_stmt = text("SELECT COUNT(*) FROM artists")
    count_result = await session.execute(count_stmt)
    total_items = count_result.scalar() or 0

    return {"artists": artists, "total_items": total_items}

@imagesRouter.get("/recomendedArtists")
async def get_recommended_artists(
    limit: int = 6,
    user: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    artists = []
    limit = max(1, min(limit, 50))
    stmnt = text("""
        SELECT a.id, a.name, a.image, COUNT(i.id) AS artworks_count
        FROM artists AS a
        LEFT JOIN images AS i ON i.artist_id = a.id
        GROUP BY a.id, a.name, a.image
        ORDER BY artworks_count DESC, a.name
        LIMIT :limit
    """)
    result = await session.execute(stmnt, {"limit": limit})
    rows = result.mappings().all()
    for row in rows:
        artista = ArtistModel(id=str(row['id']), name=row['name'], image_url=row['image'])
        artists.append(artista)
    return {"artists": artists}

async def get_image_with_id(image_id, session, user=None) -> ImageModel:
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
    print(f"Resultado de la consulta de imagen con ID {image_id}: {image_db}")
    if not image_db:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    image_model = ImageModel(
        id = str(image_db['id']),
        owner_id = str(image_db['owner_id']),
        artist_id=str(image_db['artist_id']),
        style_id=str(image_db['style_id']),
        genre_id=str(image_db['genre_id']),
        name = image_db['name'],
        artist = image_db['artist'],
        style = image_db['style'],
        genre = image_db['genre'],
        year = image_db['year'],
        image_url = image_db['local_route']
    )
    print("\n--------------------\n")
    print(f"Imagen modelada con ID {image_id}: {image_model}")
    print("\n--------------------\n")

    return image_model

async def save_image_and_get_data(contents, user, commit, session) -> ImageModel:
    #Guardamos la imagen en el servidor para poder acceder a ella posteriormente y la añadimos a la BD
    new_id = str(uuid.uuid4())
    with open(os.path.join(config.CARPETA_IMAGENES, f"User/{new_id}.jpg"), 'wb') as img_file:
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
    result = await session.execute(query, {
        "id": new_id,
        "local_route": local_route,
        "owner_id": user,
        "artist_id": artist_id,
        "style_id": style_id,
        "genre_id": genre_id
    })
    if commit:
        await session.commit()
    return ImageModel(
        id = new_id,
        artist_id=str(artist_id),
        style_id=str(style_id),
        genre_id=str(genre_id),
        name="Unknown",
        year="Unknown",
        owner_id = user,
        image_url = local_route
    )


