from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile
from fastapi import Response, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from jose import jwt, JWTError
from pydantic import BaseModel
from PIL import Image
import io
import os
import config
import regex

from database import get_chroma_collection
from PaginacionRequest import PaginacionRequest
from chromadb.api import Collection

from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from firebase_admin.auth import verify_id_token
from model_loader import embedder

bearer_scheme = HTTPBearer(auto_error=False)

from firebase_admin import auth
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

def get_firebase_user_from_token(
    token: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict | None:
    try:
        if not token:
            raise ValueError("No token provided")
        
        # verify_id_token comprueba firma, expiración y formato
        # check_revoked=True es opcional, verifica si el usuario cambió la contraseña recientemente
        user = auth.verify_id_token(token.credentials, check_revoked=False, clock_skew_seconds=10)
        return user

    except auth.ExpiredIdTokenError:
        print("DEBUG: El token ha caducado (duración máx 1 hora).")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except auth.RevokedIdTokenError:
        print("DEBUG: El token ha sido revocado (usuario cambió contraseña o deshabilitado).")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except auth.InvalidIdTokenError as e:
        # AQUÍ es donde verás si es un problema de "Issued in the future"
        print(f"DEBUG: Token inválido - Detalle real: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}", # Devuelve el error al front para testear
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except Exception as e:
        print(f"DEBUG: Error inesperado validando token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
    
@imagesRouter.post("/view")
async def get_art_paginated(datos: PaginacionRequest, user: str = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)):
    """
    Devuelve los elementos paginados.
    """

    offset = (datos.page) * datos.items_per_page
    limit = datos.items_per_page
    
    print(f"Pidiendo página {datos.page} con {datos.items_per_page} elementos.")
    print(f"Consulta DB: LIMIT {limit} OFFSET {offset}")
    
    art = []
    id = offset
    #Obtenemos solo las imagenes sin propietario (owner_id IS NULL) para mostrar en la galería pública, ordenadas por nombre
    query = text("""SELECT * FROM images WHERE owner_id IS NULL ORDER BY name LIMIT :limit OFFSET :offset""")
    result = await session.execute(query, {"limit": limit, "offset": offset})
    rows = result.mappings().all()

    count_query = text("SELECT COUNT(*) FROM images WHERE owner_id IS NULL")
    total_result = await session.execute(count_query)
    collection_size = total_result.scalar()

    for row in rows:
        relative_path = row['local_route']
        full_path = os.path.join(config.CARPETA_IMAGENES, relative_path)
        
        if os.path.exists(full_path) and os.path.getsize(full_path) > 600*1024:
            print(f"Imagen {full_path} es mayor de 600KB, se recomienda usar miniatura.")
        
        image_url = f"/art/{relative_path}"
        art.append({
            "id": row['id'],
            "name": row['name'],
            "artist": row['artist'],
            "style": row['style'],
            "genre": row['genre'],
            "image_url": image_url
        })
    return {"art": art, "total_items": collection_size}

@imagesRouter.get("/images/{image_id}")
async def get_art_by_id(image_id: str, user: str = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)):
    """
    Devuelve los detalles de una obra por su ID.
    """
    query = text("SELECT * FROM images WHERE id = :id AND (owner_id IS NULL OR owner_id = :user_id)")
    result = await session.execute(query, {"id": image_id, "user_id": user})
    row = result.mappings().first()
    
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    
    relative_path = row['local_route']
    image_url = f"/art/{relative_path}"
    
    return {
        "id": row['id'],
        "name": row['name'],
        "artist": row['artist'],
        "style": row['style'],
        "genre": row['genre'],
        "image_url": image_url
    }

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

@imagesRouter.get("/art/{image_path:path}")
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
    
    try:
        image = Image.open(full_path)
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