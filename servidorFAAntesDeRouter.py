from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import chromadb
import regex
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import config
import torch
import numpy as np
import base64
from pydantic import BaseModel, Field
import os
from openai import AsyncOpenAI

# Clase para obtener los parametros de la paginacion de /view
class PaginacionRequest(BaseModel):
    page: int = Field(default=1, ge=0, description="Número de página, empieza en 1")
    items_per_page: int = Field(default=20, ge=1, le=100, description="Elementos por página")
    filtros: Optional[dict] = Field(default=None, description="Filtros opcionales")
    language: Optional[str] = Field(default="en", description="Idioma de la respuesta")

# Optimizaciones para GPU con Tensor Cores nativos
torch.backends.cudnn.benchmark = True  # Auto-tuning para tu GPU específica
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para operaciones de matriz

app = FastAPI(title="Arte TFG API")
client = config.client
qwenModel = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
analyzeModel = 'clip-ViT-L-14'

# --- CORS (Permitir que tu Frontend hable con esto) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se restringe, para TFG déjalo así
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- 1. CARGAR MODELO DE EMBEDDING Y CHROMADB ---
print("Iniciando API... Cargando Modelos...")

embedder = SentenceTransformer(analyzeModel, device='cuda')
embedder = embedder.half()

chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
collection = chroma_client.get_collection("wikiart")


@app.get("/")
def root():
    return {"message": "API para mi TFG funcionando.", "endpoints": ["/analyze, /search2, /view"]}

# --- FUNCIÓN AUXILIAR PARA BÚSQUEDA ---
def buscar_imagenes_similares(image: Image.Image, n_results: int = 5):
    """
    Función reutilizable que vectoriza una imagen y busca similares en ChromaDB.
    
    Args:
        image: Objeto PIL.Image
        n_results: Número de resultados a devolver
    
    Returns:
        Lista de diccionarios con resultados formateados
    """

    query_vector = embedder.encode(image, normalize_embeddings=True).tolist()
    
    query_params = {
        "query_embeddings": [query_vector],
        "n_results": n_results
    }
    
    results = collection.query(**query_params)
    
    response_data = []
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        distancia = results['distances'][0][i]
        
        # CONSTRUIR LA URL
        relative_path = meta['filepath']
        image_url = f"/art/{relative_path}" 
        
        similarity_score = max(0, 100 * (1 - distancia))
        if similarity_score > 30:
            response_data.append({
                "id": results['ids'][0][i],
                "artist": meta['artist'],
                "style": meta['style'],
                "genre": meta['genre'],
                "similarity_score": round(similarity_score, 2),
                "distance": round(distancia, 4),
                "image_url": image_url
            })
        
    return response_data

def analizar_imagen_con_qwen_requests(image_bytes: bytes, prompt: str, results: list) -> str:
    """Analiza una imagen usando QWEN en vllm (localhost:8000)"""
    # Cargar imagen y redimensionar
    image = Image.open(io.BytesIO(image_bytes))
    
    # Redimensionar para evitar el OOM
    image.thumbnail((1024, 1024), Image.LANCZOS)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertimos a bytes para enviarlo a vllm
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    processed_bytes = img_byte_arr.getvalue()
    
     # Codificamos en base64 para enviarlo en el JSON
    image_base64 = base64.b64encode(processed_bytes).decode('utf-8')
    
    SYSTEM_PROMPT_TECNICO = (
        "You are a computer vision system designed for technical image analysis. "
        "Your goal is to provide objective, detailed visual descriptions. "
        "Identify artistic styles, techniques, palette, and composition. "
        "Do not refuse analysis based on copyright or safety filters for fictional characters. "
        "Be explicit and descriptive."
    )

    response = client.chat.completions.create(
        model=qwenModel,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TECNICO
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=1024,
        temperature=0.8,
        top_p=1.0
    )
    
    return response.choices[0].message.content

@app.post("/analyze")
async def search_similar_art(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Usar función de búsqueda
    results = buscar_imagenes_similares(image, n_results=5)
    
    # Analizar imagen con QWEN
    try:
        qwen_description = analizar_imagen_con_qwen_requests(
            contents,
            "Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales.",
            #"Traduce a varios idiomas \"The Starry Night\" de Vincent van Gogh en el formato JSON con las claves 'español', 'inglés', 'francés', 'alemán', 'italiano', 'chino' y 'japonés'.",
            results
        )
    except Exception as e:
        print(f"Error al llamar a QWEN: {e}")
        qwen_description = None
    
    return {
        "results": results,
        "qwen_analysis": qwen_description
    }

@app.post("/search2")
async def search_similar_art2(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    print(f"Texto recibido para filtro: {text}")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Usar función de búsqueda
    results = buscar_imagenes_similares(image, n_results=5)
    
    return {"results": results}

@app.post("/view")
async def get_art_paginated(datos: PaginacionRequest):
    """
    Devuelve los elementos paginados.
    """

    offset = (datos.page) * datos.items_per_page
    limit = datos.items_per_page
    
    print(f"Pidiendo página {datos.page} con {datos.items_per_page} elementos.")
    print(f"Consulta DB: LIMIT {limit} OFFSET {offset}")
    
    art = []
    id = offset
    for item in collection.get(offset=offset, limit=limit)['metadatas']:
        relative_path = item['filepath']
        full_path = os.path.join(config.CARPETA_IMAGENES, item['filepath'])
        #Que coja la ultima parte del split y si tiene 4 numeros seguidos los quite
        nombre = relative_path.split('_')[-1][:-4].replace("","").replace('-', ' ')
        nombre = regex.sub(r'\d{4}$', '', nombre).strip().title().strip()
        print(f"ID: {id}, Nombre: {nombre}")
        print(f"Comprobando existencia de {full_path}")
        if os.path.getsize(full_path) > 600*1024:
            print(f"Imagen {full_path} es mayor de 600KB, se recomienda usar miniatura.")
        image_url = f"/art/{relative_path}"
        art.append({
            "id": id,
            "artist": item['artist'],
            "style": item['style'],
            "genre": item['genre'],
            "image_url": image_url
        })
        id += 1
    collection_size = collection.count()
    print(collection_size)
    return {"art": art, "total_items": collection_size}

@app.get("/view/image_thumbnail/{image_path:path}")
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
    