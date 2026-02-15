from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import chromadb
from transformers import AutoProcessor, AutoModel # <--- CAMBIO IMPORTANTE
from PIL import Image
import io
import config
import torch
import numpy as np
import base64
from pydantic import BaseModel, Field
import os
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_session, init_db
from image_service import get_image_by_id, get_images_by_ids, get_all_images_paginated

class PaginacionRequest(BaseModel):
    page: int = Field(default=1, ge=1, description="Número de página, empieza en 1")
    items_per_page: int = Field(default=20, ge=1, le=100, description="Elementos por página")

# --- OPTIMIZACIONES GPU ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

app = FastAPI(title="Arte TFG API")
client = config.client

# Modelos
qwenModel = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
# Usamos el modelo SigLIP Multilingüe de Google
siglipModelName = "google/siglip-base-patch16-256-multilingual"

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET, POST"],
    allow_headers=["*"],
)

# --- 1. CARGAR MODELOS (INICIO) ---
print(">>> Iniciando API... Cargando Modelos en GPU...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carga de SigLIP usando Transformers
processor = AutoProcessor.from_pretrained(siglipModelName)
model = AutoModel.from_pretrained(siglipModelName)

# Movemos a GPU y convertimos a FP16 para velocidad
model.to(DEVICE)
if DEVICE == "cuda":
    model.half() # FP16
model.eval() # Modo inferencia

chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
collection = chroma_client.get_collection("wikiart")
print(">>> Modelos cargados y listos.")

@app.on_event("startup")
async def startup_event():
    """Inicializar base de datos al arrancar"""
    await init_db()
    print(">>> Base de datos PostgreSQL inicializada.")



@app.get("/")
def root():
    return {"message": "API TFG (SigLIP Edition) funcionando.", "endpoints": ["/analyze, /search2, /view"]}

# --- FUNCIÓN AUXILIAR: GENERAR EMBEDDING CON SIGLIP ---
def get_siglip_image_embedding(image: Image.Image):
    """Procesa la imagen y devuelve el vector normalizado"""
    try:
        # 1. Preprocesar imagen (Resize, Normalize pixels)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Si el modelo está en FP16, los inputs también deben estarlo
        if DEVICE == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # 2. Inferencia
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
            # 3. NORMALIZACIÓN (CRÍTICO para ChromaDB)
            # SigLIP saca vectores crudos. Para similitud de coseno, hay que normalizar.
            embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # 4. Devolver lista plana
            return embedding.cpu().tolist()[0]
            
    except Exception as e:
        print(f"Error generando embedding: {e}")
        return None

# --- FUNCIÓN DE BÚSQUEDA ADAPTADA ---
async def buscar_imagenes_similares(image: Image.Image, session: AsyncSession, n_results: int = 5):
    """
    Vectoriza con SigLIP y busca en ChromaDB.
    Luego recupera la info completa desde PostgreSQL.
    """
    # Usamos la nueva función de SigLIP
    query_vector = get_siglip_image_embedding(image)
    
    if query_vector is None:
        return []

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": n_results
    }
    
    results = collection.query(**query_params)
    
    # Validar que haya resultados
    if not results['ids']:
        return []

    # Extraer los IDs de las imágenes desde los metadatos
    image_ids = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        image_ids.append(int(meta['image_id']))
    
    # Obtener información completa desde PostgreSQL
    images = await get_images_by_ids(session, image_ids)
    
    # Crear un mapa para acceso rápido
    images_map = {img.id: img for img in images}
    
    response_data = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        image_id = int(meta['image_id'])
        distancia = results['distances'][0][i]
        
        # Obtener imagen desde PostgreSQL
        if image_id not in images_map:
            continue
            
        img_data = images_map[image_id]
        image_url = f"/art/{img_data.local_route}" 
        
        # Calcular score (Chroma devuelve distancia coseno, 0 es idéntico)
        similarity_score = max(0, 100 * (1 - distancia))
        
        if similarity_score > 30: # Filtro de calidad
            response_data.append({
                "id": image_id,
                "name": img_data.name,
                "artist": img_data.artist,
                "style": img_data.style,
                "genre": img_data.genre,
                "similarity_score": round(similarity_score, 2),
                "distance": round(distancia, 4),
                "image_url": image_url
            })
        
    return response_data

# --- RESTO DE ENDPOINTS (VLLM QWEN) ---
# Esta función se mantiene casi igual, solo gestiona la llamada a QWEN
def analizar_imagen_con_qwen_requests(image_bytes: bytes, prompt: str, results: list) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    image.thumbnail((1024, 1024), Image.LANCZOS)
    if image.mode != 'RGB': image = image.convert('RGB')
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    processed_bytes = img_byte_arr.getvalue()
    image_base64 = base64.b64encode(processed_bytes).decode('utf-8')
    
    SYSTEM_PROMPT_TECNICO = (
        "You are a computer vision system designed for technical image analysis. "
        "Your goal is to provide objective, detailed visual descriptions. "
        "Identify artistic styles, techniques, palette, and composition. "
        "Be explicit and descriptive."
    )

    try:
        response = client.chat.completions.create(
            model=qwenModel,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TECNICO},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},, session: AsyncSession = Depends(get_session)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB': image = image.convert('RGB')
    
    # 1. Búsqueda visual con SigLIP
    results = await buscar_imagenes_similares(image, session, n_results=5)
    
    # 2. Análisis textual con QWEN
    qwen_description = analizar_imagen_con_qwen_requests(
        contents,
        "Describe esta obra de arte en detalle, incluyendo estilo, técnica y composición.",
        results
    )
    
    return {
        "results": results,
        "qwen_analysis": qwen_description
    }

@app.post("/search2")
async def search_similar_art2(
    file: UploadFile = File(...),
    text: str = Form(...), # Si quieres buscar por texto, habría que implementar get_siglip_text_embedding
    session: AsyncSession = Depends(get_session)
):
    print(f"Texto recibido (aún no usado en filtro híbrido): {text}")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB': image = image.convert('RGB')
    
    results = await buscar_imagenes_similares(image, session
        "results": results,
        "qwen_analysis": qwen_description
    }

@app.post("/search2")
async def search_similar_art2(
    file: UploadFile = File(...),
    text: str = Form(...) # Si quieres buscar por texto, habría que implementar get_siglip_text_embedding
):
    print(f"Texto recibido (aún no usado en filtro híbrido): {text}")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB': image = image.convert('RGB')
    
    results = buscar_imagenes_similares(image, n_results=5)
    
    return {"results": results}

@app.post("/view"), session: AsyncSession = Depends(get_session)):
    offset = (datos.page - 1) * datos.items_per_page
    limit = datos.items_per_page
    
    # Obtener imágenes directamente desde PostgreSQL
    images = await get_all_images_paginated(session, offset=offset, limit=limit)
    
    art = []
    for img in images:
        image_url = f"/art/{img.local_route}"
        art.append({
            "id": img.id,
            "name": img.name,
            "artist": img.artist,
            "style": img.style,
            "genre": img.genre,
            "image_url": image_url
            "image_url": image_url
            })
            
    collection_size = collection.count()
    return {"art": art, "total_items": collection_size}

@app.get("/view/image_thumbnail/{image_path:path}")
async def get_image_thumbnail(image_path: str, size: int = 600):
    if image_path.startswith('art/'): image_path = image_path[4:]
    elif image_path.startswith('/art/'): image_path = image_path[5:]
    
    full_path = os.path.join(config.CARPETA_IMAGENES, image_pa, session: AsyncSession = Depends(get_session)):
    # Si viene con prefijo /art/, lo quitamos
    if image_path.startswith('art/'): 
        image_path = image_path[4:]
    elif image_path.startswith('/art/'): 
       404, detail="Imagen no encontrada")
    
    try:
        image = Image.open(full_path)
        image.thumbnail((size, size))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))