from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import chromadb
import open_clip  # <--- NUEVA LIBRERÍA
from PIL import Image
import io
import config
import torch
import numpy as np
import base64
from pydantic import BaseModel, Field
import os

# --- CLASES ---
class PaginacionRequest(BaseModel):
    page: int = Field(default=1, ge=1)
    items_per_page: int = Field(default=20, ge=1, le=100)

# --- OPTIMIZACIONES GPU ---
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Arte TFG API - ViT-L Edition")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET, POST"],
    allow_headers=["*"],
)

# --- SERVIR IMÁGENES ---
app.mount("/art", StaticFiles(directory=config.CARPETA_IMAGENES), name="art")

# --- CARGA DEL MODELO (OPEN CLIP ViT-L) ---
print(f">>> Cargando OpenCLIP ViT-L-14 (XLM-Roberta) en {DEVICE}...")

# 1. Cargar modelo y transformación de imagen
model, _, preprocess = open_clip.create_model_and_transforms(
    'xlm-roberta-large-ViT-L-14', 
    pretrained='frozen_laion5b_s13b_b90k'
)
model.to(DEVICE)
model.eval() # Modo evaluación

# 2. Cargar el tokenizer para texto (Multilingüe)
tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-L-14')

# 3. Base de Datos
chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
collection = chroma_client.get_collection("wikiart")
print(">>> Sistema listo.")


# --- FUNCIONES DE VECTORIZACIÓN ---

def get_image_embedding(pil_image):
    """Convierte imagen a vector usando OpenCLIP"""
    try:
        # Preprocesar (Resize, Crop, Normalize) y añadir dimensión de batch (unsqueeze)
        image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = model.encode_image(image_tensor)
            # NORMALIZAR (Obligatorio)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.cpu().tolist()[0]
    except Exception as e:
        print(f"Error embedding imagen: {e}")
        return None

def get_text_embedding(text):
    """Convierte texto (Español/Inglés) a vector usando XLM-Roberta"""
    try:
        # Tokenizar
        text_tokens = tokenizer([text]).to(DEVICE)
        
        with torch.no_grad():
            features = model.encode_text(text_tokens)
            # NORMALIZAR (Obligatorio)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.cpu().tolist()[0]
    except Exception as e:
        print(f"Error embedding texto: {e}")
        return None


# --- ENDPOINTS ---

@app.post("/analyze")
async def search_similar_art(file: UploadFile = File(...)):
    """Búsqueda inversa de imágenes (Imagen -> Imagenes similares)"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB': image = image.convert('RGB')
    
    # 1. Generar vector de la imagen subida
    query_vector = get_image_embedding(image)
    
    if not query_vector:
        raise HTTPException(status_code=500, detail="Error procesando la imagen")

    # 2. Buscar en Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5
    )
    
    # 3. Formatear respuesta
    response_data = []
    if results['ids']:
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            distancia = results['distances'][0][i]
            
            similarity_score = max(0, 100 * (1 - distancia)) # Aproximado
            
            response_data.append({
                "id": results['ids'][0][i],
                "artist": meta.get('artist', 'Unknown'),
                "title": meta.get('title', 'Unknown'), # Si lo tienes
                "similarity": round(similarity_score, 2),
                "image_url": f"/art/{meta.get('filepath', '')}"
            })
            
    # Aquí puedes volver a meter tu llamada a QWEN si quieres
    
    return {"results": response_data}


@app.post("/search_text")
async def search_by_text(prompt: str = Form(...)):
    """
    Búsqueda por texto Multilingüe.
    Ej: 'Mujer triste azul' o 'Abstract geometric shapes'
    """
    print(f"Buscando: {prompt}")
    
    # 1. Generar vector del TEXTO
    query_vector = get_text_embedding(prompt)
    
    if not query_vector:
        raise HTTPException(status_code=500, detail="Error procesando texto")
        
    # 2. Buscar en Chroma (comparando vector texto vs vectores imagen guardados)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=10
    )
    
    response_data = []
    if results['ids']:
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            response_data.append({
                "id": results['ids'][0][i],
                "artist": meta.get('artist', 'Unknown'),
                "image_url": f"/art/{meta.get('filepath', '')}"
            })
            
    return {"results": response_data}

# ... (El resto de endpoints /view y /thumbnail se mantienen igual) ...