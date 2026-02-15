import pandas as pd
import chromadb
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import config
import torch

# --- CONFIGURACIÓN GPU ---
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Usando dispositivo: {DEVICE}")

# --- FUNCIÓN COLOR (K-MEANS) ---
def get_dominant_colors(image, k=3): # Bajado k=3 para más velocidad
    try:
        # Resize agresivo para velocidad, el color dominante no necesita HD
        img_small = image.resize((150, 150)) 
        img_array = np.array(img_small)
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
             return ["#000000"]

        img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
        
        # n_init=1 es más rápido y suficiente para colores dominantes simples
        clt = KMeans(n_clusters=k, n_init=1, random_state=42)
        clt.fit(img_array)
        
        colors_hex = []
        for rgb in clt.cluster_centers_:
            r, g, b = rgb.astype(int)
            colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors_hex # Devuelve lista ["#FF0000", ...]
    except Exception:
        return []

# --- INICIALIZACIÓN MODELO SIGLIP ---
print(">>> Cargando SigLIP Multilingüe en GPU...")
MODEL_ID = "google/siglip-base-patch16-256-multilingual"

# Cargamos procesador y modelo
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

# Optimizaciones de memoria (FP16 y Eval mode)
model.to(DEVICE)
if DEVICE == "cuda":
    model.half() # FP16
model.eval() 

# --- BASE DE DATOS ---
chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
chroma_client.delete_collection("wikiart") # RESET (Cuidado!)
collection = chroma_client.get_or_create_collection(name="wikiart")

print(">>> Leyendo CSV...")
df = pd.read_csv(config.CSV_PATH)
# Mapeos seguros
df['artist_name'] = df['artist'].map(config.ID_TO_LABEL).fillna('Unknown')
df['style_name'] = df['style'].map(config.ID_TO_LABEL).fillna('Unknown')
df['genre_name'] = df['genre'].map(config.ID_TO_LABEL).fillna('Unknown')

# --- PROCESO ---
BATCH_SIZE = 96 # Con la 5070 Ti y FP16 puedes subir esto. Si da OOM, baja a 64.
batch_ids, batch_images, batch_metadatas = [], [], []

def procesar_batch(images, ids, metas):
    """Función auxiliar para procesar y guardar el lote"""
    if not images: return
    
    try:
        # 1. Preprocesamiento (Tokenización de imagen)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Si usamos FP16, convertimos los inputs
        if DEVICE == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # 2. Inferencia (Sin gradientes para ahorrar VRAM)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
            # 3. NORMALIZACIÓN (CRÍTICO para SigLIP/Chroma)
            # SigLIP no normaliza por defecto, Chroma usa Cosine Similarity
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Convertir a lista de python para Chroma
            embeddings_list = embeddings.cpu().tolist()

        # 4. Guardar en DB
        collection.upsert(ids=ids, embeddings=embeddings_list, metadatas=metas)
        
    except Exception as e:
        print(f"Error procesando batch: {e}")

print(">>> Iniciando Ingesta con SigLIP...")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        ruta_relativa = row['file']
        ruta_completa = os.path.join(config.CARPETA_IMAGENES, ruta_relativa)
        
        if not os.path.exists(ruta_completa): continue

        # Cargar Imagen
        image = Image.open(ruta_completa)
        if image.mode != 'RGB': image = image.convert('RGB')
        
        # Extraer colores (Ahora sí se usa)
        # NOTA: Si va muy lento, comenta esta línea
        #colores = get_dominant_colors(image) 
        
        batch_images.append(image)
        batch_ids.append(str(index))
        
        batch_metadatas.append({
            "artist": str(row['artist_name']),
            "style": str(row['style_name']),
            "genre": str(row['genre_name']),
            "filepath": str(ruta_relativa),
            #"colors": ",".join(colores) # Guardamos como string "hex,hex,hex"
        })

        # Procesar si el batch está lleno
        if len(batch_images) >= BATCH_SIZE:
            procesar_batch(batch_images, batch_ids, batch_metadatas)
            batch_images, batch_ids, batch_metadatas = [], [], [] # Limpiar

    except Exception as e:
        print(f"Error en fila {index}: {e}")

# Procesar remanentes
if batch_images:
    procesar_batch(batch_images, batch_ids, batch_metadatas)

print(">>> Ingesta completada con éxito.")