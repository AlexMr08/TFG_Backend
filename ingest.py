import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import config
import torch

# Optimizaciones GPU
torch.backends.cudnn.benchmark = True

# --- FUNCIÓN COLOR (K-MEANS) ---
def get_dominant_colors(image, k=5):
    try:
        img_small = image.resize((400, 400))
        img_array = np.array(img_small)
        if img_array.shape[2] == 4: img_array = img_array[:, :, :3]
        img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
        
        clt = KMeans(n_clusters=k, n_init='auto', random_state=42)
        clt.fit(img_array)
        
        colors_hex = []
        for rgb in clt.cluster_centers_:
            r, g, b = rgb.astype(int)
            colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors_hex
    except:
        return ["#000000"]

# --- INICIALIZACIÓN ---
print(">>> Cargando CLIP en GPU con FP16...")
embedder = SentenceTransformer('clip-ViT-L-14', device='cuda')
embedder = embedder.half()  # FP16 para velocidad

chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
chroma_client.delete_collection("wikiart") # Descomentar si quieres resetear
collection = chroma_client.get_or_create_collection(name="wikiart")

print(">>> Leyendo CSV...")
df = pd.read_csv(config.CSV_PATH)
df['artist_name'] = df['artist'].map(config.ID_TO_LABEL).fillna('Unknown')
df['style_name'] = df['style'].map(config.ID_TO_LABEL).fillna('Unknown')
df['genre_name'] = df['genre'].map(config.ID_TO_LABEL).fillna('Unknown')

# --- PROCESO ---
BATCH_SIZE = 64  # Aumentado más para RTX 5070 Ti
batch_ids, batch_images, batch_metadatas = [], [], []

print(">>> Iniciando Ingesta...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        ruta_relativa = row['file'] # Ej: "Realism/vangogh_01.jpg"
        ruta_completa = os.path.join(config.CARPETA_IMAGENES, ruta_relativa)
        
        if not os.path.exists(ruta_completa): continue

        # Imagen
        image = Image.open(ruta_completa)
        if image.mode != 'RGB': image = image.convert('RGB')
        
        batch_images.append(image)
        batch_ids.append(str(index))
    
        batch_metadatas.append({
            "id": index,
            "artist": row['artist_name'],
            "style": row['style_name'],
            "genre": row['genre_name'],
            "filepath": ruta_relativa,
        })

        if len(batch_images) >= BATCH_SIZE:
            embeddings = embedder.encode(
                batch_images, 
                batch_size=BATCH_SIZE,
                show_progress_bar=False, 
                normalize_embeddings=True,
                convert_to_tensor=False
            ).tolist()
            collection.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_metadatas)
            batch_images, batch_ids, batch_metadatas = [], [], []

    except Exception as e:
        print(f"Error {index}: {e}")

if batch_images:
    embeddings = embedder.encode(
        batch_images, 
        normalize_embeddings=True,
        convert_to_tensor=False
    ).tolist()
    collection.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_metadatas)

print(">>> Base de datos lista.")