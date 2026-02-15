import pandas as pd
import chromadb
import asyncio
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
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

async def main():
    # --- CONFIGURACIÓN ASYNC DB ---
    DATABASE_URL = f"postgresql+asyncpg://postgres:3201Alex@127.0.0.1:5432/tfg"
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # --- INICIALIZACIÓN ---
    print(">>> Cargando CLIP en GPU con FP16...")
    embedder = SentenceTransformer('clip-ViT-L-14', device='cuda')
    embedder = embedder.half()  # FP16 para velocidad

    chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
    chroma_client.delete_collection("wikiart") # Descomentar si quieres resetear
    collection = chroma_client.get_or_create_collection(name="wikiart")

    print(">>> Leyendo CSV...")
    df = pd.read_csv(config.CSV_PATH)
    df = df.head(200)
    df['artist_name'] = df['artist'].map(config.ID_TO_LABEL).fillna('Unknown')
    df['style_name'] = df['style'].map(config.ID_TO_LABEL).fillna('Unknown')
    df['genre_name'] = df['genre'].map(config.ID_TO_LABEL).fillna('Unknown')

    # --- PROCESO ---
    BATCH_SIZE = 64
    batch_ids, batch_images, batch_metadatas = [], [], []

    print(">>> Iniciando Ingesta...")
    async with async_session() as session:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                ruta_relativa = row['file']
                ruta_completa = os.path.join(config.CARPETA_IMAGENES, ruta_relativa)
                
                if not os.path.exists(ruta_completa): 
                    continue

                name = ruta_relativa.split('_')[-1][:-4].replace("","").replace('-', ' ').title().strip()
                
                # Insertar en PostgreSQL
                insert_sql = text("""
                    INSERT INTO images (local_route, name, artist, style, genre)
                    VALUES (:local_route, :name, :artist, :style, :genre)
                    ON CONFLICT (local_route) DO NOTHING 
                    RETURNING id;
                """)
                
                result = await session.execute(insert_sql, {
                    "local_route": ruta_relativa,
                    "name": name,
                    "artist": row['artist_name'],
                    "style": row['style_name'],
                    "genre": row['genre_name']
                })
                await session.commit()
                
                # Obtener el ID generado
                image_id = result.scalar_one()

                # Imagen
                image = Image.open(ruta_completa)
                if image.mode != 'RGB': 
                    image = image.convert('RGB')
                
                batch_images.append(image)
                batch_ids.append(str(image_id))
                
                # Guardar los MISMOS metadatos en ChromaDB
                batch_metadatas.append({
                    "id": image_id,
                    "name": name,
                    "artist": row['artist_name'],
                    "style": row['style_name'],
                    "genre": row['genre_name'],
                    "filepath": ruta_relativa,
                })

                # Procesar batch
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
                await session.rollback()
                print(f"Error {index}: {e}")

        # Procesar último batch
        if batch_images:
            embeddings = embedder.encode(
                batch_images, 
                normalize_embeddings=True,
                convert_to_tensor=False
            ).tolist()
            collection.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_metadatas)

    await engine.dispose()
    print(">>> Base de datos lista.")

if __name__ == "__main__":
    asyncio.run(main())