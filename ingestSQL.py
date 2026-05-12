import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import config
import torch
import psycopg2
from database import get_chroma_collection, get_chroma_client, view_database

def get_or_create_artist(cur, artist_name: str):
    sql = """
    INSERT INTO artists (name)
    VALUES (%s)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id;
    """
    cur.execute(sql, (artist_name,))
    return cur.fetchone()[0]

def get_or_create_genre(cur, genre_name: str):
    sql = """
    INSERT INTO genres (name)
    VALUES (%s)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id;
    """
    cur.execute(sql, (genre_name,))
    return cur.fetchone()[0]

def get_or_create_style(cur, style_name: str):
    sql = """
    INSERT INTO styles (name)
    VALUES (%s)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id;

    """
    cur.execute(sql, (style_name,))
    return cur.fetchone()[0]

def insert_image_postgresV2(conn, relative_path, name, artist_name, style, genre, year):
    with conn.cursor() as cur:
        artist_id = get_or_create_artist(cur, artist_name)
        genre_id = get_or_create_genre(cur, genre)
        style_id = get_or_create_style(cur, style)
        sql = """
        INSERT INTO images (local_route, name, artist_id, style_id, genre_id, year)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (local_route) DO NOTHING
        RETURNING id;
        """
        cur.execute(sql, (relative_path, name, artist_id, style_id, genre_id, year))
        inserted_row = cur.fetchone()
        return inserted_row[0] if inserted_row else None

def get_image_id_by_local_route(conn, relative_path):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM images WHERE local_route = %s", (relative_path,))
        row = cur.fetchone()
        return row[0] if row else None

def insert_image_postgres(ruta_relativa, name, artist, style, genre, year):
    sql = """ INSERT INTO images (local_route, name, artist, style, genre, year)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;"""
    
    image_id = None
    try:
        with psycopg2.connect(
            dbname="tfg",
            user="postgres",
            password="3201Alex",
            host="localhost",
            port="5432"
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    ruta_relativa,
                    name,
                    artist,
                    style,
                    genre,
                    year
                ))
                image_id = cur.fetchone()[0]
                conn.commit()
    except Exception as e:
        print(f"Error inserting image into Postgres: {e}")
    finally:
        return image_id


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

CHROMA_USE_HTTP = True
chroma_client = get_chroma_client(use_http=CHROMA_USE_HTTP)
chroma_client.delete_collection("wikiart")
collection = get_chroma_collection("wikiart", use_http=CHROMA_USE_HTTP)

print(">>> Leyendo CSV...")
df = pd.read_csv(config.CSV_PATH)
df = df.head(500)
df['artist_name'] = df['artist'].map(config.ID_TO_LABEL).fillna('Unknown')
df['style_name'] = df['style'].map(config.ID_TO_LABEL).fillna('Unknown')
df['genre_name'] = df['genre'].map(config.ID_TO_LABEL).fillna('Unknown')

# --- PROCESO ---
BATCH_SIZE = 64  # Aumentado más para RTX 5070 Ti
batch_ids, batch_images, batch_metadatas = [], [], []
print(">>> Iniciando Ingesta...")

conexion = psycopg2.connect(
    dbname="tfg",
    user="postgres",
    password="3201Alex",
    host="localhost",
    port="5432"
)
pending_db_writes = 0
duplicates_skipped = 0
try:
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            ruta_relativa = row['file'] # Ej: "Realism/vangogh_01.jpg"
            ruta_completa = os.path.join(config.CARPETA_IMAGENES, ruta_relativa)

            if not os.path.exists(ruta_completa):
                continue

            nombre = ruta_relativa.split('_')[-1][:-4].replace("","").title().replace('-', ' ').replace(" S ", "'s ").strip()
            year = nombre.split()[-1] if nombre.split()[-1].isdigit() and len(nombre.split()[-1]) == 4 else 'Unknown'
            if year != 'Unknown':
                nombre = nombre.replace(year, '').strip()

            image_id = insert_image_postgresV2(
                conn=conexion,
                relative_path=ruta_relativa,
                name=nombre,
                artist_name=row['artist_name'],
                style=row['style_name'],
                genre=row['genre_name'],
                year=year,
            )

            if image_id is None:
                duplicates_skipped += 1
                image_id = get_image_id_by_local_route(conexion, ruta_relativa)
                if image_id is None:
                    print(f"No se pudo recuperar el ID existente para '{ruta_relativa}'. Se omite esta imagen.")
                    continue
                print(f"Image '{nombre}' from '{row['artist_name']}' already exists in PostgreSQL. Reusing ID {image_id} for ChromaDB. Total duplicates: {duplicates_skipped}")

            pending_db_writes += 1

            # Imagen
            image = Image.open(ruta_completa)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            batch_images.append(image)
            batch_ids.append(str(image_id))

            batch_metadatas.append({
                "id": image_id,
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

                # Commit por lote para reducir overhead y mantener consistencia.
                conexion.commit()
                pending_db_writes = 0
                batch_images, batch_ids, batch_metadatas = [], [], []

        except Exception as e:
            # Deja la conexión limpia para continuar con la siguiente fila.
            conexion.rollback()
            pending_db_writes = 0
            print(f"Error {index}: {e}")
finally:
    if pending_db_writes > 0:
        conexion.commit()
    conexion.close()

if batch_images:
    embeddings = embedder.encode(
        batch_images, 
        normalize_embeddings=True,
        convert_to_tensor=False
    ).tolist()
    collection.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_metadatas)

print(">>> Base de datos lista.")
view_database(use_http=CHROMA_USE_HTTP)