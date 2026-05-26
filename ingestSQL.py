import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from app.core import config
import torch
import psycopg2
from app.db.database import get_chroma_collection, get_chroma_client, view_database

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "tfg")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "3201Alex")

ID_TO_LABEL = {
0: 'Unknown Artist', 1: 'Boris Kustodiev', 2: 'Camille Pissarro', 3: 'Childe Hassam', 4: 'Claude Monet', 
  5: 'Edgar Degas', 6: 'Eugene Boudin', 7: 'Gustave Dore', 8: 'Ilya Repin', 9: 'Ivan Aivazovsky', 
  10: 'Ivan Shishkin', 11: 'John Singer Sargent', 12: 'Marc Chagall', 13: 'Martiros Saryan', 14: 'Nicholas Roerich', 
  15: 'Pablo Picasso', 16: 'Paul Cezanne', 17: 'Pierre Auguste Renoir', 18: 'Pyotr Konchalovsky', 19: 'Raphael Kirchner', 
  20: 'Rembrandt', 21: 'Salvador Dali', 22: 'Vincent van Gogh', 23: 'Hieronymus Bosch', 24: 'Leonardo da Vinci', 
  25: 'Albrecht Durer', 26: 'Edouard Cortes', 27: 'Sam Francis', 28: 'Juan Gris', 29: 'Lucas Cranach the Elder', 
  30: 'Paul Gauguin', 31: 'Konstantin Makovsky', 32: 'Egon Schiele', 33: 'Thomas Eakins', 34: 'Gustave Moreau', 
  35: 'Francisco Goya', 36: 'Edvard Munch', 37: 'Henri Matisse', 38: 'Fra Angelico', 39: 'Maxime Maufra', 
  40: 'Jan Matejko', 41: 'Mstislav Dobuzhinsky', 42: 'Alfred Sisley', 43: 'Mary Cassatt', 44: 'Gustave Loiseau', 
  45: 'Fernando Botero', 46: 'Zinaida Serebriakova', 47: 'Georges Seurat', 48: 'Isaac Levitan', 49: 'Joaquin Sorolla', 
  50: 'Jacek Malczewski', 51: 'Berthe Morisot', 52: 'Andy Warhol', 53: 'Arkhip Kuindzhi', 54: 'Niko Pirosmani', 
  55: 'James Tissot', 56: 'Vasily Polenov', 57: 'Valentin Serov', 58: 'Pietro Perugino', 59: 'Pierre Bonnard', 
  60: 'Ferdinand Hodler', 61: 'Bartolome Esteban Murillo', 62: 'Giovanni Boldini', 63: 'Henri Martin', 64: 'Gustav Klimt', 
  65: 'Vasily Perov', 66: 'Odilon Redon', 67: 'Tintoretto', 68: 'Gene Davis', 69: 'Raphael', 
  70: 'John Henry Twachtman', 71: 'Henri de Toulouse Lautrec', 72: 'Antoine Blanchard', 73: 'David Burliuk', 74: 'Camille Corot', 
  75: 'Konstantin Korovin', 76: 'Ivan Bilibin', 77: 'Titian', 78: 'Maurice Prendergast', 79: 'Edouard Manet', 
  80: 'Peter Paul Rubens', 81: 'Aubrey Beardsley', 82: 'Paolo Veronese', 83: 'Joshua Reynolds', 84: 'Kuzma Petrov Vodkin', 
  85: 'Gustave Caillebotte', 86: 'Lucian Freud', 87: 'Michelangelo', 88: 'Dante Gabriel Rossetti', 89: 'Felix Vallotton', 
  90: 'Nikolay Bogdanov Belsky', 91: 'Georges Braque', 92: 'Vasily Surikov', 93: 'Fernand Leger', 94: 'Konstantin Somov', 
  95: 'Katsushika Hokusai', 96: 'Sir Lawrence Alma Tadema', 97: 'Vasily Vereshchagin', 98: 'Ernst Ludwig Kirchner', 99: 'Mikhail Vrubel', 
  100: 'Orest Kiprensky', 101: 'William Merritt Chase', 102: 'Aleksey Savrasov', 103: 'Hans Memling', 104: 'Amedeo Modigliani', 
  105: 'Ivan Kramskoy', 106: 'Utagawa Kuniyoshi', 107: 'Gustave Courbet', 108: 'William Turner', 109: 'Theo van Rysselberghe', 
  110: 'Joseph Wright', 111: 'Edward Burne Jones', 112: 'Koloman Moser', 113: 'Viktor Vasnetsov', 114: 'Anthony van Dyck', 
  115: 'Raoul Dufy', 116: 'Frans Hals', 117: 'Hans Holbein the Younger', 118: 'Ilya Mashkov', 119: 'Henri Fantin Latour', 
  120: 'M.C. Escher', 121: 'El Greco', 122: 'Mikalojus Ciurlionis', 123: 'James McNeill Whistler', 124: 'Karl Bryullov', 
  125: 'Jacob Jordaens', 126: 'Thomas Gainsborough', 127: 'Eugene Delacroix', 128: 'Canaletto',
  # Géneros
  129: 'Abstract Painting', 130: 'Cityscape', 131: 'Genre Painting', 132: 'Illustration', 133: 'Landscape', 
  134: 'Nude Painting', 135: 'Portrait', 136: 'Religious Painting', 137: 'Sketch and Study', 138: 'Still Life', 
  139: 'Unknown Genre',
  # Estilos
  140: 'Abstract Expressionism', 141: 'Action Painting', 142: 'Analytical Cubism', 143: 'Art Nouveau', 144: 'Baroque', 
  145: 'Color Field Painting', 146: 'Contemporary Realism', 147: 'Cubism', 148: 'Early Renaissance', 149: 'Expressionism', 
  150: 'Fauvism', 151: 'High Renaissance', 152: 'Impressionism', 153: 'Mannerism Late Renaissance', 154: 'Minimalism', 
  155: 'Naive Art Primitivism', 156: 'New Realism', 157: 'Northern Renaissance', 158: 'Pointillism', 159: 'Pop Art', 
  160: 'Post Impressionism', 161: 'Realism', 162: 'Rococo', 163: 'Romanticism', 164: 'Symbolism', 
  165: 'Synthetic Cubism', 166: 'Ukiyo e', 167: 'Unknown Style'
}

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
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
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


# Selección de dispositivo para embeddings.
# Permite forzar con INGEST_DEVICE=cpu|cuda; si no, auto-detecta.
requested_device = os.getenv("INGEST_DEVICE", "auto").strip().lower()
if requested_device == "auto":
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif requested_device == "cuda" and not torch.cuda.is_available():
    print(">>> INGEST_DEVICE=cuda pero no hay GPU/NVIDIA disponible. Se usará CPU.")
    EMBEDDING_DEVICE = "cpu"
else:
    EMBEDDING_DEVICE = requested_device

if EMBEDDING_DEVICE == "cuda":
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
print(f">>> Cargando CLIP en {EMBEDDING_DEVICE.upper()}...")
embedder = SentenceTransformer('clip-ViT-L-14', device=EMBEDDING_DEVICE)
if EMBEDDING_DEVICE == "cuda":
    embedder = embedder.half()  # FP16 solo en GPU

CHROMA_USE_HTTP = True
chroma_client = get_chroma_client(use_http=CHROMA_USE_HTTP)
chroma_client.delete_collection("wikiart")
collection = get_chroma_collection("wikiart", use_http=CHROMA_USE_HTTP)

print(">>> Leyendo CSV...")
df = pd.read_csv(config.CSV_PATH)
df = df.head(500)
df['artist_name'] = df['artist'].map(ID_TO_LABEL).fillna('Unknown')
df['style_name'] = df['style'].map(ID_TO_LABEL).fillna('Unknown')
df['genre_name'] = df['genre'].map(ID_TO_LABEL).fillna('Unknown')

# --- PROCESO ---
BATCH_SIZE = 64  # Aumentado más para RTX 5070 Ti
batch_ids, batch_images, batch_metadatas = [], [], []
print(">>> Iniciando Ingesta...")

conexion = psycopg2.connect(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
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