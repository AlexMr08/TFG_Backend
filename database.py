from typing import AsyncGenerator
import chromadb
from chromadb.config import Settings
import config


from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

_chroma_client = None
_postgres_client = None

# Opcional: Obtener la colección aquí si siempre usas la misma
def get_chroma_collection():
    global _chroma_client
    if _chroma_client is None:
        # Solo se inicializa si no existe
        _chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
    return _chroma_client.get_or_create_collection("wikiart")


DATABASE_URL_ORI = "postgresql+asyncpg://postgres:3201Alex@127.0.0.1:5432/tfg"
DATABASE_URL2 = "postgresql+asyncpg://postgres:3201Alex@127.0.0.1:5435/tfg"
DATABASE_URL = "postgresql+asyncpg://postgres:3201Alex@db:5432/tfg"
engine = create_async_engine(DATABASE_URL, echo=True, future=True, pool_size=20, max_overflow=10)

async def init_db():
    async with engine.begin() as conn:
        # Aquí crea todas las tablas heredadas de SQLModel
        await conn.run_sync(SQLModel.metadata.create_all)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

def get_db_connection():
    global _postgres_client
    if _postgres_client is None:
        # Solo se inicializa si no existe
        _postgres_client = create_async_engine(DATABASE_URL, echo=True, future=True)
    return _postgres_client


def view_database():
    collection = get_chroma_collection()
    print(">>> Base de datos lista.")

    # --- INSPECCIÓN EN CONSOLA ---
    print("\n" + "="*60)
    print("INSPECCIÓN DE CHROMADB")
    print("="*60)

    total = collection.count()
    print(f"\n📊 Total de imágenes ingresadas: {total}")

    # Ver primeros 5 elementos
    print("\n🔍 Primeros 5 elementos:")
    results = collection.get(limit=5, include=['metadatas'])

    for i, (id, meta) in enumerate(zip(results['ids'], results['metadatas']), 1):
        print(f"\n  [{i}] ID: {id}")
        print(f"      Artista: {meta.get('artist', 'N/A')}")
        print(f"      Estilo: {meta.get('style', 'N/A')}")
        print(f"      Género: {meta.get('genre', 'N/A')}")
        print(f"      Ruta: {meta.get('filepath', 'N/A')}")

    # Estadísticas por artista (primeros 10 artistas)
    print("\n📈 Distribución por artistas (top 10):")
    all_results = collection.get(include=['metadatas'])
    artists = {}
    for meta in all_results['metadatas']:
        artist = meta.get('artist', 'Unknown')
        artists[artist] = artists.get(artist, 0) + 1

    for artist, count in sorted(artists.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  • {artist}: {count} obras")

    print("\n✅ Proceso completado\n")