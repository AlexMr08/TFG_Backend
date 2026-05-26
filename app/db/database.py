from typing import AsyncGenerator
import os
import chromadb
from chromadb.config import Settings
from app.core import config


from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

_chroma_client_local = None
_chroma_client_http = None
_postgres_client = None

CHROMA_USE_HTTP = config.CHROMA_USE_HTTP
CHROMA_HOST = config.CHROMA_HOST
CHROMA_PORT = config.CHROMA_PORT
CHROMA_SSL = config.CHROMA_SSL


def get_chroma_client_local(path: str | None = None):
    """Cliente singleton de Chroma local (PersistentClient)."""
    global _chroma_client_local
    if _chroma_client_local is None:
        _chroma_client_local = chromadb.PersistentClient(path=path or config.DB_PATH)
    return _chroma_client_local


def get_chroma_client_http(
    host: str | None = None,
    port: int | None = None,
    ssl: bool | None = None,
):
    """Cliente singleton de Chroma remoto por HTTP (HttpClient)."""
    global _chroma_client_http

    resolved_host = host or CHROMA_HOST
    resolved_port = port if port is not None else CHROMA_PORT
    resolved_ssl = CHROMA_SSL if ssl is None else ssl

    if _chroma_client_http is None:
        _chroma_client_http = chromadb.HttpClient(
            host=resolved_host,
            port=resolved_port,
            ssl=resolved_ssl,
            settings=Settings(allow_reset=True),
        )
    return _chroma_client_http


def get_chroma_client(use_http: bool | None = None):
    """Devuelve el cliente Chroma activo según el modo configurado."""
    resolved_use_http = CHROMA_USE_HTTP if use_http is None else use_http
    return get_chroma_client_http() if resolved_use_http else get_chroma_client_local()


def get_chroma_collection(collection_name: str | None = None, use_http: bool | None = None):
    name = collection_name or config.CHROMA_COLLECTION
    return get_chroma_client(use_http=use_http).get_or_create_collection(name)


def get_chroma_collection_http(
    collection_name: str = "wikiart",
    host: str | None = None,
    port: int | None = None,
    ssl: bool | None = None,
):
    """Devuelve una colección de ChromaDB usando cliente HTTP."""
    return get_chroma_client_http(host=host, port=port, ssl=ssl).get_or_create_collection(collection_name)


DATABASE_URL_ORI = config.DATABASE_URL_ORI
DATABASE_URL = config.DATABASE_URL
DATABASE_URL2 = config.DATABASE_URL2

# Engine created from centralized config DATABASE_URL
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


def view_database(use_http: bool = False):
    collection = get_chroma_collection("wikiart", use_http=use_http)
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
