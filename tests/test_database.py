import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.database as db_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_singletons():
    """Resetea los singletons entre tests para evitar contaminación."""
    db_module._chroma_client_local = None
    db_module._chroma_client_http = None
    db_module._postgres_client = None


# ===========================================================================
# get_chroma_client_local — singleton local
# ===========================================================================

def test_get_chroma_client_local_returns_singleton(tmp_path):
    """Llamadas sucesivas devuelven la misma instancia."""
    reset_singletons()
    with patch("app.db.database.chromadb.PersistentClient") as mock_client:
        mock_client.return_value = MagicMock()
        c1 = db_module.get_chroma_client_local(path=str(tmp_path))
        c2 = db_module.get_chroma_client_local(path=str(tmp_path))
        assert c1 is c2
        mock_client.assert_called_once()  # solo se crea una vez


def test_get_chroma_client_local_uses_config_path_by_default():
    """Sin path explícito usa config.DB_PATH."""
    reset_singletons()
    with patch("app.db.database.chromadb.PersistentClient") as mock_client:
        with patch.object(db_module.config, "DB_PATH", "/fake/path"):
            mock_client.return_value = MagicMock()
            db_module.get_chroma_client_local()
            mock_client.assert_called_once_with(path="/fake/path")


def test_get_chroma_client_local_uses_explicit_path(tmp_path):
    """Con path explícito lo usa en lugar de config."""
    reset_singletons()
    with patch("app.db.database.chromadb.PersistentClient") as mock_client:
        mock_client.return_value = MagicMock()
        db_module.get_chroma_client_local(path=str(tmp_path))
        mock_client.assert_called_once_with(path=str(tmp_path))


# ===========================================================================
# get_chroma_client_http — singleton HTTP
# ===========================================================================

def test_get_chroma_client_http_returns_singleton():
    """Llamadas sucesivas devuelven la misma instancia."""
    reset_singletons()
    with patch("app.db.database.chromadb.HttpClient") as mock_client:
        mock_client.return_value = MagicMock()
        c1 = db_module.get_chroma_client_http()
        c2 = db_module.get_chroma_client_http()
        assert c1 is c2
        mock_client.assert_called_once()


def test_get_chroma_client_http_uses_config_defaults():
    """Sin parámetros usa los valores de config."""
    reset_singletons()
    with patch("app.db.database.chromadb.HttpClient") as mock_client:
        with patch.object(db_module, "CHROMA_HOST", "localhost"):
            with patch.object(db_module, "CHROMA_PORT", 8000):
                with patch.object(db_module, "CHROMA_SSL", False):
                    mock_client.return_value = MagicMock()
                    db_module.get_chroma_client_http()
                    call_kwargs = mock_client.call_args.kwargs
                    assert call_kwargs["host"] == "localhost"
                    assert call_kwargs["port"] == 8000
                    assert call_kwargs["ssl"] is False


def test_get_chroma_client_http_uses_explicit_params():
    """Parámetros explícitos sobreescriben los de config."""
    reset_singletons()
    with patch("app.db.database.chromadb.HttpClient") as mock_client:
        mock_client.return_value = MagicMock()
        db_module.get_chroma_client_http(host="myhost", port=9999, ssl=True)
        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs["host"] == "myhost"
        assert call_kwargs["port"] == 9999
        assert call_kwargs["ssl"] is True


# ===========================================================================
# get_chroma_client — selección HTTP vs local
# ===========================================================================

def test_get_chroma_client_uses_http_when_configured():
    """Con CHROMA_USE_HTTP=True delega en get_chroma_client_http."""
    reset_singletons()
    with patch.object(db_module, "CHROMA_USE_HTTP", True):
        with patch.object(db_module, "get_chroma_client_http") as mock_http:
            with patch.object(db_module, "get_chroma_client_local") as mock_local:
                mock_http.return_value = MagicMock()
                db_module.get_chroma_client()
                mock_http.assert_called_once()
                mock_local.assert_not_called()


def test_get_chroma_client_uses_local_when_configured():
    """Con CHROMA_USE_HTTP=False delega en get_chroma_client_local."""
    reset_singletons()
    with patch.object(db_module, "CHROMA_USE_HTTP", False):
        with patch.object(db_module, "get_chroma_client_http") as mock_http:
            with patch.object(db_module, "get_chroma_client_local") as mock_local:
                mock_local.return_value = MagicMock()
                db_module.get_chroma_client()
                mock_local.assert_called_once()
                mock_http.assert_not_called()


def test_get_chroma_client_explicit_use_http_overrides_config():
    """use_http explícito sobreescribe CHROMA_USE_HTTP de config."""
    reset_singletons()
    with patch.object(db_module, "CHROMA_USE_HTTP", False):
        with patch.object(db_module, "get_chroma_client_http") as mock_http:
            mock_http.return_value = MagicMock()
            db_module.get_chroma_client(use_http=True)  # forzar HTTP
            mock_http.assert_called_once()


def test_get_chroma_client_explicit_local_overrides_config():
    """use_http=False explícito fuerza local aunque config diga HTTP."""
    reset_singletons()
    with patch.object(db_module, "CHROMA_USE_HTTP", True):
        with patch.object(db_module, "get_chroma_client_local") as mock_local:
            mock_local.return_value = MagicMock()
            db_module.get_chroma_client(use_http=False)  # forzar local
            mock_local.assert_called_once()


# ===========================================================================
# get_chroma_collection
# ===========================================================================

def test_get_chroma_collection_uses_config_name():
    """Sin nombre explícito usa config.CHROMA_COLLECTION."""
    reset_singletons()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    with patch.object(db_module, "get_chroma_client", return_value=mock_client):
        with patch.object(db_module.config, "CHROMA_COLLECTION", "wikiart"):
            db_module.get_chroma_collection()
            mock_client.get_or_create_collection.assert_called_once_with("wikiart")


def test_get_chroma_collection_uses_explicit_name():
    """Con nombre explícito lo usa en lugar de config."""
    reset_singletons()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    with patch.object(db_module, "get_chroma_client", return_value=mock_client):
        db_module.get_chroma_collection(collection_name="mi_coleccion")
        mock_client.get_or_create_collection.assert_called_once_with("mi_coleccion")


def test_get_chroma_collection_passes_use_http():
    """Pasa use_http a get_chroma_client."""
    reset_singletons()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    with patch.object(db_module, "get_chroma_client", return_value=mock_client) as mock_get:
        db_module.get_chroma_collection(use_http=True)
        mock_get.assert_called_once_with(use_http=True)


# ===========================================================================
# get_chroma_collection_http
# ===========================================================================

def test_get_chroma_collection_http_uses_default_collection():
    """Sin nombre explícito usa 'wikiart'."""
    reset_singletons()
    with patch.object(db_module, "get_chroma_client_http") as mock_http:
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()
        mock_http.return_value = mock_client
        db_module.get_chroma_collection_http()
        mock_client.get_or_create_collection.assert_called_once_with("wikiart")


def test_get_chroma_collection_http_uses_explicit_collection():
    """Con nombre explícito lo usa."""
    reset_singletons()
    with patch.object(db_module, "get_chroma_client_http") as mock_http:
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()
        mock_http.return_value = mock_client
        db_module.get_chroma_collection_http(collection_name="otra")
        mock_client.get_or_create_collection.assert_called_once_with("otra")


# ===========================================================================
# get_db_connection — singleton engine postgres
# ===========================================================================

def test_get_db_connection_returns_singleton():
    """Llamadas sucesivas devuelven el mismo engine."""
    reset_singletons()
    with patch("app.db.database.create_async_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        e1 = db_module.get_db_connection()
        e2 = db_module.get_db_connection()
        assert e1 is e2
        mock_engine.assert_called_once()


# ===========================================================================
# get_session — generador async de sesión
# ===========================================================================

@pytest.mark.asyncio
async def test_get_session_yields_async_session():
    """get_session debe ceder una AsyncSession."""
    session_mock = AsyncMock(spec=AsyncSession)
    session_mock.__aenter__ = AsyncMock(return_value=session_mock)
    session_mock.__aexit__ = AsyncMock(return_value=False)

    mock_sessionmaker = MagicMock(return_value=session_mock)

    with patch("app.db.database.sessionmaker", return_value=mock_sessionmaker):
        gen = db_module.get_session()
        session = await gen.__anext__()
        assert session is not None
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
