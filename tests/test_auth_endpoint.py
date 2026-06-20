"""
Tests para app/routers/auth.py
Cubre: login, check-login, check-email, perfected-signup
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from app.core.auth import get_firebase_user_from_token
from app.db.database import get_session

# ---------------------------------------------------------------------------
# Helpers de construcción de mocks
# ---------------------------------------------------------------------------

def make_db_user(
    internal_id="user-uuid-1",
    name="Ana García",
    email="ana@example.com",
    firebase_uid="firebase-uid-1",
):
    """Simula una fila de BD como mapping."""
    return {
        "id": internal_id,
        "name": name,
        "email": email,
        "firebase_uid": firebase_uid,
    }


def make_session_mock(row=None, scalar=None):
    """
    Devuelve un AsyncMock que actúa como sesión SQLAlchemy.
    - row   → lo que devuelve result.mappings().one_or_none()
    - scalar → lo que devuelve result.scalar_one() / scalar_one_or_none()
    """
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = row
    result.scalar_one.return_value = scalar
    result.scalar_one_or_none.return_value = scalar

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Fixture de aplicación
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """Importa la app de FastAPI con dependencias reales deshabilitadas."""
    from fastapi import FastAPI
    from app.routers.auth import authRouter

    application = FastAPI()
    application.include_router(authRouter, prefix="/auth")
    return application


# ===========================================================================
# POST /auth/login
# ===========================================================================

class TestLogin:

    @pytest.mark.asyncio
    async def test_login_existing_user_returns_token(self, app):
        """Un usuario registrado obtiene un token JWT."""
        db_user = make_db_user()
        session = make_session_mock(row=db_user)

        firebase_user = {
            "uid": db_user["firebase_uid"],
            "name": db_user["name"],
            "email": db_user["email"],
        }

        async def override_get_session():
            yield session

        app.dependency_overrides[get_firebase_user_from_token] = lambda: firebase_user
        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token", return_value="jwt-token-abc"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/login", headers={"Authorization": "Bearer fake-token"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "login"
        assert data["id_token"] == "jwt-token-abc"
        assert data["user"]["email"] == db_user["email"]

    @pytest.mark.asyncio
    async def test_login_unknown_user_raises_404(self, app):
        """Un usuario no registrado recibe 404."""
        session = make_session_mock(row=None)
        firebase_user = {"uid": "unknown-uid", "name": "Nadie", "email": "nadie@example.com"}

        async def override_get_session():
            yield session

        app.dependency_overrides[get_firebase_user_from_token] = lambda: firebase_user
        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/login", headers={"Authorization": "Bearer fake-token"})

        assert response.status_code == 404
        assert "regístrate" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_login_returns_internal_id_in_user(self, app):
        """El campo user.internal_id refleja el ID interno de la BD."""
        db_user = make_db_user(internal_id="special-uuid-99")
        session = make_session_mock(row=db_user)
        firebase_user = {"uid": db_user["firebase_uid"], "name": db_user["name"], "email": db_user["email"]}

        async def override_get_session():
            yield session

        app.dependency_overrides[get_firebase_user_from_token] = lambda: firebase_user
        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token", return_value="tok"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/login", headers={"Authorization": "Bearer fake-token"})

        assert response.json()["user"]["internal_id"] == "special-uuid-99"


# ===========================================================================
# POST /auth/check-login
# ===========================================================================

class TestCheckLogin:

    @pytest.mark.asyncio
    async def test_check_login_existing_user_returns_login_status(self, app):
        """Token Google válido + usuario en BD → status login con token."""
        db_user = make_db_user()
        session = make_session_mock(row=db_user)

        id_info = {"email": db_user["email"]}

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.id_token.verify_oauth2_token", return_value=id_info), \
             patch("app.routers.auth.create_access_token", return_value="google-jwt"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-login", json={"google_token": "valid-google-token"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "login"
        assert data["id_token"] == "google-jwt"

    @pytest.mark.asyncio
    async def test_check_login_new_user_returns_signup_status(self, app):
        """Token Google válido pero usuario no registrado → status signup, sin token."""
        session = make_session_mock(row=None)
        id_info = {"email": "nuevo@example.com"}

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.id_token.verify_oauth2_token", return_value=id_info):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-login", json={"google_token": "valid-google-token"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "signup"
        assert data["id_token"] is None
        assert data["user"] is None

    @pytest.mark.asyncio
    async def test_check_login_invalid_token_returns_401(self, app):
        """Token Google inválido → 401."""
        with patch("app.routers.auth.id_token.verify_oauth2_token", side_effect=ValueError("bad token")):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-login", json={"google_token": "bad-token"})

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_check_login_db_error_returns_500(self, app):
        """Error inesperado en BD → 500."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB down"))

        id_info = {"email": "test@example.com"}

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.id_token.verify_oauth2_token", return_value=id_info):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-login", json={"google_token": "valid-token"})

        assert response.status_code == 500


# ===========================================================================
# POST /auth/check-email
# ===========================================================================

class TestCheckEmail:

    @pytest.mark.asyncio
    async def test_check_email_exists_returns_true(self, app):
        """Email registrado → {exist: true}."""
        session = make_session_mock(scalar=1)

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-email", json={"email": "ana@example.com"})

        assert response.status_code == 200
        assert response.json()["exist"] is True

    @pytest.mark.asyncio
    async def test_check_email_not_exists_returns_false(self, app):
        """Email no registrado → {exist: false}."""
        session = make_session_mock(scalar=None)

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-email", json={"email": "nuevo@example.com"})

        assert response.status_code == 200
        assert response.json()["exist"] is False

    @pytest.mark.asyncio
    async def test_check_email_strips_and_lowercases(self, app):
        """El endpoint normaliza el email antes de consultar."""
        session = make_session_mock(scalar=None)
        captured = {}

        original_execute = session.execute

        async def capture_execute(query, params=None):
            captured["params"] = params
            return await original_execute(query, params)

        session.execute = capture_execute

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                await client.post("/auth/check-email", json={"email": "  ANA@EXAMPLE.COM  "})

        assert captured["params"]["email"] == "ana@example.com"

    @pytest.mark.asyncio
    async def test_check_email_db_error_returns_500(self, app):
        """Error en BD → 500."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB error"))

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.create_access_token"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/check-email", json={"email": "test@example.com"})

        assert response.status_code == 500


# ===========================================================================
# POST /auth/perfected-signup
# ===========================================================================

class TestPerfectedSignup:

    def _make_user_record(self, uid="fb-uid-new", email="nuevo@example.com"):
        record = MagicMock()
        record.uid = uid
        record.email = email
        return record

    @pytest.mark.asyncio
    async def test_signup_with_email_password_creates_user(self, app):
        """Signup con email+password crea usuario en Firebase y BD."""
        session = make_session_mock(scalar="new-internal-uuid")
        user_record = self._make_user_record()
        fake_token = b"custom-firebase-token"

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.auth.create_user", return_value=user_record), \
             patch("app.routers.auth.auth.create_custom_token", return_value=fake_token), \
             patch("app.routers.auth.create_access_token"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/perfected-signup", json={
                    "name": "Nuevo Usuario",
                    "email": "nuevo@example.com",
                    "password": "securePass123",
                    "google_token": None,
                })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "login"
        assert data["id_token"] == "custom-firebase-token"

    @pytest.mark.asyncio
    async def test_signup_with_google_token_creates_user(self, app):
        """Signup con token de Google crea usuario en Firebase y BD."""
        session = make_session_mock(scalar="google-internal-uuid")
        user_record = self._make_user_record(uid="fb-google-uid", email="googleuser@example.com")
        fake_token = b"google-custom-token"
        id_info = {"email": "googleuser@example.com"}

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.id_token.verify_oauth2_token", return_value=id_info), \
             patch("app.routers.auth.auth.create_user", return_value=user_record), \
             patch("app.routers.auth.auth.create_custom_token", return_value=fake_token), \
             patch("app.routers.auth.create_access_token"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/perfected-signup", json={
                    "name": "Google User",
                    "email": None,
                    "password": None,
                    "google_token": "valid-google-token",
                })

        assert response.status_code == 200
        assert response.json()["status"] == "login"

    @pytest.mark.asyncio
    async def test_signup_invalid_google_token_returns_401(self, app):
        """Token Google inválido en signup → 401."""
        with patch("app.routers.auth.id_token.verify_oauth2_token", side_effect=ValueError("bad")):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/perfected-signup", json={
                    "name": "X",
                    "email": None,
                    "password": None,
                    "google_token": "bad-token",
                })

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_signup_firebase_error_returns_500(self, app):
        """Error al crear usuario en Firebase → 500."""
        with patch("app.routers.auth.auth.create_user", side_effect=Exception("Firebase down")):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/perfected-signup", json={
                    "name": "X",
                    "email": "x@example.com",
                    "password": "pass123",
                    "google_token": None,
                })

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_signup_commits_to_db(self, app):
        """El signup hace commit en la sesión de BD."""
        session = make_session_mock(scalar="uuid-x")
        user_record = self._make_user_record()
        fake_token = b"tok"

        async def override_get_session():
            yield session

        app.dependency_overrides[get_session] = override_get_session

        with patch("app.routers.auth.auth.create_user", return_value=user_record), \
             patch("app.routers.auth.auth.create_custom_token", return_value=fake_token), \
             patch("app.routers.auth.create_access_token"):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                await client.post("/auth/perfected-signup", json={
                    "name": "Test",
                    "email": "test@example.com",
                    "password": "pass",
                    "google_token": None,
                })

        session.commit.assert_called_once()
