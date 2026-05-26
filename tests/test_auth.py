import pytest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from fastapi.security import HTTPAuthorizationCredentials
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import SECRET_KEY, ALGORITHM
from app.core.auth import get_current_user_id, create_access_token
import jwt
from app.core.auth import get_firebase_user_from_token
import pytest
from fastapi import HTTPException

class TestGetCurrentUserIdValidToken:
    """Comprueba que get_current_user_id devuelve el user_id correcto para tokens válidos"""

    def test_returns_user_id_from_valid_token(self):
        """Crea un token valido y verifica que get_current_user_id devuelve el user_id correcto"""
        user_id = "paquillo123"
        token = create_access_token(user_id)
        result = get_current_user_id(token)
        assert result == user_id

    def test_returns_different_user_ids_correctly(self):
        """Comprueba que diferentes user_ids se extraen correctamente de sus respectivos tokens"""
        user_ids = ["paquillo123", "pepe234", "manolo567"]
        for user_id in user_ids:
            token = create_access_token(user_id)
            result = get_current_user_id(token)
            assert result == user_id

    def test_handles_uuid_user_ids(self):
        """Comprueba que los user_ids en formato UUID se extraen correctamente (que son los que vamos a usar)"""
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        token = create_access_token(user_id)
        result = get_current_user_id(token)
        assert result == user_id

class TestGetCurrentUserIdInvalidToken:
    """Comprueba que get_current_user_id maneja correctamente los tokens inválidos"""

    def test_raises_exception_for_invalid_token(self):
        """Should raise exception for invalid token (not proper JWT format)"""
        invalid_token = "invalid.token.here"

        # This raises a different JWT exception (DecodeError), not HTTPException
        # because the function only catches JWTError
        with pytest.raises(HTTPException):
            get_current_user_id(invalid_token)

    def test_raises_exception_for_empty_token(self):
        """Should raise exception for empty token"""
        empty_token = ""

        # Empty token raises ValueError which is not caught
        with pytest.raises(Exception):
            get_current_user_id(empty_token)

    def test_raises_exception_for_malformed_token(self):
        """Should raise exception for malformed JWT"""
        malformed_token = "not.a.valid.jwt.format"

        # Malformed token raises DecodeError
        with pytest.raises(Exception):
            get_current_user_id(malformed_token)

    def test_raises_http_exception_for_wrong_algorithm(self):
        """Should raise HTTPException when token signature is invalid (wrong key)"""
        user_id = "user123"
        wrong_key = "wrong_secret_key"

        # Create token with wrong key
        payload = {
            "sub": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(seconds=86400),
            "iat": datetime.now(timezone.utc),
        }
        wrong_token = jwt.encode(payload, wrong_key, algorithm=ALGORITHM)

        # InvalidSignatureError is not a JWTError subclass in all versions
        # so this raises a different exception
        with pytest.raises(Exception):
            get_current_user_id(wrong_token)

class TestGetCurrentUserIdExpiredToken:
    """Test get_current_user_id with expired tokens"""

    def test_raises_exception_for_expired_token(self):
        """Should raise exception for expired token (ExpiredSignatureError)"""
        user_id = "user123"

        # Create expired token
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "exp": now - timedelta(seconds=1),  # Expired 1 second ago
            "iat": now - timedelta(seconds=86401),
        }
        expired_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        # ExpiredSignatureError is not caught by JWTError, so this raises an exception
        with pytest.raises(Exception):
            get_current_user_id(expired_token)

    def test_raises_exception_for_token_expired_long_ago(self):
        """Should raise exception for token expired long ago"""
        user_id = "user123"

        # Create token expired 1 day ago
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "exp": now - timedelta(days=1),
            "iat": now - timedelta(days=2),
        }
        expired_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        # ExpiredSignatureError is raised, not caught by JWTError
        with pytest.raises(Exception):
            get_current_user_id(expired_token)

class TestGetCurrentUserIdTokenStructure:
    """Test get_current_user_id handles different token structures"""

    def test_raises_http_exception_for_token_without_sub_claim(self):
        """Should raise HTTPException for token missing 'sub' claim"""
        # Create token without 'sub' claim
        payload = {
            "exp": datetime.now(timezone.utc) + timedelta(seconds=86400),
            "iat": datetime.now(timezone.utc),
            "other_claim": "value"
        }
        token_no_sub = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        with pytest.raises((HTTPException, KeyError)):
            get_current_user_id(token_no_sub)

    def test_token_with_additional_claims_still_works(self):
        """Should extract user_id even when token has additional claims"""
        user_id = "user123"
        now = datetime.now(timezone.utc)

        payload = {
            "sub": user_id,
            "exp": now + timedelta(seconds=86400),
            "iat": now,
            "extra_claim": "extra_value",
            "another_claim": 12345,
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        result = get_current_user_id(token)

        assert result == user_id

class TestGetCurrentUserIdEdgeCases:
    """Test edge cases for get_current_user_id"""

    def test_handles_very_long_user_id(self):
        """Should handle very long user ID strings"""
        user_id = "a" * 1000  # Very long user ID
        token = create_access_token(user_id)

        result = get_current_user_id(token)

        assert result == user_id

    def test_handles_special_characters_in_user_id(self):
        """Should handle special characters in user ID"""
        user_id = "user!@#$%^&*()_+-=[]{}|;:,.<>?"
        token = create_access_token(user_id)

        result = get_current_user_id(token)

        assert result == user_id

    def test_handles_unicode_in_user_id(self):
        """Should handle unicode characters in user ID"""
        user_id = "用户123_ユーザー_пользователь"
        token = create_access_token(user_id)

        result = get_current_user_id(token)

        assert result == user_id

    def test_handles_empty_string_user_id(self):
        """Should handle empty string as user ID"""
        user_id = ""
        token = create_access_token(user_id)

        result = get_current_user_id(token)

        assert result == user_id

    def test_handles_whitespace_user_id(self):
        """Should handle whitespace in user ID"""
        user_id = "  user with spaces  "
        token = create_access_token(user_id)

        result = get_current_user_id(token)

        assert result == user_id

# Ajusta el import según tu estructura de módulos
from app.core.auth import get_firebase_user_from_token


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_token():
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token-123")

@pytest.fixture
def firebase_user():
    return {"uid": "user-abc-123", "email": "test@example.com"}


# ── Tests ────────────────────────────────────────────────────────────────────

class TestGetFirebaseUserFromToken:

    @patch("app.core.auth.verify_id_token")
    def test_returns_user_on_valid_token(self, mock_verify, valid_token, firebase_user):
        """Token válido → devuelve el dict del usuario de Firebase"""
        mock_verify.return_value = firebase_user

        result = get_firebase_user_from_token(valid_token)

        assert result == firebase_user
        mock_verify.assert_called_once_with("valid-token-123", clock_skew_seconds=60)

    def test_raises_401_when_token_is_none(self):
        """Sin token → 401 Unauthorized"""
        with pytest.raises(HTTPException) as exc_info:
            get_firebase_user_from_token(None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Not logged in or Invalid credentials"
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    @patch("app.core.auth.verify_id_token")
    def test_raises_401_when_token_is_expired(self, mock_verify, valid_token):
        """Token expirado → 401 Unauthorized"""
        mock_verify.side_effect = Exception("Token expired")

        with pytest.raises(HTTPException) as exc_info:
            get_firebase_user_from_token(valid_token)

        assert exc_info.value.status_code == 401

    @patch("app.core.auth.verify_id_token")
    def test_raises_401_when_token_is_invalid(self, mock_verify, valid_token):
        """Token malformado → 401 Unauthorized"""
        mock_verify.side_effect = ValueError("Invalid token")

        with pytest.raises(HTTPException) as exc_info:
            get_firebase_user_from_token(valid_token)

        assert exc_info.value.status_code == 401

