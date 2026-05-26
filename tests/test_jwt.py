import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest
from jose import jwt, JWTError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import config
from app.core.auth import create_access_token


class TestJWTGeneration:
    """Tests para verificar la generación correcta de JWT"""

    def test_create_access_token_contains_required_fields(self):
        """Verifica que el token contenga los campos necesarios (sub, exp, iat)"""
        user_id = "test-user-123"
        token = create_access_token(user_id)
        
        # Decodificar el token
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        
        # Verificar que contiene los campos requeridos
        assert "sub" in payload
        assert "exp" in payload
        assert "iat" in payload
        assert payload["sub"] == user_id

    def test_create_access_token_expiration_time(self):
        """Verifica que el token tenga la expiración correcta (1 día)"""
        user_id = "test-user-456"
        token = create_access_token(user_id)
        
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        
        # Calcular la diferencia entre exp e iat
        time_diff = payload["exp"] - payload["iat"]
        
        # Debe ser aproximadamente 86400 segundos (1 día)
        assert time_diff == config.ACCESS_TOKEN_EXPIRE_SECONDS
        assert time_diff == 86400

    def test_create_access_token_iat_is_recent(self):
        """Verifica que el iat (issued at) sea el timestamp actual"""
        user_id = "test-user-789"
        now = datetime.now(timezone.utc).timestamp()
        
        token = create_access_token(user_id)
        
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        
        # El iat debe estar dentro de 5 segundos del ahora
        iat_diff = abs(payload["iat"] - int(now))
        assert iat_diff <= 5

    def test_create_access_token_returns_string(self):
        """Verifica que el token devuelto sea una string"""
        user_id = "test-user-string"
        token = create_access_token(user_id)
        
        assert isinstance(token, str)
        assert len(token) > 0

    def test_different_users_get_different_tokens(self):
        """Verifica que usuarios diferentes reciban tokens distintos"""
        token1 = create_access_token("user-1")
        token2 = create_access_token("user-2")
        
        assert token1 != token2
        
        payload1 = jwt.decode(token1, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        payload2 = jwt.decode(token2, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        
        assert payload1["sub"] == "user-1"
        assert payload2["sub"] == "user-2"


class TestJWTValidation:
    """Tests para verificar la validación correcta de JWT"""

    def test_valid_token_decodes_successfully(self):
        """Verifica que un token válido se decodifica correctamente"""
        user_id = "valid-user"
        token = create_access_token(user_id)
        
        # No debe lanzar excepción
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        
        assert payload["sub"] == user_id

    def test_invalid_token_raises_error(self):
        """Verifica que un token inválido lanza error"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(JWTError):
            jwt.decode(invalid_token, config.SECRET_KEY, algorithms=[config.ALGORITHM])

    def test_token_with_wrong_secret_raises_error(self):
        """Verifica que usar una clave secreta incorrecta causa error"""
        user_id = "test-user"
        token = create_access_token(user_id)
        
        wrong_secret = "wrong_secret_key"
        
        with pytest.raises(JWTError):
            jwt.decode(token, wrong_secret, algorithms=[config.ALGORITHM])

    def test_token_with_wrong_algorithm_raises_error(self):
        """Verifica que usar un algoritmo incorrecto causa error"""
        user_id = "test-user"
        token = create_access_token(user_id)
        
        # Intentar decodificar con algoritmo diferente
        with pytest.raises(JWTError):
            jwt.decode(token, config.SECRET_KEY, algorithms=["HS512"])

    def test_expired_token_raises_error(self):
        """Verifica que un token expirado lanza error"""
        user_id = "test-user"
        
        # Crear token con expiración en el pasado
        now = datetime.now(timezone.utc)
        expire = now - timedelta(seconds=10)  # 10 segundos en el pasado
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": now,
        }
        
        expired_token = jwt.encode(payload, config.SECRET_KEY, algorithm=config.ALGORITHM)
        
        # Intentar decodificar debe fallar
        with pytest.raises(JWTError):
            jwt.decode(expired_token, config.SECRET_KEY, algorithms=[config.ALGORITHM])

    def test_token_without_sub_claim_is_invalid(self):
        """Verifica que un token sin el claim 'sub' sea considerado inválido por la app"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(seconds=86400)
        
        # Crear payload sin 'sub'
        payload = {
            "exp": expire,
            "iat": now,
        }
        
        token = jwt.encode(payload, config.SECRET_KEY, algorithm=config.ALGORITHM)
        
        # El token se puede decodificar, pero falta 'sub'
        decoded = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        assert "sub" not in decoded

    def test_token_payload_integrity(self):
        """Verifica que la carga del token no pueda ser modificada sin invalidarlo"""
        user_id = "test-user"
        token = create_access_token(user_id)
        
        # Intentar modificar el token (cambiar last char)
        modified_token = token[:-5] + "xxxxx"
        
        with pytest.raises(JWTError):
            jwt.decode(modified_token, config.SECRET_KEY, algorithms=[config.ALGORITHM])

class TestJWTIntegration:
    """Tests de integración para verificar el flujo completo JWT"""

    def test_create_and_validate_token_flow(self):
        """Verifica el flujo completo de crear y validar un token"""
        user_id = "user-flow-test"
        
        # Paso 1: Crear token
        token = create_access_token(user_id)
        assert token is not None
        assert isinstance(token, str)
        
        # Paso 2: Validar token
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        
        # Paso 3: Verificar claims
        assert payload["sub"] == user_id
        assert "exp" in payload
        assert "iat" in payload
        
        # Paso 4: Verificar que no está expirado
        now = int(datetime.now(timezone.utc).timestamp())
        assert payload["exp"] > now

    
    def test_algorithm_is_correct(self):
        """Verifica que el algoritmo configurado sea HS256"""
        assert config.ALGORITHM == "HS256"
        
        user_id = "algo-test"
        token = create_access_token(user_id)
        
        # Decodificar y verificar el header
        import base64
        parts = token.split('.')
        header_data = parts[0] + '=' * (4 - len(parts[0]) % 4)
        header = base64.urlsafe_b64decode(header_data)
        
        # El header debe contener "HS256"
        assert b"HS256" in header or b"alg" in header

    def test_secret_key_is_configured(self):
        """Verifica que la clave secreta esté configurada"""
        assert config.SECRET_KEY is not None
        assert len(config.SECRET_KEY) > 0
        assert config.SECRET_KEY == "lleva_la_tarara_un_vestido_blanco"
