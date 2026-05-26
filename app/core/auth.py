from typing import Annotated
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin.auth import verify_id_token

from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_SECONDS


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def create_access_token(user_id: str):
    now = datetime.now(timezone.utc)
    expire = now + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
    }
    print("DEBUG create_access_token exp:", int(expire.timestamp()), " iat: ", int(now.timestamp()))
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        now = int(datetime.now().timestamp())
        print("DEBUG now/exp delta:", now, payload.get("exp"), payload.get("exp") - now)
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


bearer_scheme = HTTPBearer(auto_error=False)


def get_firebase_user_from_token(
    token: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict | None:
    """Uses bearer token to identify firebase user id
    Args:
        token : the bearer token. Can be None as we set auto_error to False
    Returns:
        dict: the firebase user on success
    Raises:
        HTTPException 401 if user does not exist or token is invalid
    """
    try:
        if not token:
            raise ValueError("No token")
        user = verify_id_token(token.credentials, clock_skew_seconds=60)
        return user
    except Exception:
        print("Error verificando token de Firebase. Asegúrate de enviar un token válido en el formato 'Bearer <token'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not logged in or Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
