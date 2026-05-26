import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import firebase_admin.auth as auth

from app.core import config
from app.core.auth import get_firebase_user_from_token, create_access_token
from app.db.database import get_session
from app.clases.Responses import LoginResponse
from app.clases.UserModel import UserData
from app.clases.Requests import CheckLoginRequest, CheckEmailRequest, SignUpWithGoogleRequest


logger = logging.getLogger("uvicorn.error")

authRouter = APIRouter()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()


@authRouter.post("/login")
async def login(
    firebase_user: dict = Depends(get_firebase_user_from_token),
    session: AsyncSession = Depends(get_session),
):
    """gets the firebase connected user"""
    print(firebase_user)
    user = {
        "uid": firebase_user["uid"],
        "name": firebase_user.get("name", "Unknown"),
        "email": firebase_user.get("email", "Unknown"),
    }

    query = text("SELECT * FROM users WHERE firebase_uid = :uid")
    result = await session.execute(query, {"uid": user["uid"]})
    usuario_db = result.mappings().one_or_none()
    internal_id = None
    name = None
    token = None
    if usuario_db:
        internal_id = usuario_db["id"]
        name = usuario_db["name"]
        token = create_access_token(str(internal_id))
        print(f"Usuario con ID interno: {internal_id} ha iniciado sesión.")
    else:
        internal_id = -1
        print(f"Usuario con UID Firebase: {user['uid']} no encontrado en la base de datos.")
        raise HTTPException(status_code=404, detail="Usuario no encontrado en la base de datos. Por favor, regístrate primero.")
    return LoginResponse(id_token=token, status="login", user=UserData(internal_id=str(internal_id), name=name, email=user["email"]))


@authRouter.post("/check-login", response_model=LoginResponse)
async def check_user(
    request: CheckLoginRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        print(f"Verificando token de Google: {request}")
        id_info = id_token.verify_oauth2_token(request.google_token, requests.Request(), GOOGLE_CLIENT_ID)
        email = id_info.get("email")
        print(f"Token válido para el usuario: {email}")

        query = text("SELECT * FROM users WHERE email = :email")
        result = await session.execute(query, {"email": email})
        usuario_db = result.mappings().one_or_none()
        print(usuario_db)
        if usuario_db:
            user_data = UserData(
                internal_id=str(usuario_db["id"]), name=usuario_db["name"], email=usuario_db["email"]
            )
            print(f"Usuario encontrado: {user_data}")
            token = create_access_token(user_data.internal_id)
            return LoginResponse(id_token=token, status="login", user=user_data)
        return LoginResponse(id_token=None, status="signup", user=None)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Error en check-login: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@authRouter.post("/check-email")
async def check_email_exists(
    request: CheckEmailRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        email_clean = request.email.lower().strip()

        query = text("SELECT 1 FROM users WHERE email = :email LIMIT 1")
        result = await session.execute(query, {"email": email_clean})
        user_exists = result.scalar_one_or_none() is not None

        logger.info(f"Verificación de email '{email_clean}': {'Existe' if user_exists else 'Disponible'}")
        return {"exist": user_exists}

    except Exception as e:
        logger.error(f"Error verificando email: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno verificando el email")


@authRouter.post("/perfected-signup")
async def signup_user_google(
    request: SignUpWithGoogleRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        custom_token = None
        print(f"Datos recibidos para signup: {request}")
        if not request.google_token and request.password and request.email:
            user_record = auth.create_user(
                email=request.email,
                password=request.password,
                display_name=request.name,
                email_verified=True,
            )
            firebase_uid = user_record.uid
            insert_sql = text(
                """
                INSERT INTO users (firebase_uid, name, email)
                VALUES (:uid, :name, :email)
                ON CONFLICT (firebase_uid) DO NOTHING
                RETURNING id, name, email;
            """
            )
            result = await session.execute(
                insert_sql,
                {
                    "uid": firebase_uid,
                    "name": request.name,
                    "email": request.email,
                },
            )
            await session.commit()
            internal_id = result.scalar_one()
            print(f"Usuario creado: {request.name}, UID: {firebase_uid}, ID interno: {internal_id}")
            custom_token = auth.create_custom_token(firebase_uid)

        elif request.google_token:
            id_info = id_token.verify_oauth2_token(
                request.google_token,
                requests.Request(),
                GOOGLE_CLIENT_ID,
                clock_skew_in_seconds=60,
            )
            email = id_info.get("email")
            user_record = auth.create_user(email=email, display_name=request.name)
            firebase_uid = user_record.uid
            insert_sql = text(
                "INSERT INTO users (firebase_uid, name, email) VALUES (:uid, :name, :email) ON CONFLICT (firebase_uid) DO NOTHING RETURNING id, name, email;"
            )
            result = await session.execute(
                insert_sql,
                {
                    "uid": firebase_uid,
                    "name": request.name,
                    "email": user_record.email,
                },
            )
            await session.commit()
            internal_id = result.scalar_one()
            print(f"Usuario creado: {request.name}, UID: {firebase_uid}, ID interno: {internal_id}")
            custom_token = auth.create_custom_token(firebase_uid)
        return {"id_token": custom_token.decode("utf-8"), "status": "login"}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"ERROR REAL DEL SERVIDOR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
