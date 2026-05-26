import os
from typing import Annotated
from jose import JWTError, jwt
from openai import OpenAI
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin.auth import verify_id_token

# --- RUTAS ---
# Asegúrate de que esta carpeta apunta a donde descomprimiste el dataset
CARPETA_IMAGENES = os.getenv("CARPETA_IMAGENES", "D:\\wikiart")
CSV_PATH = os.getenv("CSV_PATH", "/data/wikiart/wclasses.csv")
DB_PATH = "./arte_db"

# Database connection strings (override via env vars if needed)
DATABASE_URL_ORI = os.getenv(
  "DATABASE_URL_ORI",
  "postgresql+asyncpg://postgres:3201Alex@127.0.0.1:5432/tfg",
)
DATABASE_URL = os.getenv(
  "DATABASE_URL",
  "postgresql+asyncpg://postgres:3201Alex@127.0.0.1:5435/tfg",
)
DATABASE_URL2 = os.getenv(
  "DATABASE_URL2",
  "postgresql+asyncpg://postgres:3201Alex@db:5432/tfg",
)


SECRET_KEY = "lleva_la_tarara_un_vestido_blanco"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 86400  # 1 day

# --- CLIENTE VLLM ---

# VLLM / OpenAI-like client configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "your_api_key_here")

client = OpenAI(
  base_url=VLLM_BASE_URL,
  api_key=VLLM_API_KEY,
)

# --- CHROMA / Embedding DB ---
CHROMA_USE_HTTP = os.getenv("CHROMA_USE_HTTP", "false").strip().lower() in {"1", "true", "yes", "y"}
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8004"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").strip().lower() in {"1", "true", "yes", "y"}
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "wikiart")

# --- OAuth / External IDs ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

# Misc
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


    