from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging
from app.routers.images import imagesRouter
from app.db.database import view_database
import firebase_admin
from app.routers.auth import authRouter
from app.routers.chats import (
	chatRouter,
)
from app.routers.stream import streamRouter
#from estrategia import Contexto, EstrategiaMensajeSimple, EstrategiaMensajeComplejo

logger = logging.getLogger("uvicorn.error")

# Optimizaciones para GPU con Tensor Cores nativos
torch.backends.cudnn.benchmark = True  # Auto-tuning para tu GPU específica
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para operaciones de matriz

#Creamos la app
app = FastAPI(title="Arte TFG API")
load_dotenv(".env")
firebase_admin.initialize_app()

#Incluimos el router encargado de la gestion de imagenes (Para obtener paginacion y miniaturas)
app.include_router(imagesRouter)
app.include_router(chatRouter)
app.include_router(streamRouter)
app.include_router(authRouter)
# --- CORS (Permitir que tu Frontend hable con esto) ---
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["GET", "POST"],
	allow_headers=["*"],
)

print("Iniciando API...")

@app.get("/")
def root():
	return {"message": "API para mi TFG funcionando.", "endpoints": ["/analyze, /search2, /view"]}


@app.get("/health")
def healthcheck():
	return {"status": "ok"}

@app.get("/collection-info")
def collection_info():
	view_database()

__all__ = ["app"]

