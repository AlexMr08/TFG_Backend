"""
Módulo para cargar y compartir modelos entre diferentes partes de la aplicación.
"""
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv(".env")
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# Configuración del modelo
analyzeModel = 'clip-ViT-L-14'

# Cargar el embedder una sola vez
print(f"Cargando modelo {analyzeModel}...")
embedder = SentenceTransformer(analyzeModel, device='cuda', token=hf_token)
embedder = embedder.half()
print("Modelo cargado con éxito.")
