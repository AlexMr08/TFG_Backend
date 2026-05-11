from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, OAuth2PasswordBearer
from jose import jwt, JWTError
from PIL import Image
import io
import config
import torch
import numpy as np
import base64
import logging
from pydantic import BaseModel, EmailStr, Field
from typing import Annotated, Optional
import os
from model_loader import embedder
from images import imagesRouter, get_image_with_id, save_image_and_get_data
from database import AsyncSessionLocal, get_chroma_collection, view_database
import firebase_admin
import firebase_admin.auth as auth
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin.auth import verify_id_token
from database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from google.oauth2 import id_token
from google.auth.transport import requests
import uuid
from datetime import datetime, timezone
import os
import io
import base64
import numpy as np
import json
from clases.ImageModel import ImageModel
from clases.Enums import TipoMensaje, Estados
from clases.ChatModel import ChatModel
from clases.UserModel import UserData
from clases.MessageInfo import MessageInfoV2
from clases.CompleteChatResponse import CompleteChatResponse
from chats import add_created_msg, add_received_msg, create_chat_with_image, get_chat_with_id, add_related_images_2_db, get_internal_chat_id
from chats import chatRouter
#from estrategia import Contexto, EstrategiaMensajeSimple, EstrategiaMensajeComplejo

import asyncio
from sse_starlette.sse import EventSourceResponse
from functools import partial

logger = logging.getLogger("uvicorn.error")

class LoginResponse(BaseModel):
    id_token: Optional[str] = None
    status: str
    user: Optional[UserData] = None

class SignupData(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    additional_data: Optional[dict] = None

class CheckLoginRequest(BaseModel):
    google_token: str = Field(..., description="Firebase ID token to verify")

class CheckEmailRequest(BaseModel):
    email: EmailStr

class SignUpWithGoogleRequest(BaseModel):
    google_token: Optional[str] = None
    name: str = Field(..., description="Name of the user")
    email: Optional[str] = None
    password: Optional[str] = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        now = int(datetime.now().timestamp())
        print("DEBUG now/exp delta:", now, payload.get("exp"), payload.get("exp") - now)
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Optimizaciones para GPU con Tensor Cores nativos
torch.backends.cudnn.benchmark = True  # Auto-tuning para tu GPU específica
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para operaciones de matriz

#Creamos la app
app = FastAPI(title="Arte TFG API")
load_dotenv(".env")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
firebase_admin.initialize_app()

#Incluimos el router encargado de la gestion de imagenes (Para obtener paginacion y miniaturas)
app.include_router(imagesRouter)
app.include_router(chatRouter)
client = config.client
qwenModelOld = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
qwenModel = "Qwen/Qwen3-VL-8B-Instruct-FP8"
qwenModelAWQ = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
# --- CORS (Permitir que tu Frontend hable con esto) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

print("Iniciando API...")

collection = get_chroma_collection()

message_queues: dict[str, asyncio.Queue] = {}

def get_queue(user_id: str) -> asyncio.Queue:
    if user_id not in message_queues:
        message_queues[user_id] = asyncio.Queue()
    return message_queues[user_id]

@app.get("/stream")
async def stream(user: dict = Depends(get_current_user_id)):
    queue = get_queue(user)

    async def event_generator():
        try:
            while True:
                message = await queue.get()
                yield {"data": message}
        except asyncio.CancelledError:
            pass  # cliente desconectado

    return EventSourceResponse(event_generator())

@app.get("/")
def root():
    return {"message": "API para mi TFG funcionando.", "endpoints": ["/analyze, /search2, /view"]}

@app.get("/collection-info")
def collection_info():
    view_database()

# --- FUNCIÓN AUXILIAR PARA BÚSQUEDA ---
async def buscar_imagenes_similares(image: Image.Image, n_results: int = 5, session: AsyncSession = Depends(get_session), user: str = ""):
    """
    Función reutilizable que vectoriza una imagen y busca similares en ChromaDB.
    
    Args:
        image: Objeto PIL.Image
        n_results: Número de resultados a devolver
        session: Sesión de base de datos
        user: ID del usuario
    
    Returns:
        Lista de diccionarios con resultados formateados
    """

    query_vector = embedder.encode(image, normalize_embeddings=True).tolist()
    
    query_params = {
        "query_embeddings": [query_vector],
        "n_results": n_results
    }
    
    results = collection.query(**query_params)
    total_results = len(results['ids'][0]) if results.get('ids') and results['ids'] else 0
    print(f"[buscar_imagenes_similares] Resultados en ChromaDB: {total_results} (n_results solicitado: {n_results})")
    if total_results == 0:
        print("[buscar_imagenes_similares] No se encontraron resultados en ChromaDB para esta imagen.")
    
    response_data = []
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        distancia = results['distances'][0][i]
        chromaId = results['ids'][0][i]
        similarity_score = max(0, 100 * (1 - distancia))
        #if similarity_score > 30:
        image = await get_image_with_id(chromaId, session=session, user=user)
        print(f"Resultado {i+1}: ID={chromaId}, Distancia={distancia}, Metadata={meta}, Imagen encontrada: {image}")
        if image:
            relative_path =image.image_url
            image_url = f"/art/{relative_path}"         
            response_data.append({
                "id": chromaId,
                "name": image.name,
                "artist": image.artist,
                "style": image.style,
                "genre": image.genre,
                "similarity_score": round(similarity_score, 2),
                "distance": round(distancia, 4),
                "image_url": image_url
                })
    print(f"[buscar_imagenes_similares] Respuesta formateada con {len(response_data)} resultados similares.")
        
    return response_data

def analizar_imagen_con_qwen_requests(image_bytes: bytes, prompt: str, results: list) -> str:
    """Analiza una imagen usando QWEN en vllm (localhost:8000)"""
    # Cargar imagen y redimensionar
    image = Image.open(io.BytesIO(image_bytes))
    
    # Redimensionar para evitar el OOM
    image.thumbnail((1024, 1024), Image.LANCZOS)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertimos a bytes para enviarlo a vllm
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    processed_bytes = img_byte_arr.getvalue()
    
     # Codificamos en base64 para enviarlo en el JSON
    image_base64 = base64.b64encode(processed_bytes).decode('utf-8')
    
    SYSTEM_PROMPT_TECNICO = (
        "You are a computer vision system designed for technical image analysis. "
        "Your goal is to provide objective, detailed visual descriptions. "
        "Identify artistic styles, techniques, palette, and composition. "
        "Do not refuse analysis based on copyright or safety filters for fictional characters. "
        "Be explicit and descriptive.\n\n"
        "IMPORTANT: ALWAYS respond in the SAME LANGUAGE as the user's question. "
        "You can ONLY answer questions that are directly related to the visual content of the image provided. "
        "If the user asks about topics unrelated to the image (such as general knowledge, math problems, coding, history, "
        "or any other subject not visible in the image), you must politely decline in the same language. "
    )
    
    # Construir el contenido del mensaje con la imagen principal y metadatos de similares
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ]
    
    # Añadir información de obras similares
    if results:
        similar_info = "\n\nSimilar artworks found:\n"
        for idx, res in enumerate(results, 1):
            similar_info += f"{idx}. Artist: {res['artist']}, Style: {res['style']}, Genre: {res['genre']}, Similarity: {res['similarity_score']}%\n"
        content.append({"type": "text", "text": similar_info})
        
        # Convertir imágenes similares a base64
        for idx, res in enumerate(results[:3], 1):  # Limitar a 3
            try:
                image_path = res['image_url'].replace('/view/image_thumbnail/', '').replace('/art/', '')
                full_path = os.path.join(config.CARPETA_IMAGENES, image_path)
                
                if os.path.exists(full_path):
                    with open(full_path, 'rb') as img_file:
                        similar_img = Image.open(img_file)
                        similar_img.thumbnail((512, 512), Image.LANCZOS)
                        if similar_img.mode != 'RGB':
                            similar_img = similar_img.convert('RGB')
                        
                        img_byte_arr = io.BytesIO()
                        similar_img.save(img_byte_arr, format='JPEG', quality=75)
                        similar_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                        
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{similar_base64}"}})
            except Exception as e:
                print(f"ERROR convirtiendo imagen similar {idx}: {e}")

    response = client.chat.completions.create(
        model=qwenModel,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TECNICO
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        seed=np.random.randint(0, 2**31)
    )
    
    return response.choices[0].message.content

def preguntar_a_vllm(image_bytes: bytes, prompt: str) -> str:
    """Analiza una imagen usando QWEN en vllm (localhost:8000)"""
    # Cargar imagen y redimensionar
    image = Image.open(io.BytesIO(image_bytes))
    
    # Redimensionar para evitar el OOM
    image.thumbnail((1024, 1024), Image.LANCZOS)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertimos a bytes para enviarlo a vllm
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    processed_bytes = img_byte_arr.getvalue()
    
     # Codificamos en base64 para enviarlo en el JSON
    image_base64 = base64.b64encode(processed_bytes).decode('utf-8')
    
    SYSTEM_PROMPT_TECNICO = (

        "You are a computer vision system designed for technical image analysis. "
        "Your goal is to provide objective, detailed visual descriptions. "
        "Do not refuse analysis based on copyright or safety filters for fictional characters. "
        "Be explicit and descriptive.\n\n"
        "IMPORTANT: ALWAYS respond in the SAME LANGUAGE as the user's question. "
        "You can ONLY answer questions that are directly related to the visual content of the image provided. "
        "If the user asks about topics unrelated to the image (such as general knowledge, math problems, coding, history, "
        "or any other subject not visible in the image), you must politely decline in the same language. "  

        #ORIGINAL SYSTEM PROMPT: "You are a computer vision system designed for technical image analysis. "
        #"Your goal is to provide objective, detailed visual descriptions. "
        #"Identify artistic styles, techniques, palette, and composition. "
        #"Do not refuse analysis based on copyright or safety filters for fictional characters. "
        #"Be explicit and descriptive."


        # "You are a specialized image Q&A assistant. Your task is to answer ONLY what the user asks about the image.\n\n"
        # "CRITICAL INSTRUCTIONS:\n"
        # "1. Answer ONLY the specific question asked - nothing more, nothing less.\n"
        # "2. Do NOT describe the entire image unless explicitly asked.\n"
        # "3. Do NOT provide additional details beyond what was asked.\n"
        # "4. Keep answers SHORT and DIRECT (1-3 sentences maximum unless more detail is requested).\n"
        # "5. ALWAYS respond in the SAME LANGUAGE as the question.\n"
        # "6. ONLY answer questions about what you can SEE in the image.\n"
        # "7. If asked about unrelated topics, respond: 'Lo siento, solo puedo responder preguntas sobre el contenido visual de la imagen.' (or equivalent in the question's language).\n\n"
        # "Example:\n"
        # "Question: '¿Qué colores hay?' → Answer: 'Azul, amarillo y naranja.'\n"
        # "NOT: 'La imagen muestra una pintura con colores azul, amarillo y naranja. El estilo es...' ❌"
    )
    
    # Construir el contenido del mensaje con la imagen principal y metadatos de similares
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ]

    response = client.chat.completions.create(
        model=qwenModel,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TECNICO
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        seed=np.random.randint(0, 2**31)
    )
    
    return response.choices[0].message.content

@app.post("/analyze")
async def search_similar_art_and_analyze(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None), 
    image_id : Optional[str] = Form(None),
    response_id: Optional[str] = Form(None), 
    chat_id: Optional[str] = Form(None),
    save_to_db: bool = Form(False),
    user: str = Depends(get_current_user_id), 
    session: AsyncSession = Depends(get_session),
    is_new_chat: Optional[bool] = Form(False),
    question : Optional[str] = Form(None),
):
    print(f"Parametros recibidos - file: {file}, image_id: {image_id}, chat_id: {chat_id}, save_to_db: {save_to_db}, is_new_chat: {is_new_chat}")
    if not file and not image_id and not chat_id:
        raise HTTPException(status_code=400, detail="Debe proporcionar un archivo o una ruta.")
    
    messageJson = json.loads(question) if question else None
    mesInfo = MessageInfoV2(**messageJson) if messageJson else None
    print(f"Pregunta parseada con Pydantic: {mesInfo}\n")
    
    if save_to_db:
        chat_internal = None
        image = None
        if is_new_chat:
            print(f"Nuevo chat, creando chat y asociando imagen. is_new_chat: {is_new_chat}, image_id: {image_id}, file: {file}")
            #SI el chat es nuevo, debemos crearlo, por lo que se nos debe proporcionar una imagen ya sea como id pork ya exista en DB o como archivo a subir.
            if image_id and not file:
                print(f"Creando nuevo chat con imagen existente. image_id: {image_id}")
                chat_data = await create_chat_with_image(session=session, image_id=image_id, user=user, external_id=chat_id)
                chat_internal = chat_data.id
                image_data = await get_image_with_id(session=session, image_id=image_id, user=user)
                image_path = os.path.join(config.CARPETA_IMAGENES, image_data.image_url)
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        current_image = img_file.read()
                        image = Image.open(io.BytesIO(current_image))
                else:
                    raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {image_data.image_url}")
            elif image_id and file:
                current_image = await file.read()
                #Guardamos la imagen en el servidor para poder acceder a ella posteriormente y la añadimos a la BD
                image_data = await save_image_and_get_data(session=session, contents=current_image, user=user, commit=False)
                savedImageId = image_data.id
                #Creamos el chat asociado a esta imagen y a este usuario
                chat_data = await create_chat_with_image(session=session, image_id=savedImageId, user=user, external_id=chat_id)
                #await add_equivalence_chat_id(session=session, external_id=local_chat_id, internal_id=chat_data['id'], user_id=user, should_commit=False)
                chat_internal = chat_data.id
                image = Image.open(io.BytesIO(current_image))        
        else:
            #SI el chat ya existe, debemos obtener la imagen asociada a dicho chat.
            print(f"Chat no nuevo, buscando imagen asociada al chat {chat_id} (ID interno: {chat_internal})")
            if chat_id:
                chat_internal = chat_id
                chat_data = await get_chat_with_id(session=session, chat_id=chat_internal, user=user)
                if not chat_data:
                    print(f"No se encontró un chat con ID interno {chat_internal} para el usuario {user}. Intentando buscar equivalencia con ID externo {chat_id}.")
                    chat_data = await get_internal_chat_id(session=session, external_id=chat_id, user_id=user)
                if not chat_data:
                    raise HTTPException(status_code=404, detail=f"Chat no encontrado para el ID proporcionado: {chat_id}")
                
                image_data = await get_image_with_id(session=session, image_id=chat_data.image_id, user=user)
                image_path = os.path.join(config.CARPETA_IMAGENES, image_data.image_url)
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        current_image = img_file.read()
                        image = Image.open(io.BytesIO(current_image))
                else:
                    raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {image_data.image_url}")

        #Ahora insertamos el mensaje con la pregunta, sin hacer commit aún
        message_data = await add_received_msg(session=session, chat_id=chat_internal, message=mesInfo, should_commit=False, question_id=None)
        
        #Ahora lanzamos toda la parafernalia
        error = False
        returned_message = await add_created_msg(session=session, chat_id=chat_internal, response=True, content="Estamos procesando tu solicitud...", should_commit=False, question_id=message_data.id, status=Estados.ERROR if error else Estados.WAITING, type=message_data.type)        
        await session.commit()
        
        response = CompleteChatResponse(
            chat_info=chat_data,
            request_info=message_data,
            message_info=returned_message,
            image_info=image_data
        )

        print(f"RESPUESTA FINAL QUE SE ENVIA AL FRONT: {response}")
        
        if message_data.type == TipoMensaje.SIMPLE:
            # Lanzamos la tarea en segundo plano para procesar la pregunta simple y actualizar la respuesta posteriormente
            print("Lanzando tarea en background para pregunta simple...")   
            background_tasks.add_task(
                procesar_mensaje_simple_recibido,
                message_data,
                returned_message,
                chat_data,
                image_data,
            )
        elif message_data.type == TipoMensaje.COLORS:
            # Lanzamos la tarea en segundo plano para procesar la extracción de colores y actualizar la respuesta posteriormente
            print("Lanzando tarea en background para extracción de colores...")
        elif message_data.type == TipoMensaje.COMPLETE:
            # Lanzamos la tarea en segundo plano para procesar el mensaje completo y actualizar la respuesta posteriormente
#            contexto = Contexto()
#            contexto.establecer_estrategia(EstrategiaMensajeComplejo())
            background_tasks.add_task(
                procesar_mensaje_complejo_recibido,
                message_data,
                returned_message,
                chat_data,
                image_data,
            )

        return response
    else:
        try:
            if image_id:
                #TODO el metodo para obtener la imagen a partir de una existente y no guardar en BD el chat.
                raise HTTPException(
                    status_code=400,
                    detail="El flujo save_to_db=false con image_id aún no está implementado. Envia file o activa save_to_db."
                )
            elif file:
                current_image = await file.read()
                image = Image.open(io.BytesIO(current_image))
            else:
                raise HTTPException(status_code=400, detail="Debes proporcionar file o image_id.")

            #Ahora lanzamos toda la parafernalia
            results = await buscar_imagenes_similares(image, n_results=3, session=session, user=user)

            # Analizar imagen con QWEN
            loop = asyncio.get_event_loop()
            qwen_description = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(
                        analizar_imagen_con_qwen_requests,
                        current_image,
                        "Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, Comparacion rapida con las similares en caso de haber.",
                        results,
                    ),
                ),
                timeout=120,
            )
        except HTTPException:
            raise
        except asyncio.TimeoutError:
            print("Timeout al llamar a QWEN en flujo save_to_db=false")
            qwen_description = "Tiempo de espera agotado al analizar la imagen con QWEN."
            results = []
        except Exception as e:
            print(f"Error al llamar a QWEN: {e}")
            qwen_description = "Error al analizar la imagen con QWEN."
            results = []
        
        print(f"Descripción obtenida de QWEN: {qwen_description}")
        
        response = CompleteChatResponse(
            chat_info=None,
            image_info=None,
            message_info=MessageInfoV2(
                id=str(uuid.uuid4()),
                chat_id=chat_id if chat_id else "no_chat",
                response=True,
                content=qwen_description,
                created_at=datetime.now(timezone.utc),
                related_images=results
            )
        )

        return response

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
            # raise and catch to return 401, only needed because fastapi returns 403
            # by default instead of 401 so we set auto_error to False
            raise ValueError("No token")
        # Añadimos un margen de 60 segundos para la validación del token
        user = verify_id_token(token.credentials, clock_skew_seconds=60)
        return user
    # lots of possible exceptions, see firebase_admin.auth,
    # but most of the time it is a credentials issue
    except Exception:
        # we also set the header
        # see https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/
        print("Error verificando token de Firebase. Asegúrate de enviar un token válido en el formato 'Bearer <token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not logged in or Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
@app.post("/login")
async def get_userid(firebase_user: Annotated[dict, Depends(get_firebase_user_from_token)], session: AsyncSession = Depends(get_session)):
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
        internal_id = usuario_db['id']
        name = usuario_db['name']
        token = config.create_access_token(str(internal_id))
        print(f"Usuario con ID interno: {internal_id} ha iniciado sesión.")
    else:
        internal_id = -1
        print(f"Usuario con UID Firebase: {user['uid']} no encontrado en la base de datos.")
        raise HTTPException(status_code=404, detail="Usuario no encontrado en la base de datos. Por favor, regístrate primero.")
    return {"id": internal_id, "name": name, "email": user["email"], "token": token}

 #Version con verificacion de token de Google directamente, evitando crear usuario en Firebase hasta que se complete el signup

@app.post("/check-login", response_model=LoginResponse)
async def check_user(request: CheckLoginRequest, session: AsyncSession = Depends(get_session),):
    try:
        # Verificamos el token de Google y obtenemos el email
        print(f"Verificando token de Google: {request}")
        id_info = id_token.verify_oauth2_token(request.google_token, requests.Request(), GOOGLE_CLIENT_ID)
        email = id_info.get('email')
        print(f"Token válido para el usuario: {email}")

        # Lo buscamos en nuestra base de datos
        query = text("SELECT * FROM users WHERE email = :email")
        result = await session.execute(query, {"email": email})
        usuario_db = result.mappings().one_or_none()
        print(usuario_db)
        # Si existe, obtenemos su firebase_uid y creamos el custom token (No se si seria mejor enviar el usuario completo y ahorrar una llamada en el login)
        if usuario_db:
            #creamos un user del tipo UserData y los datos recibidos de la base de datos
            user_data = UserData(
                internal_id=str(usuario_db['id']), name=usuario_db['name'], email=usuario_db['email'], avatar=usuario_db['profile_icon'])
            print(f"Usuario encontrado: {user_data}")
            token = config.create_access_token(user_data.internal_id) 
            return LoginResponse(id_token=token, status="login", user=user_data)
            #return {"id_token": custom_token.decode('utf-8'), "status": "login"}
        else:
            return LoginResponse(id_token=None, status="signup", user=None)
            #return {"id_token": None, "status": "signup"}
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error :(")

@app.post("/check-email")
async def check_email_exists(
    request: CheckEmailRequest, 
    session: AsyncSession = Depends(get_session)
):
    try:
        # Convertimos a minúsculas para asegurar consistencia
        email_clean = request.email.lower().strip()
        
        # 2. Consulta optimizada: SELECT 1 ... LIMIT 1
        # Esto es más rápido que COUNT(*) porque se detiene al encontrar el primero
        query = text("SELECT 1 FROM users WHERE email = :email LIMIT 1")
        
        result = await session.execute(query, {"email": email_clean})
        
        # scalar_one_or_none devolverá 1 si existe, o None si no.
        user_exists = result.scalar_one_or_none() is not None
        
        logger.info(f"Verificación de email '{email_clean}': {'Existe' if user_exists else 'Disponible'}")
        return {"exist": user_exists}

    except Exception as e:
        # 3. Logueamos el error REAL para que tú lo veas en la consola
        logger.error(f"Error verificando email: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno verificando el email")

@app.post("/perfected-signup")
async def signup_user_google(request: SignUpWithGoogleRequest, session: AsyncSession = Depends(get_session)):
    try:
        custom_token = None
        print(f"Datos recibidos para signup: {request}")
        if (not request.google_token and request.password and request.email):
            user_record = auth.create_user(email=request.email, password=request.password, display_name=request.name, email_verified=True)
            firebase_uid = user_record.uid
            insert_sql = text("""
                INSERT INTO users (firebase_uid, name, email)
                VALUES (:uid, :name, :email)
                ON CONFLICT (firebase_uid) DO NOTHING
                RETURNING id, name, email;
            """)
            result = await session.execute(insert_sql, {
                "uid": firebase_uid,
                "name": request.name,
                "email": request.email
            })
            await session.commit()
            internal_id = result.scalar_one()
            print(f"Usuario creado: {request.name}, UID: {firebase_uid}, ID interno: {internal_id}")
            custom_token = auth.create_custom_token(firebase_uid)
        
        elif(request.google_token):
            id_info = id_token.verify_oauth2_token(request.google_token, requests.Request(), GOOGLE_CLIENT_ID, clock_skew_in_seconds=60)
            email = id_info.get('email')
            user_record = auth.create_user(email=email, display_name=request.name)
            firebase_uid = user_record.uid
            insert_sql = text("INSERT INTO users (firebase_uid, name, email) VALUES (:uid, :name, :email) ON CONFLICT (firebase_uid) DO NOTHING RETURNING id, name, email;")
            result = await session.execute(insert_sql, {
                "uid": firebase_uid,
                "name": request.name,
                "email": user_record.email
            })
            await session.commit()
            internal_id = result.scalar_one()
            print(f"Usuario creado: {request.name}, UID: {firebase_uid}, ID interno: {internal_id}")
            custom_token = auth.create_custom_token(firebase_uid)
        return {"id_token": custom_token.decode('utf-8'), "status": "login"}
        
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
            # ¡IMPORTANTE! Imprime el error real para verlo en los logs
            print(f"ERROR REAL DEL SERVIDOR: {str(e)}")
            # Para desarrollo, puedes devolver el detalle del error al cliente
            raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
            
async def procesar_mensaje_complejo_recibido(receiveMessage: MessageInfoV2, newMessage: MessageInfoV2, chat : ChatModel, image: ImageModel):    
    async with AsyncSessionLocal() as session:
        try:
            print(f"✅ Procesando mensaje recibido con ID {receiveMessage.id}, y su respuesta {newMessage.id} para el chat ID {chat.id} y la imagen ID {image.id}")
            # Cargar la imagen desde disco usando la ruta del ImageModel
            full_path = os.path.join(config.CARPETA_IMAGENES, image.image_url)
            with open(full_path, 'rb') as img_file:
                current_image = img_file.read()
            
            # Convertir a PIL Image para la búsqueda
            pil_image = Image.open(io.BytesIO(current_image))
            
            # Buscar imágenes similares
            print("[procesar_mensaje_complejo_recibido] Ejecutando búsqueda de imágenes similares...")
            results = await buscar_imagenes_similares(pil_image, n_results=3, session=session, user=chat.user_id)
            print(f"[procesar_mensaje_complejo_recibido] Búsqueda completada. Resultados útiles: {len(results)}")
            
            # Analizar con QWEN
            loop = asyncio.get_event_loop()
            qwen_description = await loop.run_in_executor(
                None,  # usa el ThreadPoolExecutor por defecto
                partial(
                    analizar_imagen_con_qwen_requests,
                    current_image,
                    """Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. 
                    El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, 
                    Comparacion rapida con las similares en caso de haber.""",
                    results
                )
            )
            status = Estados.SUCCESS
            
        except Exception as e:
            print(f"❌ Error al procesar imagen: {e}")
            qwen_description = "Error al analizar la imagen con QWEN."
            results = []
            status = Estados.ERROR
        
        # Actualizar el mensaje en la base de datos
        try:
            query_update = text("""
                UPDATE messages 
                SET content = :content, status = :status
                WHERE id = :message_id
            """)
            await session.execute(query_update, {
                "content": qwen_description,
                "status": status.value,
                "message_id": newMessage.id
            })
            
            # Guardar imágenes relacionadas si hay
            if results:
                await add_related_images_2_db(
                    message_id=newMessage.id,
                    related_images=results,
                    session=session,
                    should_commit=False
                )
            
            await session.commit()
            print(f"✅ Procesamiento completado para mensaje {newMessage.id}")
            queue = get_queue(chat.user_id)
            result = MessageInfoV2(
                id=str(newMessage.id),
                chat_id=str(newMessage.chat_id),
                response=newMessage.response,
                content=qwen_description,
                created_at=newMessage.created_at,
                status=status,
                related_images=results,
                question_id=newMessage.question_id
            )
            fin = CompleteChatResponse(
                message_info=result,
                request_info=receiveMessage,
                chat_info=chat,
                image_info=image
            )
            print(f"Enviando resultado al cliente a través de SSE: {fin}")
            await queue.put(fin.model_dump_json())
        except Exception as e:
            print(f"❌ Error al actualizar la base de datos: {e}")
            await session.rollback()
            
async def procesar_mensaje_simple_recibido(receiveMessage: MessageInfoV2, newMessage: MessageInfoV2, chat : ChatModel, image: ImageModel):    
    async with AsyncSessionLocal() as session:
        try:
            print(f"✅ Procesando mensaje SIMPLE recibido con ID {receiveMessage.id}, y su respuesta {newMessage.id} para el chat ID {chat.id} y la imagen ID {image.id}")
            # Cargar la imagen desde disco usando la ruta del ImageModel
            full_path = os.path.join(config.CARPETA_IMAGENES, image.image_url)
            with open(full_path, 'rb') as img_file:
                current_image = img_file.read()
            
            # Convertir a PIL Image para la búsqueda
            pil_image = Image.open(io.BytesIO(current_image))
            print("[procesar_mensaje_simple_recibido] Flujo SIMPLE: no se ejecuta buscar_imagenes_similares.")
            
            # Analizar con QWEN
            loop = asyncio.get_event_loop()
            qwen_description = await loop.run_in_executor(
                None,  # usa el ThreadPoolExecutor por defecto
                partial(
                    preguntar_a_vllm,
                    current_image,
                    receiveMessage.content
                )
            )
            status = Estados.SUCCESS
            
        except Exception as e:
            print(f"❌ Error al procesar imagen: {e}")
            qwen_description = "Error al analizar la imagen con QWEN."
            status = Estados.ERROR
        
        # Actualizar el mensaje en la base de datos
        try:
            query_update = text("""
                UPDATE messages 
                SET content = :content, status = :status
                WHERE id = :message_id
            """)
            await session.execute(query_update, {
                "content": qwen_description,
                "status": status.value,
                "message_id": newMessage.id
            })
            
            await session.commit()
            print(f"✅ Procesamiento completado para mensaje {newMessage.id}")
            queue = get_queue(chat.user_id)
            result = MessageInfoV2(
                id=str(newMessage.id),
                chat_id=str(newMessage.chat_id),
                response=newMessage.response,
                content=qwen_description,
                created_at=newMessage.created_at,
                status=status,
                question_id=newMessage.question_id
            )
            fin = CompleteChatResponse(
                message_info=result,
                request_info=receiveMessage,
                chat_info=chat,
                image_info=image
            )
            print(f"Enviando resultado al cliente a través de SSE: {fin}")
            await queue.put(fin.model_dump_json())
            
        except Exception as e:
            print(f"❌ Error al actualizar la base de datos: {e}")
            await session.rollback()

def retry_message_processing(receiveMessage: MessageInfoV2, newMessage: MessageInfoV2, chat : ChatModel, image: ImageModel):
    print(f"Reintentando procesamiento para mensaje {newMessage.id}")
    if newMessage.type == TipoMensaje.SIMPLE:
        asyncio.create_task(procesar_mensaje_simple_recibido(receiveMessage, newMessage, chat, image))
    else:
        asyncio.create_task(procesar_mensaje_complejo_recibido(receiveMessage, newMessage, chat, image))

@app.post("/retry")
async def retry_endpoint(
    background_tasks: BackgroundTasks,
    response_id: Optional[str] = Form(None), 
    chat_id: Optional[str] = Form(None),
    user: str = Depends(get_current_user_id), 
    session: AsyncSession = Depends(get_session)
):
    print(f"Parametros recibidos para retry - response_id: {response_id}, chat_id: {chat_id}")
    sql_chat = text("SELECT * FROM chats WHERE id = :chat_id AND user_id = :user_id")
    result = await session.execute(sql_chat, {"chat_id": chat_id, "user_id": user})
    chat_db = result.mappings().one_or_none()
    if not chat_db:
        raise HTTPException(status_code=404, detail="Chat no encontrado para el ID proporcionado")
    sql_resp = text("SELECT * FROM messages WHERE id = :response_id AND chat_id = :chat_id")
    result = await session.execute(sql_resp, {"response_id": response_id, "chat_id": chat_id})
    resp_db = result.mappings().one_or_none()
    if not resp_db:
        raise HTTPException(status_code=404, detail="Respuesta no encontrada para el ID proporcionado")
    sql_preg = text("SELECT * FROM messages WHERE id = :question_id AND chat_id = :chat_id")
    result = await session.execute(sql_preg, {"question_id": resp_db['question_id'], "chat_id": chat_id})
    preg_db = result.mappings().one_or_none()
    if not preg_db:
        raise HTTPException(status_code=404, detail="Pregunta no encontrada para el ID proporcionado")
    image = await get_image_with_id(session=session, image_id=chat_db['image_id'], user=user)
    update_resp = text("UPDATE messages SET status = 'WAITING' WHERE id = :response_id")
    await session.execute(update_resp, {"response_id": response_id})
    await session.commit()
    chat = ChatModel(
        id=str(chat_db['id']),
        user_id=str(chat_db['user_id']),
        image_id=str(chat_db['image_id']),
        created_at=chat_db['created_at'],
    )
    question_message = MessageInfoV2(
        id=str(preg_db['id']),
        chat_id=str(preg_db['chat_id']),
        response=preg_db['response'],
        content=preg_db['content'],
        created_at=preg_db['created_at'],
        status=preg_db['status'],
        type=preg_db['type']
    )
    receive_message = MessageInfoV2(
        id=str(resp_db['id']),
        chat_id=str(resp_db['chat_id']),
        response=resp_db['response'],
        content=resp_db['content'],
        created_at=resp_db['created_at'],
        status=Estados.WAITING,
        question_id=str(resp_db['question_id']) if resp_db['question_id'] else None,
        type=resp_db['type']
    )
    
    if receive_message.type == TipoMensaje.SIMPLE:
            # Lanzamos la tarea en segundo plano para procesar la pregunta simple y actualizar la respuesta posteriormente
            print("Lanzando tarea en background para pregunta simple...")   
            background_tasks.add_task(
                procesar_mensaje_simple_recibido,
                question_message,
                receive_message,
                chat,
                image,
            )
    elif receive_message.type == TipoMensaje.COLORS:
        # Lanzamos la tarea en segundo plano para procesar la extracción de colores y actualizar la respuesta posteriormente
        print("Lanzando tarea en background para extracción de colores...")
    elif receive_message.type == TipoMensaje.COMPLETE:
        # Lanzamos la tarea en segundo plano para procesar el mensaje completo y actualizar la respuesta posteriormente
        # contexto = Contexto()
        # contexto.establecer_estrategia(EstrategiaMensajeComplejo())
        background_tasks.add_task(
            procesar_mensaje_complejo_recibido,
            question_message,
            receive_message,
            chat,
            image,
        )
        
    return receive_message


# Buscamos el mensaje original y su respuesta en la base de datos para obtener toda la información necesaria para reintentar el procesamiento
# Ademas del chat y la imagen asociados
# Una vez tenemos todo, se comprueba el tipo de mensaje y se llama al metodo correspondiente.