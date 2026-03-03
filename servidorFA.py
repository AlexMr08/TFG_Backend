from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, OAuth2PasswordBearer
from jose import jwt, JWTError
from PIL import Image, ImageOps
import io
import config
import torch
import numpy as np
import base64
import logging
from pydantic import BaseModel, EmailStr, Field, model_validator
from typing import Annotated, Optional
import os
from model_loader import embedder
from images import imagesRouter
from database import get_chroma_collection, view_database
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


GOOGLE_CLIENT_ID = "214539922631-sqhb8bkibaq8ai4ke593qa2u4h9cko3q.apps.googleusercontent.com"
logger = logging.getLogger("uvicorn.error")

class UserData(BaseModel):
    internal_id: str      # Tu ID de SQL
    name: str             # Obligatorio
    email: Optional[str] = None      # Opcional (puede ser None)
    avatar: Optional[str] = None  # Opcional (puede ser None)

#         chat_dict['id'] = str(chat_dict['id'])
#        chat_dict['user_id'] = str(chat_dict['user_id'])
#        chat_dict['image_id'] = str(chat_dict['image_id'])
#        chat_dict['created_at'] = chat_dict['created_at'].isoformat()
#        chat_dict['local_route'] = chat_dict['local_route']
#        chat_dict['image_name'] = chat_dict['image_name']
#        chat_dict['image_url'] = f"/art/{chat_dict['local_route']}"
#        chat_dict['last_message'] = chat_dict['last_message']

class ChatModel(BaseModel):
    id: str
    user_id: str
    image_id: Optional[str] = None
    collection_id: Optional[str] = None
    created_at: str
    external_id: Optional[str] = None
    status: Optional[str] = None

    @model_validator(mode='after')
    def check_exclusive_ids(self):
        """Valida que solo uno de image_id o collection_id esté presente"""
        has_image = self.image_id is not None
        has_collection = self.collection_id is not None
        
        if has_image and has_collection:
            raise ValueError("Un chat no puede tener image_id y collection_id simultáneamente")
        
        if not has_image and not has_collection:
            raise ValueError("Un chat debe tener image_id o collection_id")
        
        return self

class MessageInfoV2(BaseModel):
    id: str
    chat_id: str
    response: bool
    content: str
    created_at: str
    status: Optional[str] = "success"


class ImageModel(BaseModel):
    id: str
    name: Optional[str] = "Unknown"
    artist: Optional[str] = "Unknown"
    style: Optional[str] = "Unknown"
    genre: Optional[str] = "Unknown"
    year: Optional[str] = "Unknown"
    owner_id: Optional[str] = None
    image_url: str

class CompleteChatResponse(BaseModel):
    chat_info: Optional[ChatModel] = None
    message_info: Optional[dict] = None
    image_info: Optional[ImageModel] = None

class ChatIdEquivalence(BaseModel):
    external_id: str
    internal_id: str

class ChatListResponse(BaseModel):
    chats: list[ChatModel]
    equivalences: list[ChatIdEquivalence] = []

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
        print("DEBUG: Payload decodificado del token:", payload)
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Optimizaciones para GPU con Tensor Cores nativos
torch.backends.cudnn.benchmark = True  # Auto-tuning para tu GPU específica
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para operaciones de matriz

#Creamos la app
app = FastAPI(title="Arte TFG API")
load_dotenv(".env")
firebase_admin.initialize_app()

#Incluimos el router encargado de la gestion de imagenes (Para obtener paginacion y miniaturas)
app.include_router(imagesRouter)
client = config.client
qwenModelOld = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
qwenModel = "Qwen/Qwen3-VL-8B-Instruct-FP8"

# --- CORS (Permitir que tu Frontend hable con esto) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se restringe, para TFG déjalo así
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- 1. CARGAR CHROMADB ---
print("Iniciando API...")

collection = get_chroma_collection()


@app.get("/")
def root():
    return {"message": "API para mi TFG funcionando.", "endpoints": ["/analyze, /search2, /view"]}

@app.get("/collection-info")
def collection_info():
    view_database()

# --- FUNCIÓN AUXILIAR PARA BÚSQUEDA ---
def buscar_imagenes_similares(image: Image.Image, n_results: int = 5):
    """
    Función reutilizable que vectoriza una imagen y busca similares en ChromaDB.
    
    Args:
        image: Objeto PIL.Image
        n_results: Número de resultados a devolver
    
    Returns:
        Lista de diccionarios con resultados formateados
    """

    query_vector = embedder.encode(image, normalize_embeddings=True).tolist()
    
    query_params = {
        "query_embeddings": [query_vector],
        "n_results": n_results
    }
    
    results = collection.query(**query_params)
    
    response_data = []
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        distancia = results['distances'][0][i]
        
        # CONSTRUIR LA URL
        relative_path = meta['filepath']
        image_url = f"/art/{relative_path}" 
        
        similarity_score = max(0, 100 * (1 - distancia))
        #if similarity_score > 30:
        response_data.append({
            "id": results['ids'][0][i],
            "artist": meta['artist'],
            "style": meta['style'],
            "genre": meta['genre'],
            "similarity_score": round(similarity_score, 2),
            "distance": round(distancia, 4),
            "image_url": image_url
            })
        
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
        "Identify artistic styles, techniques, palette, and composition. "
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
    file: Optional[UploadFile] = File(None), 
    image_id : Optional[str] = Form(None), 
    chat_id: Optional[str] = Form(None),
    save_to_db: bool = Form(False),
    user: str = Depends(get_current_user_id), 
    session: AsyncSession = Depends(get_session),
    is_new_chat: Optional[bool] = Form(False),
    question : Optional[str] = Form(None),
    local_chat_id: Optional[str] = Form(None)
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
            if image_id:
                chat_data = await create_chat_with_image(session=session, image_id=image_id, user=user, external_id=chat_id)
                #await add_equivalence_chat_id(session=session, external_id=local_chat_id, internal_id=chat_data['id'], user_id=user, should_commit=False)
                chat_internal = chat_data.id
                image_data = await get_image_with_id(session=session, image_id=image_id, user=user)
                image_path = os.path.join(config.CARPETA_IMAGENES, image_data.image_url)
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        current_image = img_file.read()
                        image = Image.open(io.BytesIO(current_image))
                else:
                    raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {image_data.image_url}")
            elif file:
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
        #         message_data = await addMsg2BD(session=session, chat_id=chat_internal, response=False, content="Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales.", should_commit=False, question_id=None)
        message_data = await add_received_msg(session=session, chat_id=chat_internal, message=mesInfo, should_commit=False, question_id=None)
        #Ahora lanzamos toda la parafernalia

        results = buscar_imagenes_similares(image, n_results=3)

        # Analizar imagen con QWEN
        try:
            qwen_description = analizar_imagen_con_qwen_requests(
                current_image,
                "Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, Comparacion rapida con las similares en caso de haber.",
                results
            )
        except Exception as e:
            print(f"Error al llamar a QWEN: {e}")
            # Si hay error y estamos guardando en BD, hacer rollback
            await session.rollback()
            raise HTTPException(status_code=500, detail="Error al procesar la imagen con QWEN")
            
        returned_message = await add_created_msg(session=session, chat_id=chat_internal, response=True, content=qwen_description or "Error al analizar la imagen con QWEN.", should_commit=False, question_id=message_data.id)
        
        await add_related_images_2_db(message_id=returned_message['id'], related_images=results, session=session, should_commit=True)
        
        await session.commit()

        returned_message['related_images'] = results

        res_img = image_data

        res_chat = chat_data

        response = CompleteChatResponse(
            chat_info=res_chat,
            message_info=returned_message,
            image_info=res_img
        )

        print(f"RESPUESTA FINAL QUE SE ENVIA AL FRONT: {response}")

        return response
    else:
        if image_id:
            #TODO el metodo para obtener la imagen a partir de una existente y no guardar en BD el chat.
            print("TODO")
        elif file:
            current_image = await file.read()
            image = Image.open(io.BytesIO(current_image))

        #Ahora lanzamos toda la parafernalia

        results = buscar_imagenes_similares(image, n_results=3)

        # Analizar imagen con QWEN
        try:
            qwen_description = analizar_imagen_con_qwen_requests(
                current_image,
                "Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, Comparacion rapida con las similares en caso de haber.",
                results
            )
        except Exception as e:
            print(f"Error al llamar a QWEN: {e}")
            # Si hay error y estamos guardando en BD, hacer rollback
            raise HTTPException(status_code=500, detail="Error al procesar la imagen con QWEN")
            
        response = CompleteChatResponse(
            chat_info=None,
            image_info=None,
            message_info={
                "content": qwen_description,
                "related_images": results
            }
        )

        return response

@app.post("/search2")
async def search_similar_art2(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    print(f"Texto recibido para filtro: {text}")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Usar función de búsqueda
    results = buscar_imagenes_similares(image, n_results=5)
    
    return {"results": results}

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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not logged in or Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
@app.post("/signup")
async def signup_user(
    firebase_user: Annotated[dict, Depends(get_firebase_user_from_token)], 
    session: AsyncSession = Depends(get_session), 
    user_data: SignupData = SignupData()
):
    """signs up a new firebase user"""
    print(firebase_user)
    print(f"Datos recibidos: {user_data}")
    
    # Usar datos del JSON si están disponibles, sino usar los de firebase
    user = {
        "uid": firebase_user["uid"],
        "name": user_data.name or firebase_user.get("name", "Unknown"),
        "email": user_data.email or firebase_user.get("email", "Unknown"),
        }
    
    insert_sql = text("""
        INSERT INTO users (firebase_uid, name, email)
        VALUES (:uid, :name, :email)
        ON CONFLICT (firebase_uid) DO NOTHING 
        RETURNING id, name;
    """)
    result = await session.execute(insert_sql, user)
    await session.commit()
    internal_id = result.scalar_one()
    print(f"Usuario Firebase creado: {user['name']}, UID: {firebase_user['uid']}, ID interno: {internal_id}")
    return {"id": internal_id, "name": user["name"], "email": user["email"], "exist": True}

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
    token = "None"
    exist = True
    if usuario_db:
        internal_id = usuario_db['id']
        name = usuario_db['name']
        token = config.create_access_token(str(internal_id))
        print(f"Usuario con ID interno: {internal_id} ha iniciado sesión.")
    else:
        exist = False
        internal_id = -1
        print(f"Usuario con UID Firebase: {user['uid']} no encontrado en la base de datos.")
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
            firebase_uid = usuario_db['firebase_uid']
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
        print(f"Datos recibidos para signup: {request}")
        if (not request.google_token and request.password and request.email):
            user_record = auth.create_user(email=request.email, password=request.password, display_name=request.name, email_verified=True)
            firebase_uid = user_record.uid
            insert_sql = text("""
                INSERT INTO users (firebase_uid, name, email)
                VALUES (:uid, :name, :email)
                ON CONFLICT (firebase_uid) DO NOTHING
                RETURNING id, name;
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
            return {"id_token": custom_token.decode('utf-8'), "status": "login"}
        
        elif(request.google_token):
            id_info = id_token.verify_oauth2_token(request.google_token, requests.Request(), GOOGLE_CLIENT_ID, clock_skew_in_seconds=60)
            email = id_info.get('email')
            user_record = auth.create_user(email=email, display_name=request.name)
            firebase_uid = user_record.uid
            insert_sql = text("INSERT INTO users (firebase_uid, name, email) VALUES (:uid, :name, :email) ON CONFLICT (firebase_uid) DO NOTHING RETURNING id, name;")
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
    
@app.post("/question-answer")
async def question_answer(
    file: Optional[UploadFile] = File(None),
    question: str = Form(...),
    chat_id: Optional[str] = Form(None),
    user: dict = Depends(get_current_user_id),
):
    if not file and not chat_id:
        raise HTTPException(status_code=400, detail="Debe proporcionar un archivo o un ID de chat.")

    if file: 
        contents = await file.read()
    else:

        contents = None
    
    # Analizar imagen con QWEN
    try:
        qwen_description = preguntar_a_vllm(
            contents,
            question
        )
    except Exception as e:
        print(f"Error al llamar a QWEN: {e}")
        qwen_description = None
    
    return {
        "qwen_analysis": qwen_description
    }

@app.get("/me/chats/{chat_id}")
async def get_chat_details(chat_id: str, user: dict = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)):
    query_chat = text("""
        SELECT C.*, i.local_route, i.name AS image_name
        FROM chats AS C 
        INNER JOIN images AS i ON C.image_id = i.id
        WHERE C.id = :chat_id AND C.user_id = :user_id
    """)
    result_chat = await session.execute(query_chat, {"chat_id": chat_id, "user_id": user})
    chat = result_chat.mappings().one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado o no pertenece al usuario")
    
    chat_dict = dict(chat)
    chat_details = ChatModel(
        id=str(chat_dict['id']), 
        user_id=str(chat_dict['user_id']), 
        image_id=str(chat_dict['image_id']), 
        collection_id="",
        created_at=chat_dict['created_at'].isoformat(),
        status=chat_dict['status']
    )
    
    return chat_details

@app.get("/me/chats2")
async def get_user_chats2(last_date : Optional[str] = None, user: dict = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)) -> ChatListResponse:
    query_chats = text("""
        SELECT C.*
        FROM chats AS C WHERE C.user_id = :user_id
        ORDER BY C.created_at DESC
    """)
    result_chats = await session.execute(query_chats, {"user_id": user, "last_date": last_date})
    chats = result_chats.mappings().all()
    
    # Obtener equivalencias del usuario
    query_equivalences = text("""
        SELECT external_id, internal_id
        FROM chat_id_equivalences
        WHERE user_id = :user_id
    """)
    result_equivalences = await session.execute(query_equivalences, {"user_id": user})
    equivalences = result_equivalences.mappings().all()
    
    chatsRes = ChatListResponse(chats=[], equivalences=[])
    for chat in chats:
        chat_dict = dict(chat)
        new_chat = ChatModel(
            id=str(chat_dict['id']), 
            user_id=str(chat_dict['user_id']), 
            image_id=str(chat_dict['image_id']), 
            created_at=chat_dict['created_at'].isoformat(),
            status=chat_dict['status']
        )
        chatsRes.chats.append(new_chat)
    
    for eq in equivalences:
        eq_dict = dict(eq)
        new_eq = ChatIdEquivalence(
            external_id=str(eq_dict['external_id']),
            internal_id=str(eq_dict['internal_id'])
        )
        chatsRes.equivalences.append(new_eq)
    
    print(f"Chats encontrados para el usuario {user}: {chatsRes}")
    return chatsRes

@app.get("/me/chats")
async def get_user_chats(last_date : Optional[str] = None, user: dict = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)) -> ChatListResponse:
    query_chats = text("""
        SELECT C.*
        FROM chats AS C 
        WHERE C.user_id = :user_id
        ORDER BY C.created_at DESC
    """)
    result_chats = await session.execute(query_chats, {"user_id": user, "last_date": last_date})
    chats = result_chats.mappings().all()
    
    chatsRes = []
    for chat in chats:
        chat_dict = dict(chat)
        new_chat = ChatModel(
            id=str(chat_dict['id']), 
            user_id=str(chat_dict['user_id']), 
            image_id=str(chat_dict['image_id']), 
            created_at=chat_dict['created_at'].isoformat(),
            external_id=str(chat_dict['external_id']),
            status=chat_dict['status']
        )
        chatsRes.append(new_chat)
    
    print(f"Chats encontrados para el usuario {user}: {chatsRes}")
    return ChatListResponse(chats=chatsRes)

@app.get("/me/chats23")
async def get_user_chats(last_date : Optional[str] = None, user: dict = Depends(get_current_user_id), session: AsyncSession = Depends(get_session)) -> ChatListResponse:
    query_chats = text("""
        SELECT C.*, i.local_route, i.name AS image_name, m.content as last_message
        FROM chats AS C 
        INNER JOIN images AS i ON C.image_id = i.id
        LEFT JOIN LATERAL (
            SELECT content 
            FROM messages 
            WHERE chat_id = C.id 
            ORDER BY created_at DESC 
            LIMIT 1
        ) m ON true
        WHERE C.user_id = :user_id
        ORDER BY C.created_at DESC
    """)
    result_chats = await session.execute(query_chats, {"user_id": user, "last_date": last_date})
    chats = result_chats.mappings().all()
    chatsRes = ChatListResponse(chats=[])
    for chat in chats:
        chat_dict = dict(chat)
        new_chat = ChatModel(
            id=str(chat_dict['id']), 
            user_id=str(chat_dict['user_id']), 
            image_id=str(chat_dict['image_id']), 
            created_at=chat_dict['created_at'].isoformat(), 
            local_route=chat_dict['local_route'], 
            image_name=chat_dict['image_name'], 
            image_url=f"/art/{chat_dict['local_route']}", 
            last_message=chat_dict['last_message'],
            status=str(chat_dict['status'])
        )
        chatsRes.chats.append(new_chat)
    print(f"Chats encontrados para el usuario {user}: {chatsRes}")
    return chatsRes

@app.get("/chats/messages")
async def get_chat_messages(
    chat_id: Optional[str] = None,
    user: dict = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session)
):
    """Obtiene todos los mensajes de un chat específico"""
    # Verificar que el chat pertenece al usuario
    chat_details = await get_chat_with_id(session=session, chat_id=chat_id, user=user)
    if not chat_details:
        chat_details = await get_internal_chat_id(session=session, external_id=chat_id, user_id=user)
    if not chat_details:
        raise HTTPException(status_code=404, detail="Chat no encontrado o no pertenece al usuario")
    
    # Obtener mensajes ordenados por fecha de creación
    query_msgs = text("""
        SELECT id, chat_id, response, content, created_at 
        FROM messages 
        WHERE chat_id = :chat_id 
        ORDER BY created_at DESC
    """)
    result_msgs = await session.execute(query_msgs, {"chat_id": chat_details.id})
    messages = result_msgs.mappings().all()

    messages_res = []
    for msg in messages:
        msg_dict = dict(msg)
        msg_dict['id'] = str(msg_dict['id'])
        msg_dict['chat_id'] = str(msg_dict['chat_id'])
        msg_dict['created_at'] = msg_dict['created_at'].isoformat()
        
        # Obtener imágenes relacionadas con este mensaje
        query_related = text("""
            SELECT ri.id as relation_id, ri.similarity, 
                   i.id, i.local_route, i.name, i.artist, i.style, i.genre, i.year
            FROM related_images ri
            JOIN images i ON ri.image_id = i.id
            WHERE ri.message_id = :message_id
            ORDER BY ri.similarity DESC
        """)
        result_related = await session.execute(query_related, {"message_id": msg['id']})
        related_images = result_related.mappings().all()
        
        # Formatear imágenes relacionadas
        related_images_list = []
        for img in related_images:
            img_dict = {
                'id': str(img['id']),
                'name': img['name'],
                'artist': img['artist'],
                'style': img['style'],
                'genre': img['genre'],
                'year': img['year'],
                'similarity_score': float(img['similarity']),
                'image_url': f"/art/{img['local_route']}"
            }
            related_images_list.append(img_dict)
        
        msg_dict['related_images'] = related_images_list
        messages_res.append(msg_dict)
    
    image_info = await get_image_with_id(session=session, image_id=chat_details.image_id, user=user)

    return {"messages": messages_res,
            "chat_details": chat_details,
            "image_info": image_info}

@app.post("/chat/send_message")
async def send_message(
    file: Optional[UploadFile] = File(None), 
    image_id : Optional[str] = Form(None), 
    chat_id: Optional[str] = Form(None),
    question: str = Form(...),
    user: dict = Depends(get_current_user_id), 
    session: AsyncSession = Depends(get_session),
    message : Optional[str] = Form(None)
):
    #Imprimo los parametros recibidos para verificar lo que llega al endpoint
    print(f"Parametros recibidos - file: {file}, image_id: {image_id}, chat_id: {chat_id}, question: {question}")
    
    print(f"\nMensaje recibido en el endpoint: {message}\n")

    messageJson = json.loads(message) if message else None
    mesInfo = MessageInfoV2(**messageJson) if messageJson else None
    print(f"Mensaje parseado con Pydantic: {mesInfo}\n")
    print(f"Contenido del mensaje: {mesInfo.content if mesInfo else 'No se pudo parsear el mensaje'}\n")

    query_msg = text("""INSERT INTO messages (chat_id, response, content, created_at) VALUES (:chat_id, :response, :content, :created_at) RETURNING id""")
    created_at = datetime.now()
    result_msg = await session.execute(query_msg, {
        "chat_id": mesInfo.chat_id,
        "response": mesInfo.response,
        "content": mesInfo.content,
        "created_at": created_at
    })
    message_id = result_msg.fetchone()[0]

    # Si sabemos que no habra errores, podemos commitear tranquilamente, pero generalmente no lo haremos
    await session.commit()

    raise HTTPException(status_code=400, detail="Este endpoint está en desarrollo, no se pueden enviar mensajes por ahora.")

    #Aqui me cargo el endpoint si no me llega ni un archivo ni un ID.
    if not file and not image_id and not chat_id:
        raise HTTPException(status_code=400, detail="Debe proporcionar un archivo, una ruta o un id.")
    current_image = None
    current_id = None
    #Empezamos priorizando el chat existente, por si se cuelan otros parametros.
    if chat_id:
        # Buscamos que el chat exista y sea del usuario recibido
        current_id = chat_id
        local_route = await get_image_asociated_to_chat(chat_id=chat_id, user=user, session=session)

        #Ahora busco y cargo la imagen, ya que el LLM la quiere en Bytes... Si no existe, pues 404 y listo
        image_path = os.path.join(config.CARPETA_IMAGENES, local_route)
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                current_image = img_file.read()
        else:
            raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {local_route}")
    #Ahora pasamos al caso en que recibimos el ID de una imagen, comprobamos si existe y si pertenece a un chat actual
    elif image_id:
        print(f"Buscando imagen asociada al id: {image_id} para el usuario {user}")
        recovered_image = await get_image_with_id(session=session, image_id=image_id, user=user)
        if recovered_image:
            print(f"Imagen recuperada con ID {image_id} para el usuario {user}")
            chat_db = await get_chat_by_image_and_user(session=session, image_id=image_id, user=user)
            if chat_db:
                print(f"Chat encontrado para la imagen ID {image_id}: {chat_db}")
                current_id = chat_db['id']
            else:
                #Creamos el chat asociado a esta imagen y a este usuario
                print(f"No se encontró un chat asociado a la imagen ID {image_id} para el usuario {user}")
                chat_db = await create_chat_with_image(session=session, image_id=image_id, user=user)
                current_id = chat_db['id']
            
            #Ahora busco y cargo la imagen, ya que el LLM la quiere en Bytes... Si no existe, pues 404 y listo
            image_path = os.path.join(config.CARPETA_IMAGENES, recovered_image['local_route'])
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    current_image = img_file.read()
            else:
                raise HTTPException(status_code=404, detail=f"Imagen no encontrada en el servidor: {recovered_image['local_route']}")  
    #Acabamos con el caso de un archivo nuevo.
    elif file:
        #Pues aqui debemos subir la imagen, guardarla en la bd, crear el chat, guardar la pregunta, llamar al LLM y guardar la respuesta...
        current_image = await file.read()
        #Guardamos la imagen en el servidor para poder acceder a ella posteriormente y la añadimos a la BD
        savedImageId = await save_image_and_get_id(session=session, contents=current_image, user=user, commit=False)

        #Creamos el chat asociado a esta imagen y a este usuario
        chat_data = await create_chat_with_image(session=session, image_id=savedImageId, user=user)
        current_id = chat_data['id']
    # Insertar pregunta con timestamp explícito (sin commit aún)
    message_data = await add_created_msg(session=session, chat_id=current_id, response=False, content=question, should_commit=False)

    #Ahora si, llego el momento de llamar al LLM con la imagen y la pregunta
    try:
        qwen_description = preguntar_a_vllm(
            current_image,
            question
        )
    except Exception as e:
        print(f"Error al llamar a QWEN: {e}")
        qwen_description = None
        raise HTTPException(status_code=500, detail="Error al procesar la imagen con QWEN")

    #Guardo la respuesta del LLM en la base de datos.
    returned_message = await add_created_msg(session=session, chat_id=current_id, response=True, content=qwen_description or "Error al analizar la imagen con QWEN.", should_commit=False)

    # Finalmente commiteamos todo, para evitar tener mensajes sin respuesta.
    await session.commit()

    # Devuelvo al cliente la respuesta del LLM
    return returned_message

async def add_created_msg(session, chat_id, response, content, should_commit=False, question_id=None):
    # Insertar mensaje con timestamp explícito (sin commit aún)
    query_msg = text("""INSERT INTO messages (chat_id, response, content, created_at, question_id) VALUES (:chat_id, :response, :content, :created_at, :question_id) RETURNING id""")
    created_at = datetime.now()
    result_msg = await session.execute(query_msg, {
        "chat_id": chat_id,
        "response": response,
        "content": content,
        "created_at": created_at,
        "question_id": question_id
    })
    message_id = result_msg.fetchone()[0]

    # Si sabemos que no habra errores, podemos commitear tranquilamente, pero generalmente no lo haremos
    if should_commit:
        await session.commit()

    returned_message = {
        "id": str(message_id),
        "chat_id": str(chat_id),
        "response": response,
        "content": content,
        "created_at": created_at
        }

    return returned_message

async def add_received_msg(session, chat_id, message : MessageInfoV2, should_commit=False, question_id=None):
    # Insertar mensaje con timestamp explícito (sin commit aún)
    print(f"Guardando mensaje en BD - Chat ID: {chat_id}, Response: {message.response}, Content: {message.content}, Question ID: {question_id}")
    query_msg = text("""INSERT INTO messages (id, chat_id, response, content, created_at, question_id) VALUES (:id, :chat_id, :response, :content, :created_at, :question_id) RETURNING id""")
    created_at = datetime.now()
    result_msg = await session.execute(query_msg, {
        "id": message.id,
        "chat_id": chat_id,
        "response": message.response,
        "content": message.content,
        "created_at": created_at,
        "question_id": question_id
    })
    message_id = result_msg.fetchone()[0]

    # Si sabemos que no habra errores, podemos commitear tranquilamente, pero generalmente no lo haremos
    if should_commit:
        await session.commit()

    returned_message = MessageInfoV2(
        id=str(message_id),
        chat_id=str(chat_id),
        response=message.response,
        content=message.content,
        created_at=created_at.isoformat(),
    )

    return returned_message

async def get_chat_by_image_and_user(image_id, user, session) -> Optional[dict]:
    query_chat = text("SELECT * FROM chats WHERE image_id = :image_id and user_id = :user_id")
    result_chat = await session.execute(query_chat, {"image_id": image_id, "user_id": user})
    chat_db = result_chat.mappings().one_or_none()
    if not chat_db:
        return None
    print(f"Chat encontrado para la imagen ID {image_id} y usuario {user}: {chat_db}")
    return chat_db

async def create_chat_with_image(image_id, user, session, external_id=None) -> ChatModel | None:
    chat_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    query_chat = text("""INSERT INTO chats (id, user_id, image_id, created_at, external_id) VALUES (:id, :user_id, :image_id, :created_at, :external_id) ON CONFLICT (user_id, image_id) DO UPDATE SET id=chats.id RETURNING id""")
    result_chat = await session.execute(query_chat, {
    "id": chat_id,
    "user_id": user,
    "image_id": image_id,
    "created_at": created_at,
    "external_id": external_id
    })
    if result_chat.rowcount == 0:
        return None
    else:
        chat_db = ChatModel(
            id=chat_id,
            user_id=user,
            image_id=image_id,
            created_at=created_at.isoformat(),
            topic="",
            external_id=external_id,
            status="SUCCESS"
        )
        print(f"Chat creado para la imagen ID {image_id} y usuario {user}: {chat_db}")
        return chat_db

async def get_image_asociated_to_chat(chat_id, user, session):
    # Primero buscamos el chat para ver si existe y obtener el ID de la imagen asociada
    query_chat = text("SELECT image_id FROM chats WHERE id = :chat_id and user_id = :user_id")
    result_chat = await session.execute(query_chat, {"chat_id": chat_id, "user_id": user})
    chat_db = result_chat.mappings().one_or_none()

    #Si el chat no existe, devolvemos un error 404
    if not chat_db:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    # Si el chat existe, cogemos el id de la imagen asociada y la buscamos en la tabla de imágenes para obtener sus datos
    image_id = chat_db['image_id']
    image_db = await get_image_with_id(image_id=image_id, session=session, user=user)
    
    #Si todo va bien, de momento, devolvemos unicamente la ruta local
    local_route = image_db['local_route']
    return local_route

async def save_image_and_get_id(contents, user, commit, session):
    #Guardamos la imagen en el servidor para poder acceder a ella posteriormente y la añadimos a la BD
    new_id = str(uuid.uuid4())
    with open(os.path.join(config.CARPETA_IMAGENES, f"User/{new_id}.jpg"), 'wb') as img_file:
        img_file.write(contents)
    local_route = f"User/{new_id}.jpg"
    query = text("""INSERT INTO images (id, local_route, owner_id) 
                VALUES (:id, :local_route, :owner_id) RETURNING id""")
    result = await session.execute(query, {
        "id": new_id,
        "local_route": local_route,
        "owner_id": user
    })
    if commit:
        await session.commit()

    return result.scalar_one()

async def save_image_and_get_data(contents, user, commit, session) -> ImageModel:
    #Guardamos la imagen en el servidor para poder acceder a ella posteriormente y la añadimos a la BD
    new_id = str(uuid.uuid4())
    with open(os.path.join(config.CARPETA_IMAGENES, f"User/{new_id}.jpg"), 'wb') as img_file:
        img_file.write(contents)
    local_route = f"User/{new_id}.jpg"
    query = text("""INSERT INTO images (id, local_route, owner_id) 
                VALUES (:id, :local_route, :owner_id) RETURNING id""")
    result = await session.execute(query, {
        "id": new_id,
        "local_route": local_route,
        "owner_id": user
    })
    if commit:
        await session.commit()
    return ImageModel(
        id = new_id,
        name="Unknown",
        artist="Unknown",
        style="Unknown",
        genre="Unknown",
        year="Unknown",
        owner_id = user,
        image_url = local_route
    )
    image_data ={
        "id": new_id,
        "image_url": local_route,
        "owner_id": user,
        "name": "Unknown",
        "artist": "Unknown",
        "style": "Unknown",
        "genre": "Unknown",
        "year": "Unknown"
    }
    return image_data

async def get_image_with_id(image_id, session, user=None) -> ImageModel:
    query_img = text("SELECT * FROM images WHERE id = :id and (owner_id = :user_id OR owner_id IS NULL)")
    result_img = await session.execute(query_img, {"id": image_id, "user_id": user})
    image_db = result_img.mappings().one_or_none()
    print(f"Resultado de la consulta de imagen con ID {image_id}: {image_db}")
    if not image_db:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    image_data = {
        "id": str(image_db['id']),
        "image_url": image_db['local_route'],
        "owner_id": str(image_db['owner_id']),
        "name": image_db['name'],
        "artist": image_db['artist'],
        "style": image_db['style'],
        "genre": image_db['genre'],
        "year": image_db['year']
        
    }

    image_model = ImageModel(
        id = str(image_db['id']),
        image_url = image_db['local_route'],
        owner_id = str(image_db['owner_id']),
        name = image_db['name'],
        artist = image_db['artist'],
        style = image_db['style'],
        genre = image_db['genre'],
        year = image_db['year']
    )
    print("\n--------------------\n")
    print(f"Imagen modelada con ID {image_id}: {image_model}")
    print("\n--------------------\n")

    return image_model

async def get_chat_with_id(chat_id, session, user) -> ChatModel | None:
    query_chat = text("SELECT * FROM chats WHERE id = :id and user_id = :user_id")
    result_chat = await session.execute(query_chat, {"id": chat_id, "user_id": user})
    chat_data = result_chat.mappings().one_or_none()
    print(f"Resultado de la consulta de chat con ID {chat_id}: {chat_data}")
    if not chat_data:
        return None
    else:
        return ChatModel(
            id=str(chat_data['id']),
            user_id=str(chat_data['user_id']),
            image_id=str(chat_data['image_id']),
            created_at=chat_data['created_at'].isoformat(),
            topic="",
            status=chat_data['status'],
            external_id=str(chat_data['external_id'])
        )

async def add_related_images_2_db(message_id, related_images, session, should_commit=False):
    final_related_images = []
    for img in related_images:
        print(f"Guardando imagen relacionada en la BD: {img}")
        query = text("INSERT INTO related_images (message_id, image_id, similarity) VALUES (:message_id, :image_id, :similarity) returning id")
        result = await session.execute(query, {"message_id": message_id, "image_id": img['id'], "similarity": img['similarity_score']})
        img_id = result.scalar_one()
        print(f"Inserted related image with ID: {img_id}")

    if should_commit:
        await session.commit()

async def add_equivalence_chat_id(session, external_id, internal_id, user_id, should_commit=False):
    query = text("INSERT INTO chat_id_equivalences (external_id, internal_id, user_id) VALUES (:external_id, :internal_id, :user_id) returning internal_id")
    result = await session.execute(query, {"external_id": external_id, "internal_id": internal_id, "user_id": user_id})
    new_id = result.scalar_one()   
    if should_commit:
        await session.commit()
    return new_id

async def get_internal_chat_idv0(session, external_id, user_id):
    """
    Retrieves the internal chat ID from the chat_id_equivalences table for
    the given external_id and user_id using an asynchronous database
    session. Returns the internal ID if found, otherwise returns None.
    """
    query = text("SELECT internal_id FROM chat_id_equivalences WHERE external_id = :external_id and user_id = :user_id")
    result = await session.execute(query, {"external_id": external_id, "user_id": user_id})
    internal_id = result.scalar_one_or_none()
    return internal_id

async def get_internal_chat_id(session, external_id, user_id) -> ChatModel | None:
    query = text("SELECT * FROM chats WHERE external_id = :external_id and user_id = :user_id")
    result = await session.execute(query, {"external_id": external_id, "user_id": user_id})
    internal_id = result.mappings().one_or_none()
    print(f"Resultado de la consulta de equivalencia con external_id {external_id} y user_id {user_id}: {internal_id}")
    if not internal_id:
        return None
    else:
        return ChatModel(
            id=str(internal_id['id']),
            user_id=str(internal_id['user_id']),
            image_id=str(internal_id['image_id']),
            created_at=internal_id['created_at'].isoformat(),
            topic=internal_id['topic'],
            external_id=str(internal_id['external_id']),
            status=internal_id['status']
        )