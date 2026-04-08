# from functools import partial
# import os
# from typing import Any, Awaitable, Callable

# from PIL import Image
# from scipy import io
# from sqlalchemy import text
# from tqdm import asyncio

# from clases import ImageModel
# import config
# from database import AsyncSessionLocal
# from servidorFA import ChatModel, CompleteChatResponse, MessageInfoV2, add_related_images_2_db, analizar_imagen_con_qwen_requests, buscar_imagenes_similares, get_queue


# class estrategia:
#     """Interfaz base para estrategias de procesamiento de mensajes."""

#     async def ejecutar(self, receiveMessage, newMessage, chat, image) -> None:
#         raise NotImplementedError("Debes implementar el metodo ejecutar en una subclase")

# class EstrategiaMensajeSimple(estrategia):
#     """Estrategia concreta para procesamiento de mensajes simples."""

#     async def ejecutar(self, receiveMessage: MessageInfoV2, newMessage: MessageInfoV2, chat : ChatModel, image: ImageModel):
#         async with AsyncSessionLocal() as session:
#             try:
#                 print(f"✅ Procesando mensaje recibido con ID {receiveMessage.id}, y su respuesta {newMessage.id} para el chat ID {chat.id} y la imagen ID {image.id}")
#                 # Cargar la imagen desde disco usando la ruta del ImageModel
#                 full_path = os.path.join(config.CARPETA_IMAGENES, image.image_url)
#                 with open(full_path, 'rb') as img_file:
#                     current_image = img_file.read()
                
#                 # Convertir a PIL Image para la búsqueda
#                 pil_image = Image.open(io.BytesIO(current_image))
                
#                 # Buscar imágenes similares
#                 print("[procesar_mensaje_complejo_recibido] Ejecutando búsqueda de imágenes similares...")
#                 results = await buscar_imagenes_similares(pil_image, n_results=3, session=session, user=chat.user_id)
#                 print(f"[procesar_mensaje_complejo_recibido] Búsqueda completada. Resultados útiles: {len(results)}")
                
#                 # Analizar con QWEN
#                 loop = asyncio.get_event_loop()
#                 qwen_description = await loop.run_in_executor(
#                     None,  # usa el ThreadPoolExecutor por defecto
#                     partial(
#                         analizar_imagen_con_qwen_requests,
#                         current_image,
#                         """Describe esta obra de arte en detalle, incluyendo estilo, técnica, composición y elementos visuales. 
#                         El formato debe ser Descripcion, Estilo, Técnica, Paleta de colores, Composición, 
#                         Comparacion rapida con las similares en caso de haber.""",
#                         results
#                     )
#                 )
#                 status = "SUCCESS"
                
#             except Exception as e:
#                 print(f"❌ Error al procesar imagen: {e}")
#                 qwen_description = "Error al analizar la imagen con QWEN."
#                 results = []
#                 status = "ERROR"
            
#             # Actualizar el mensaje en la base de datos
#             try:
#                 query_update = text("""
#                     UPDATE messages 
#                     SET content = :content, status = :status
#                     WHERE id = :message_id
#                 """)
#                 await session.execute(query_update, {
#                     "content": qwen_description,
#                     "status": status,
#                     "message_id": newMessage.id
#                 })
                
#                 # Guardar imágenes relacionadas si hay
#                 if results:
#                     await add_related_images_2_db(
#                         message_id=newMessage.id,
#                         related_images=results,
#                         session=session,
#                         should_commit=False
#                     )
                
#                 await session.commit()
#                 print(f"✅ Procesamiento completado para mensaje {newMessage.id}")
#                 queue = get_queue(chat.user_id)
#                 result = MessageInfoV2(
#                     id=str(newMessage.id),
#                     chat_id=str(newMessage.chat_id),
#                     response=newMessage.response,
#                     content=qwen_description,
#                     created_at=newMessage.created_at,
#                     status=status,
#                     related_images=results,
#                     question_id=newMessage.question_id
#                 )
#                 fin = CompleteChatResponse(
#                     message_info=result,
#                     request_info=receiveMessage,
#                     chat_info=chat,
#                     image_info=image
#                 )
#                 print(f"Enviando resultado al cliente a través de SSE: {fin}")
#                 await queue.put(fin.model_dump_json())
                
#             except Exception as e:
#                 print(f"❌ Error al actualizar la base de datos: {e}")
#                 await session.rollback()


# class EstrategiaMensajeComplejo(estrategia):
#     """Estrategia concreta para procesamiento de mensajes complejos."""

#     def __init__(self, procesador_complejo: Callable[[Any, Any, Any, Any], Awaitable[None]]):
#         self._procesador_complejo = procesador_complejo

#     async def ejecutar(self, receiveMessage, newMessage, chat, image) -> None:
#         await self._procesador_complejo(receiveMessage, newMessage, chat, image)


# class Contexto:
#     """Mantiene una estrategia activa y delega su ejecucion."""

#     def __init__(self, estrategia_activa: estrategia):
#         self._estrategia = estrategia_activa

#     def establecer_estrategia(self, estrategia_activa: estrategia) -> None:
#         self._estrategia = estrategia_activa

#     async def ejecutar_estrategia(self, receiveMessage, newMessage, chat, image) -> None:
#         await self._estrategia.ejecutar(receiveMessage, newMessage, chat, image)
