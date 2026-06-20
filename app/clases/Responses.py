from pydantic import BaseModel
from typing import Optional
from app.clases.user_model import UserData
from pydantic import BaseModel
from app.clases.chat_model import ChatModel
from app.clases.message_info import MessageInfoV2
from app.clases.image_model import ImageModel
from typing import Optional


class LoginResponse(BaseModel):
    id_token: Optional[str] = None
    status: str
    user: Optional[UserData] = None
    
class CompleteChatResponse(BaseModel):
    chat_info: Optional[ChatModel] = None
    request_info: Optional[MessageInfoV2] = None
    message_info: Optional[MessageInfoV2] = None
    image_info: Optional[ImageModel] = None

class ChatListResponse(BaseModel):
    chats: list[ChatModel]
