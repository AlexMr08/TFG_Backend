from pydantic import BaseModel
from typing import Optional
from app.clases.UserModel import UserData
from pydantic import BaseModel
from app.clases.ChatModel import ChatModel
from app.clases.MessageInfo import MessageInfoV2
from app.clases.ImageModel import ImageModel
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
