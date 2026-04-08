from pydantic import BaseModel
from clases.ChatModel import ChatModel
from clases.MessageInfo import MessageInfoV2
from clases.ImageModel import ImageModel
from typing import Optional


class CompleteChatResponse(BaseModel):
    chat_info: Optional[ChatModel] = None
    request_info: Optional[MessageInfoV2] = None
    message_info: Optional[MessageInfoV2] = None
    image_info: Optional[ImageModel] = None
