from pydantic import BaseModel, ConfigDict
from datetime import datetime
from app.clases.Enums import Estados, TipoMensaje
from typing import Optional

class MessageInfoV2(BaseModel):
    id: str
    chat_id: str
    response: bool
    content: str
    created_at: datetime
    status: Optional[Estados] = Estados.SUCCESS
    related_images : Optional[list[dict]] = None
    question_id: Optional[str] = None
    type: TipoMensaje = TipoMensaje.COMPLETE
    model_config = ConfigDict(
    json_encoders={datetime: lambda v: v.isoformat()}  # ✅ conserva el offset
    )
