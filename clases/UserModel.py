from pydantic import BaseModel
from typing import Optional

class UserData(BaseModel):
    internal_id: str      # Tu ID de SQL
    name: str             # Obligatorio
    email: Optional[str] = None      # Opcional (puede ser None)
    avatar: Optional[str] = None  # Opcional (puede ser None)