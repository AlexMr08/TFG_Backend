from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator, ConfigDict

from app.clases.Enums import Estados

class ChatModel(BaseModel):
    id: str
    user_id: str
    image_id: Optional[str] = None
    collection_id: Optional[str] = None
    created_at: datetime
    external_id: Optional[str] = None
    status: Optional[Estados] = None
    
    model_config = ConfigDict(
    json_encoders={datetime: lambda v: v.isoformat()}  # conserva el offset
    )

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
    