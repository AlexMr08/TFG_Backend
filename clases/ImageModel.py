from pydantic import BaseModel
from typing import Optional

class ImageModel(BaseModel):
    id: str
    name: Optional[str] = "Unknown"
    artist_id: Optional[str] = ""
    artist: Optional[str] = "Unknown"
    style_id: Optional[str] = ""
    style: Optional[str] = "Unknown"
    genre_id: Optional[str] = ""
    genre: Optional[str] = "Unknown"
    year: Optional[str] = "Unknown"
    owner_id: Optional[str] = None
    similarity_score: Optional[float] = None
    image_url: str
    
class ArtistModel(BaseModel):
    id: str
    name: Optional[str] = "Unknown"
    image_url: Optional[str] = None