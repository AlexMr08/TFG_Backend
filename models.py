from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Image(SQLModel, table=True):
    __tablename__ = "images"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    local_route: str = Field(nullable=False)
    name: str = Field(default="Unknown")
    artist: str = Field(default="Unknown")
    style: str = Field(default="Unknown")
    genre: str = Field(default="Unknown")
