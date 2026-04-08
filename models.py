from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class Image(SQLModel, table=True):
    __tablename__ = "images"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    local_route: str = Field(nullable=False)
    name: str = Field(default="Unknown")
    artist_id: Optional[int] = Field(default=None, foreign_key="artists.id")
    artist: str = Field(default="Unknown")
    style: str = Field(default="Unknown")
    genre: str = Field(default="Unknown")
    year: Optional[str] = Field(default="Unknown")
    owner_id: Optional[str] = Field(default=None)


class Artist(SQLModel, table=True):
    __tablename__ = "artists"

    id: Optional[UUID] = Field(default=None, primary_key=True)
    name: str
    image: Optional[str] = None
