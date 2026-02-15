# Clase para obtener los parametros de la paginacion de /view
from pydantic import BaseModel, Field
from typing import Optional

class PaginacionRequest(BaseModel):
    page: int = Field(default=1, ge=0, description="Número de página, empieza en 1")
    items_per_page: int = Field(default=20, ge=1, le=100, description="Elementos por página")
    filtros: Optional[dict] = Field(default=None, description="Filtros opcionales")
    language: Optional[str] = Field(default="en", description="Idioma de la respuesta")