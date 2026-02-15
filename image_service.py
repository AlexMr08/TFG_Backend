from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Image
from typing import List, Optional


async def get_image_by_id(session: AsyncSession, image_id: int) -> Optional[Image]:
    """Obtiene una imagen por su ID"""
    result = await session.execute(
        select(Image).where(Image.id == image_id)
    )
    return result.scalar_one_or_none()


async def get_images_by_ids(session: AsyncSession, image_ids: List[int]) -> List[Image]:
    """Obtiene múltiples imágenes por sus IDs"""
    result = await session.execute(
        select(Image).where(Image.id.in_(image_ids))
    )
    return result.scalars().all()


async def get_all_images_paginated(session: AsyncSession, offset: int = 0, limit: int = 20) -> List[Image]:
    """Obtiene imágenes paginadas"""
    result = await session.execute(
        select(Image).offset(offset).limit(limit)
    )
    return result.scalars().all()
