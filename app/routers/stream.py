import asyncio

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.core.auth import get_current_user_id
from app.services.sse_queues import get_queue

streamRouter = APIRouter()


@streamRouter.get("/stream")
async def stream(user: dict = Depends(get_current_user_id)):
    queue = get_queue(user)

    async def event_generator():
        try:
            while True:
                message = await queue.get()
                yield {"data": message}
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(
        event_generator(),
        ping=15,
        headers={"Cache-Control": "no-cache"},
    )
