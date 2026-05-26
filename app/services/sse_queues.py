import asyncio

message_queues: dict[str, asyncio.Queue] = {}


def get_queue(user_id: str) -> asyncio.Queue:
    key = str(user_id)
    if key not in message_queues:
        message_queues[key] = asyncio.Queue()
    return message_queues[key]
