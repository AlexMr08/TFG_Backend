from enum import Enum


class Estados(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    WAITING = "WAITING"
    ERROR = "ERROR"

class TipoMensaje(str, Enum):
    SIMPLE = "SIMPLE"
    COMPLETE = "COMPLETE"
    COLORS = "COLORS"