"""Validación defensiva de entradas."""
import re

class InputValidationError(ValueError):
    """Error controlado para entradas no aceptadas."""

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def validate_user_message(message: object, max_length: int = 500) -> str:
    """Valida tipo, contenido y longitud del mensaje."""
    if not isinstance(message, str):
        raise InputValidationError("El mensaje debe ser texto.")
    cleaned = _CONTROL_CHARS.sub("", message).strip()
    if not cleaned:
        raise InputValidationError("Escribe una consulta antes de enviarla.")
    if len(cleaned) > max_length:
        raise InputValidationError(
            f"El mensaje supera el máximo permitido de {max_length} caracteres."
        )
    return cleaned
