"""Logging de eventos sin texto del usuario."""
import logging

def configure_logging() -> logging.Logger:
    """Configura logger de consola idempotente."""
    logger = logging.getLogger("skepsis_chatbot")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
