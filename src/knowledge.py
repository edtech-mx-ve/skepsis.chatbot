"""Carga segura de la base de conocimiento local."""
import json
from pathlib import Path
from typing import Any

class KnowledgeBaseError(RuntimeError):
    """Error controlado de conocimiento local."""

def load_knowledge_base(path: Path) -> dict[str, Any]:
    """Carga JSON validando extensión, existencia, tamaño y esquema mínimo."""
    if not isinstance(path, Path):
        raise KnowledgeBaseError("La ruta debe ser un objeto Path.")
    if path.suffix.lower() != ".json":
        raise KnowledgeBaseError("La base de conocimiento debe ser JSON.")
    if not path.exists() or not path.is_file():
        raise KnowledgeBaseError("No se encontró la base de conocimiento.")
    if path.stat().st_size > 1_000_000:
        raise KnowledgeBaseError("La base de conocimiento supera el tamaño permitido.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError("No fue posible leer la base de conocimiento.") from exc

    required = {"company", "services", "methodology", "solutions", "contact", "guardrail"}
    missing = required.difference(data)
    if missing:
        raise KnowledgeBaseError(f"Faltan secciones: {sorted(missing)}")
    if not isinstance(data["services"], dict) or not data["services"]:
        raise KnowledgeBaseError("La sección services debe contener información.")
    return data
