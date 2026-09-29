"""Formateo puro del análisis visible de PLN y servicios."""
from __future__ import annotations

from src.chatbot import ChatResponse


_MATURITY_LABELS = {
    "servicio_estable": "servicio estable",
    "capacidad_tecnica_y_prototipado": "capacidad técnica / prototipado",
    "requiere_validacion_especifica": "requiere validación específica",
}


def format_analysis_text(
    response: ChatResponse,
    sentiment_label: str | None,
    sentiment_confidence: float | None,
) -> str:
    """Devuelve una línea compacta de trazabilidad para la interfaz."""
    parts = [
        f"Intención: {response.intent} ({response.confidence:.2f})",
    ]

    if response.facet:
        facet_confidence = (
            f" ({response.facet_confidence:.2f})"
            if response.facet_confidence is not None
            else ""
        )
        parts.append(
            f"Faceta: {response.facet_label or response.facet}"
            f"{facet_confidence}"
        )

    if response.maturity:
        parts.append(
            "Madurez: "
            + _MATURITY_LABELS.get(
                response.maturity,
                response.maturity.replace("_", " "),
            )
        )

    sentiment = (
        f"{sentiment_label} ({sentiment_confidence:.2f})"
        if (
            sentiment_label is not None
            and sentiment_confidence is not None
        )
        else "no disponible"
    )
    parts.append(f"Sentimiento: {sentiment}")

    source = response.source
    if (
        response.facet_source
        and response.facet_source != response.source
    ):
        source += f" + {response.facet_source}"
    parts.append(f"Fuente: {source}")
    parts.append("Generación: controlada")
    return " · ".join(parts)
