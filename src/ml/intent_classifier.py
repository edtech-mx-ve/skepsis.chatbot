"""Servicio de inferencia para el baseline de clasificación de intenciones."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


class IntentModelError(RuntimeError):
    """Error controlado al cargar o usar el clasificador de intenciones."""


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    """Predicción única con etiqueta y confianza."""

    label: str
    confidence: float


class IntentClassifier:
    """Encapsula la inferencia del pipeline TF-IDF + Logistic Regression."""

    def __init__(self, model: object) -> None:
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise IntentModelError("El artefacto no implementa la interfaz esperada.")
        self._model = model

    @classmethod
    def from_file(cls, path: Path) -> "IntentClassifier":
        """Carga un artefacto local validando ruta, extensión y tamaño."""
        if not isinstance(path, Path):
            raise IntentModelError("La ruta del modelo debe ser un objeto Path.")
        if path.suffix.lower() != ".joblib":
            raise IntentModelError("El modelo debe usar extensión .joblib.")
        if not path.exists() or not path.is_file():
            raise IntentModelError("No se encontró el modelo de intenciones.")
        if path.stat().st_size > 100_000_000:
            raise IntentModelError("El artefacto supera el límite permitido de 100 MB.")

        try:
            model = joblib.load(path)
        except (OSError, ValueError, EOFError) as exc:
            raise IntentModelError("No fue posible cargar el modelo.") from exc
        return cls(model)

    def predict(self, text: str) -> IntentPrediction:
        """Predice una intención y reporta la probabilidad máxima."""
        if not isinstance(text, str) or not text.strip():
            raise IntentModelError("El texto para inferencia debe ser una cadena no vacía.")

        probabilities = self._model.predict_proba([text])[0]
        classes = self._model.classes_
        best_index = int(np.argmax(probabilities))
        return IntentPrediction(
            label=str(classes[best_index]),
            confidence=float(probabilities[best_index]),
        )
