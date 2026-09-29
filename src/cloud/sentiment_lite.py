"""Inferencia de sentimiento Word2Vec sin PyTorch para el perfil cloud."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import unicodedata

import joblib
import numpy as np
from gensim.models import KeyedVectors


class CloudSentimentError(RuntimeError):
    """Error controlado del analizador de sentimiento ligero."""


@dataclass(frozen=True, slots=True)
class CloudSentimentPrediction:
    label: str
    confidence: float


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", text)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = value.lower()
    value = re.sub(r"[^a-z0-9ñ\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return normalized.split() if normalized else []


class CloudSentimentAnalyzer:
    """Analizador seleccionado del Sprint 4 sin dependencias neuronales."""

    def __init__(self, model: object, vectors: KeyedVectors) -> None:
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            raise CloudSentimentError(
                "El modelo de sentimiento no implementa la interfaz esperada."
            )
        self._model = model
        self._vectors = vectors

    @classmethod
    def from_files(
        cls,
        model_path: Path,
        metadata_path: Path,
        word2vec_path: Path,
    ) -> "CloudSentimentAnalyzer":
        for path in (model_path, metadata_path, word2vec_path):
            if not path.exists() or not path.is_file():
                raise CloudSentimentError(
                    f"Falta el artefacto requerido: {path.name}"
                )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("embedding_kind") != "word2vec":
            raise CloudSentimentError(
                "El perfil cloud espera Word2Vec como embedding seleccionado."
            )

        try:
            model = joblib.load(model_path)
            vectors = KeyedVectors.load(str(word2vec_path), mmap="r")
        except (OSError, ValueError, EOFError) as exc:
            raise CloudSentimentError(
                "No fue posible cargar los artefactos de sentimiento."
            ) from exc

        return cls(model=model, vectors=vectors)

    def _vectorize(self, text: str) -> np.ndarray:
        vectors = [
            self._vectors[token]
            for token in _tokens(text)
            if token in self._vectors.key_to_index
        ]
        if not vectors:
            return np.zeros(
                self._vectors.vector_size,
                dtype=np.float32,
            )
        return np.mean(
            np.asarray(vectors, dtype=np.float32),
            axis=0,
        )

    @staticmethod
    def _rule_prediction(
        text: str,
    ) -> CloudSentimentPrediction | None:
        normalized = _normalize_text(text)

        negative_phrases = (
            "no funciona",
            "esta fallando",
            "esta causando problemas",
            "muchos errores",
            "demasiados errores",
            "resultados inconsistentes",
            "perdiendo tiempo",
            "estoy frustrado",
            "estoy molesto",
            "estamos molestos",
            "nos preocupa",
            "estamos preocupados",
            "afectando el trabajo",
            "es lento",
            "demasiado trabajo manual",
            "mas problemas",
        )
        positive_phrases = (
            "funciona muy bien",
            "funcionando muy bien",
            "estoy satisfecho",
            "estoy muy satisfecho",
            "estamos muy satisfechos",
            "estamos satisfechos",
            "estamos conformes",
            "muy util",
            "resultado favorable",
            "resultado positivo",
            "nos ahorra tiempo",
            "ahorro tiempo",
            "simplifico el trabajo",
            "respondio mejor",
        )

        if any(phrase in normalized for phrase in negative_phrases):
            return CloudSentimentPrediction(
                label="negativo",
                confidence=0.95,
            )
        if any(phrase in normalized for phrase in positive_phrases):
            return CloudSentimentPrediction(
                label="positivo",
                confidence=0.95,
            )

        neutral_followup_markers = {
            "cuentame mas",
            "dime mas",
            "continua",
            "continuemos",
            "explicame mas",
            "amplia la informacion",
            "puedes ampliar",
        }
        if normalized in neutral_followup_markers:
            return CloudSentimentPrediction(
                label="neutral",
                confidence=0.90,
            )

        neutral_predictive_markers = (
            "quiero predecir",
            "necesito predecir",
            "quiero pronosticar",
            "necesito pronosticar",
            "quiero proyectar",
            "necesito proyectar",
            "predecir la demanda",
            "pronosticar la demanda",
            "proyectar las ventas",
        )
        if any(
            marker in normalized
            for marker in neutral_predictive_markers
        ):
            return CloudSentimentPrediction(
                label="neutral",
                confidence=0.90,
            )

        neutral_technical_request_markers = (
            "quiero entrenar",
            "necesito entrenar",
            "quiero clasificar",
            "necesito clasificar",
            "quiero analizar",
            "necesito analizar",
            "quiero adaptar",
            "necesito adaptar",
            "quiero optimizar",
            "necesito optimizar",
            "quiero representar",
            "necesito representar",
            "quiero usar",
            "necesito usar",
            "quiero planificar",
            "necesito planificar",
            "quiero reutilizar",
            "necesito reutilizar",
            "quiero explicar",
            "necesito explicar",
            "motor de reglas",
            "sistema basado en reglas",
        )
        if any(
            marker in normalized
            for marker in neutral_technical_request_markers
        ):
            return CloudSentimentPrediction(
                label="neutral",
                confidence=0.90,
            )

        neutral_question_markers = (
            "cual es",
            "cuál es",
            "que es",
            "qué es",
            "como ",
            "cómo ",
            "donde ",
            "dónde ",
            "cuando ",
            "cuándo ",
            "quien ",
            "quién ",
            "quiero saber",
            "quisiera saber",
            "necesito informacion",
            "necesito información",
            "que servicios",
            "qué servicios",
            "que tecnologias",
            "qué tecnologías",
        )
        if (
            "?" in text
            or any(
                marker in normalized
                for marker in neutral_question_markers
            )
        ):
            return CloudSentimentPrediction(
                label="neutral",
                confidence=0.90,
            )

        return None

    def predict(self, text: str) -> CloudSentimentPrediction:
        if not isinstance(text, str) or not text.strip():
            raise CloudSentimentError(
                "El texto no puede estar vacío."
            )

        rule = self._rule_prediction(text)
        if rule is not None:
            return rule

        vector = self._vectorize(text)
        probabilities = self._model.predict_proba([vector])[0]
        classes = self._model.classes_
        best = int(np.argmax(probabilities))

        return CloudSentimentPrediction(
            label=str(classes[best]),
            confidence=float(probabilities[best]),
        )
