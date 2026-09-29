"""Detección determinista y explicable de facetas de servicio."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from src.services.service_knowledge import ServiceKnowledge
from src.text_utils import contains_term, normalize_text


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "al", "con", "de", "del", "el", "en", "la", "las", "lo",
        "los", "me", "mi", "necesito", "para", "por", "que", "quiero",
        "una", "un", "y",
    }
)

_FACET_ALIASES = MappingProxyType(
    {
        "deep_learning": (
            "deep learning",
            "aprendizaje profundo",
            "red neuronal",
            "redes neuronales",
            "feedforward",
            "backpropagation",
            "gan",
            "red generativa adversarial",
        ),
        "vision_computacional": (
            "vision computacional",
            "vision artificial",
            "cnn",
            "red convolucional",
            "redes convolucionales",
            "clasificar imagenes",
            "clasificacion de imagenes",
            "detectar objetos",
            "deteccion de objetos",
            "segmentar imagenes",
            "segmentacion de imagenes",
            "resnet",
            "vgg",
            "alexnet",
            "lenet",
        ),
        "pln_profundo": (
            "procesamiento de lenguaje natural",
            "pln profundo",
            "nlp",
            "generacion de texto",
            "analisis de sentimiento",
            "analisis de sentimientos",
            "word2vec",
            "glove",
            "embeddings",
        ),
        "rnn_lstm_gru": (
            "rnn",
            "lstm",
            "gru",
            "red recurrente",
            "redes recurrentes",
        ),
        "transfer_learning": (
            "transfer learning",
            "aprendizaje por transferencia",
            "modelo preentrenado",
            "modelo pre entrenado",
        ),
        "optimizacion_modelos": (
            "hiperparametros",
            "ajuste de hiperparametros",
            "optimizar modelo",
            "optimizar hiperparametros",
        ),
        "representacion_conocimiento": (
            "representacion del conocimiento",
            "logica proposicional",
            "logica de predicados",
            "red semantica",
            "redes semanticas",
        ),
        "sistemas_reglas": (
            "motor de reglas",
            "sistema basado en reglas",
            "sistemas basados en reglas",
            "reglas de negocio",
            "razonamiento deductivo",
            "razonamiento no monotono",
            "reglas con excepciones",
        ),
        "busqueda_heuristica": (
            "busqueda heuristica",
            "heuristica",
            "heuristicas",
            "metodo de busqueda",
            "metodos de busqueda",
        ),
        "planificacion_strips": (
            "strips",
            "planificacion automatica",
            "planificar acciones",
            "secuencia de acciones",
            "planificacion inteligente",
        ),
        "cbr": (
            "cbr",
            "razonamiento basado en casos",
            "casos anteriores",
            "casos previos",
            "experiencias previas",
        ),
        "decision_explicable": (
            "decision explicable",
            "decisiones explicables",
            "explicar decision",
            "explicar la decision",
            "trazabilidad de decision",
            "por que el sistema tomo",
            "por que decidio",
        ),
    }
)

_FACET_LABELS = MappingProxyType(
    {
        "prediccion": "Predicción",
        "clasificacion": "Clasificación",
        "recomendacion": "Recomendación",
        "pln_chatbot": "PLN y chatbots",
        "vision_computacional": "Visión por computadora",
        "anomalias": "Detección de anomalías",
        "prototipo_ml": "Prototipo de machine learning",
        "deep_learning": "Aprendizaje profundo",
        "pln_profundo": "PLN, sentimiento y embeddings",
        "rnn_lstm_gru": "RNN / LSTM / GRU",
        "transfer_learning": "Transfer learning",
        "optimizacion_modelos": "Optimización de modelos",
        "representacion_conocimiento": "Representación del conocimiento",
        "sistemas_reglas": "Sistemas basados en reglas",
        "busqueda_heuristica": "Búsqueda heurística",
        "planificacion_strips": "Planificación automática / STRIPS",
        "cbr": "Razonamiento basado en casos (CBR)",
        "decision_explicable": "Decisión explicable",
        "analisis_exploratorio": "Análisis exploratorio",
        "limpieza": "Limpieza de datos",
        "calidad_datos": "Calidad de datos",
        "indicadores": "Indicadores",
        "segmentacion": "Segmentación",
        "estadistica": "Estadística",
        "visualizacion": "Visualización",
        "preparacion_ml": "Preparación para machine learning",
        "reportes": "Automatización de reportes",
        "archivos": "Procesamiento de archivos",
        "integracion": "Integración",
        "etl": "ETL",
        "validaciones": "Validaciones",
        "tareas_programadas": "Tareas programadas",
        "iot_monitoreo": "IoT y monitoreo",
        "dashboard_bi": "Dashboard / BI",
        "app_interna": "Aplicación interna",
        "api": "API",
        "portal": "Portal",
        "demostrador": "Demostrador",
        "mvp": "MVP",
        "despliegue": "Despliegue",
    }
)

_HIGH_RISK_TERMS: tuple[str, ...] = (
    "biometria productiva",
    "identificacion regulada",
    "reconocimiento facial productivo",
    "control industrial critico",
    "control robotico critico",
    "robot autonomo",
    "decision autonoma de alto impacto",
    "sin supervision humana",
)


@dataclass(frozen=True, slots=True)
class FacetMatch:
    """Resultado explicable de detección de faceta."""

    facet: str
    label: str
    suggested_intent: str
    confidence: float
    source: str
    capability: str | None
    maturity: str


def facet_label(facet: str) -> str:
    """Etiqueta legible para la interfaz."""
    value = _FACET_LABELS.get(facet)
    if value is not None:
        return value
    return facet.replace("_", " ").strip().title()


def _content_tokens(text: str) -> frozenset[str]:
    """Tokens informativos para similitud de ejemplos."""
    return frozenset(
        token
        for token in normalize_text(text).split()
        if token not in _STOPWORDS and len(token) > 1
    )


def _route_similarity(text: str, phrase: str) -> float:
    """Similitud conservadora por solapamiento de tokens."""
    text_tokens = _content_tokens(text)
    phrase_tokens = _content_tokens(phrase)
    if not text_tokens or not phrase_tokens:
        return 0.0

    overlap = len(text_tokens & phrase_tokens)
    if overlap < 2:
        return 0.0

    ratio = overlap / len(phrase_tokens)
    if ratio < 0.60:
        return 0.0
    return min(0.89, 0.70 + 0.20 * ratio)


class ServiceFacetDetector:
    """Detecta facetas después de la intención o como política de dominio."""

    def __init__(self, knowledge: ServiceKnowledge) -> None:
        self._knowledge = knowledge

    def _facet_allowed(
        self,
        facet: str,
        primary_intent: str | None,
    ) -> bool:
        if primary_intent is None:
            return self._knowledge.preferred_intent_for_facet(
                facet
            ) is not None
        return facet in self._knowledge.facets_for_intent(primary_intent)

    def _maturity(
        self,
        text: str,
        facet: str,
        intent: str,
    ) -> tuple[str | None, str]:
        profile = self._knowledge.capability_for_facet(
            facet,
            intent=intent,
        )
        capability = profile.key if profile is not None else None
        maturity = (
            profile.maturity
            if profile is not None
            else "servicio_estable"
        )

        if any(
            contains_term(text, term)
            for term in _HIGH_RISK_TERMS
        ):
            maturity = "requiere_validacion_especifica"

        return capability, maturity

    def detect(
        self,
        text: str,
        primary_intent: str | None = None,
    ) -> FacetMatch | None:
        """Detecta la mejor faceta con reglas y ejemplos aprobados."""
        normalized = normalize_text(text)
        if not normalized:
            return None

        candidates: list[
            tuple[float, int, str, str, str]
        ] = []

        for facet, aliases in _FACET_ALIASES.items():
            if not self._facet_allowed(facet, primary_intent):
                continue
            for alias in aliases:
                if contains_term(normalized, alias):
                    intent = (
                        primary_intent
                        or self._knowledge.preferred_intent_for_facet(
                            facet
                        )
                    )
                    if intent is None:
                        continue
                    # Reglas de alias son deliberadamente de alta precisión.
                    candidates.append(
                        (1.0, len(normalize_text(alias)), facet, intent, "alias_rules")
                    )

        for route in self._knowledge.routes:
            if (
                primary_intent is not None
                and route.intent != primary_intent
            ):
                continue
            if not self._facet_allowed(
                route.facet,
                primary_intent,
            ):
                continue

            if contains_term(normalized, route.phrase):
                score = 0.96
                source = "approved_route"
            else:
                score = _route_similarity(normalized, route.phrase)
                source = "route_similarity"

            if score > 0:
                candidates.append(
                    (
                        score,
                        len(normalize_text(route.phrase)),
                        route.facet,
                        route.intent,
                        source,
                    )
                )

        if not candidates:
            return None

        score, _, facet, intent, source = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        capability, maturity = self._maturity(
            normalized,
            facet,
            intent,
        )

        return FacetMatch(
            facet=facet,
            label=facet_label(facet),
            suggested_intent=intent,
            confidence=float(score),
            source=source,
            capability=capability,
            maturity=maturity,
        )

    def from_context(
        self,
        facet: str,
        intent: str,
        maturity: str | None,
    ) -> FacetMatch:
        """Reconstruye una faceta conservada en contexto."""
        profile = self._knowledge.capability_for_facet(
            facet,
            intent=intent,
        )
        return FacetMatch(
            facet=facet,
            label=facet_label(facet),
            suggested_intent=intent,
            confidence=1.0,
            source="context",
            capability=(
                profile.key
                if profile is not None
                else None
            ),
            maturity=(
                maturity
                or (
                    profile.maturity
                    if profile is not None
                    else "servicio_estable"
                )
            ),
        )
