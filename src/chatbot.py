"""Motor conversacional híbrido: reglas + baseline ML del Sprint 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.domain import ConversationState
from src.generation.controlled import ControlledResponseGenerator
from src.text_utils import contains_any, count_matches, normalize_text


class IntentPredictor(Protocol):
    """Interfaz mínima para cualquier clasificador de intenciones."""

    def predict(self, text: str) -> object:
        """Devuelve un objeto con atributos label y confidence."""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Respuesta estructurada del motor conversacional."""

    text: str
    intent: str
    confidence: float
    source: str = "rules"


class ChatbotEngine:
    """Orquesta conocimiento, reglas, ML y estado de conversación."""

    _GREETINGS = [
        "hola", "buen dia", "buenos dias", "buenas tardes", "buenas noches",
        "saludos", "buen dia equipo", "buen dia skepsis", "hola equipo"
    ]
    _FAREWELLS = ["gracias", "adios", "hasta luego", "eso es todo"]
    _CONTACT = [
        "contacto", "contactar", "contactarlos", "comunicarme", "correo",
        "email", "whatsapp", "hablar con alguien", "sitio web",
        "pagina web", "página web", "website", "sitio oficial",
        "web de contacto", "direccion web", "dirección web"
    ]
    _METHODOLOGY = ["metodologia", "como trabajan", "como desarrollan", "proceso de trabajo", "que etapas", "que fases", "proceso siguen", "desde la idea", "hasta la entrega", "hasta la validacion", "organizan el trabajo"]
    _TECH = ["tecnologias", "tecnologia", "stack", "herramientas", "lenguajes", "frameworks", "librerias"]
    _SERVICES = ["servicios", "que hacen", "que ofrecen", "capacidades", "tipos de proyectos", "tipo de proyectos", "clase de soluciones", "puede desarrollar", "pueden construir"]
    _CONSULTING = [
        "no se que", "no se si", "necesito orientacion", "quiero orientacion",
        "diagnostico", "por donde empezar", "que tecnologia necesito",
        "solucion adecuada", "definir la solucion", "recomendacion"
    ]
    _FOLLOW_UP = ["cuentame mas", "mas informacion", "explicame mas", "y eso", "como funciona"]
    _PREDICTIVE_STRONG = [
        "anticipar", "proyectar", "pronosticar", "pronostico",
        "prevision", "prever"
    ]
    _PREDICTIVE_WEAK = ["estimar", "estimacion", "calcular"]
    _FUTURE_HINTS = [
        "futuro", "proximo", "siguiente", "mes que viene",
        "trimestre", "semestre", "periodo futuro"
    ]
    _SERVICE_INTENTS = {
        "ia_aplicada",
        "ciencia_datos",
        "automatizacion",
        "aplicaciones_empresariales",
    }

    def __init__(
        self,
        knowledge: dict[str, Any],
        intent_predictor: IntentPredictor | None = None,
        ml_threshold: float = 0.35,
    ) -> None:
        self._knowledge = knowledge
        self._intent_predictor = intent_predictor
        self._ml_threshold = ml_threshold
        self._response_generator = ControlledResponseGenerator(knowledge)

    def _service_response(
        self,
        key: str,
        state: ConversationState,
        sentiment: str | None = None,
        follow_up: bool = False,
    ) -> str:
        """Delega la composición de servicios al generador controlado."""
        return self._response_generator.service_response(
            key=key,
            state=state,
            sentiment=sentiment,
            follow_up=follow_up,
        )

    def _response_for_intent(
        self,
        intent: str,
        state: ConversationState,
        sentiment: str | None = None,
        follow_up: bool = False,
    ) -> str:
        """Genera texto verificable a partir de intención y contexto."""
        return self._response_generator.generate(
            intent=intent,
            state=state,
            sentiment=sentiment,
            follow_up=follow_up,
        )

    def _route_service_rules(self, message: str) -> str | None:
        scores = {
            key: count_matches(message, list(service.get("keywords", [])))
            for key, service in self._knowledge["services"].items()
        }
        best_key = max(scores, key=scores.get)
        return best_key if scores[best_key] > 0 else None

    def _reply_with_rules(
        self,
        normalized: str,
        state: ConversationState,
        sentiment: str | None = None,
    ) -> ChatResponse:
        if contains_any(normalized, self._GREETINGS):
            intent, confidence = "saludo", 1.0
        elif contains_any(normalized, self._FAREWELLS):
            intent, confidence = "despedida", 1.0
        elif contains_any(normalized, self._CONTACT):
            intent, confidence = "contacto", 1.0
        elif contains_any(normalized, self._METHODOLOGY):
            intent, confidence = "metodologia", 1.0
        elif contains_any(normalized, self._CONSULTING):
            intent, confidence = "consultoria", 0.90
        elif contains_any(normalized, self._TECH):
            intent, confidence = "tecnologias", 1.0
        elif contains_any(normalized, self._SERVICES):
            intent, confidence = "servicios", 1.0
        else:
            service_key = self._route_service_rules(normalized)
            if service_key:
                intent, confidence = service_key, 0.90
            else:
                intent, confidence = "fuera_dominio", 0.0

        text = self._response_for_intent(intent, state, sentiment=sentiment)
        return ChatResponse(
            text=text,
            intent=intent,
            confidence=confidence,
            source="rules",
        )

    def _is_contact_request(self, normalized: str) -> bool:
        """Detecta solicitudes explícitas de canales o sitio de contacto."""
        if contains_any(normalized, self._CONTACT):
            return True

        return (
            " web " in f" {normalized} "
            and contains_any(
                normalized,
                ["cual es", "donde", "direccion", "dirección", "pagina", "página", "sitio"],
            )
        )

    def _is_predictive_request(self, normalized: str) -> bool:
        """Detecta señales inequívocas de intención predictiva.

        Se usa como política conservadora para evitar que consultas de pronóstico
        sean absorbidas por ciencia de datos descriptiva cuando la confianza ML
        es marginal.
        """
        if contains_any(normalized, self._PREDICTIVE_STRONG):
            return True
        return (
            contains_any(normalized, self._PREDICTIVE_WEAK)
            and contains_any(normalized, self._FUTURE_HINTS)
        )

    def reply(
        self,
        message: str,
        state: ConversationState,
        sentiment: str | None = None,
    ) -> ChatResponse:
        """Genera respuesta usando contexto, ML si está disponible y fallback de reglas."""
        normalized = normalize_text(message)

        # Una pregunta de seguimiento debe respetar el contexto vigente.
        if contains_any(normalized, self._FOLLOW_UP) and state.service_interest:
            intent = state.service_interest
            response = ChatResponse(
                text=self._service_response(
                    intent,
                    state=state,
                    sentiment=sentiment,
                    follow_up=True,
                ),
                intent=intent,
                confidence=0.85,
                source="context",
            )
            state.register_turn(intent)
            return response

        # Hotfix 6.1: las solicitudes explícitas de contacto deben tener
        # prioridad sobre una predicción ML errónea de fuera de dominio.
        if self._is_contact_request(normalized):
            intent = "contacto"
            response = ChatResponse(
                text=self._response_for_intent(intent, state, sentiment=sentiment),
                intent=intent,
                confidence=1.0,
                source="policy",
            )
            state.register_turn(intent)
            return response

        # Política de dominio: las señales claras de pronóstico tienen prioridad
        # sobre una predicción ML marginal hacia análisis descriptivo.
        if self._is_predictive_request(normalized):
            intent = "ia_aplicada"
            response = ChatResponse(
                text=self._response_for_intent(intent, state, sentiment=sentiment),
                intent=intent,
                confidence=0.90,
                source="policy",
            )
            state.register_turn(intent)
            return response

        if self._intent_predictor is not None:
            prediction = self._intent_predictor.predict(message)
            label = str(getattr(prediction, "label"))
            confidence = float(getattr(prediction, "confidence"))
            if confidence >= self._ml_threshold:
                text = self._response_for_intent(label, state, sentiment=sentiment)
                response = ChatResponse(
                    text=text,
                    intent=label,
                    confidence=confidence,
                    source="ml",
                )
                state.register_turn(label)
                return response

        response = self._reply_with_rules(normalized, state, sentiment=sentiment)
        state.register_turn(response.intent)
        return response
