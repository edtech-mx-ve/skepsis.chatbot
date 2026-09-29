"""Motor conversacional híbrido: intención + facetas + generación controlada."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.domain import ConversationState
from src.generation.controlled import ControlledResponseGenerator
from src.services.facet_detector import FacetMatch, ServiceFacetDetector
from src.services.service_knowledge import ServiceKnowledge
from src.services.service_response import ServiceResponseComposer
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
    facet: str | None = None
    facet_label: str | None = None
    facet_confidence: float | None = None
    facet_source: str | None = None
    maturity: str | None = None


class ChatbotEngine:
    """Orquesta conocimiento, reglas, ML, facetas y estado de conversación."""

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
    _METHODOLOGY = [
        "metodologia", "como trabajan", "como desarrollan",
        "proceso de trabajo", "que etapas", "que fases", "proceso siguen",
        "desde la idea", "hasta la entrega", "hasta la validacion",
        "organizan el trabajo"
    ]
    _TECH = [
        "tecnologias", "tecnologia", "stack", "herramientas",
        "lenguajes", "frameworks", "librerias"
    ]
    _SERVICES = [
        "servicios", "que hacen", "que ofrecen", "capacidades",
        "tipos de proyectos", "tipo de proyectos", "clase de soluciones",
        "puede desarrollar", "pueden construir"
    ]
    _CONSULTING = [
        "no se que", "no se si", "necesito orientacion",
        "quiero orientacion", "diagnostico", "por donde empezar",
        "que tecnologia necesito", "solucion adecuada",
        "definir la solucion", "recomendacion"
    ]
    _FOLLOW_UP = [
        "cuentame mas", "mas informacion", "explicame mas",
        "y eso", "como funciona"
    ]
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
    _FACET_POLICY_MIN_CONFIDENCE = 0.95

    def __init__(
        self,
        knowledge: dict[str, Any],
        intent_predictor: IntentPredictor | None = None,
        ml_threshold: float = 0.35,
        service_knowledge: ServiceKnowledge | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._intent_predictor = intent_predictor
        self._ml_threshold = ml_threshold
        self._response_generator = ControlledResponseGenerator(knowledge)
        self._service_knowledge = service_knowledge
        self._facet_detector = (
            ServiceFacetDetector(service_knowledge)
            if service_knowledge is not None
            else None
        )
        self._service_composer = (
            ServiceResponseComposer(knowledge, service_knowledge)
            if service_knowledge is not None
            else None
        )

    def _service_response(
        self,
        key: str,
        state: ConversationState,
        sentiment: str | None = None,
        follow_up: bool = False,
    ) -> str:
        """Delega la composición general de servicios al generador controlado."""
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

    def _detect_facet(
        self,
        message: str,
        primary_intent: str | None = None,
    ) -> FacetMatch | None:
        if self._facet_detector is None:
            return None
        return self._facet_detector.detect(
            message,
            primary_intent=primary_intent,
        )

    def _context_facet(
        self,
        state: ConversationState,
    ) -> FacetMatch | None:
        if (
            self._facet_detector is None
            or not state.service_interest
            or not state.service_facet
        ):
            return None
        return self._facet_detector.from_context(
            facet=state.service_facet,
            intent=state.service_interest,
            maturity=state.service_maturity,
        )

    def _build_response(
        self,
        *,
        intent: str,
        confidence: float,
        source: str,
        state: ConversationState,
        sentiment: str | None,
        facet_match: FacetMatch | None = None,
        follow_up: bool = False,
    ) -> ChatResponse:
        """Compone la respuesta manteniendo intención y faceta separadas."""
        if intent in self._SERVICE_INTENTS:
            if facet_match is not None:
                state.service_interest = intent
                state.service_facet = facet_match.facet
                state.service_maturity = facet_match.maturity

                if (
                    self._service_composer is not None
                    and (
                        facet_match.capability is not None
                        or self._service_composer.supports_facet(
                            facet_match.facet
                        )
                    )
                ):
                    text = self._service_composer.compose(
                        facet_match,
                        sentiment=sentiment,
                        follow_up=follow_up,
                    )
                else:
                    text = self._service_response(
                        intent,
                        state=state,
                        sentiment=sentiment,
                        follow_up=follow_up,
                    )

                return ChatResponse(
                    text=text,
                    intent=intent,
                    confidence=confidence,
                    source=source,
                    facet=facet_match.facet,
                    facet_label=facet_match.label,
                    facet_confidence=facet_match.confidence,
                    facet_source=facet_match.source,
                    maturity=facet_match.maturity,
                )

            if not follow_up:
                state.service_facet = None
                state.service_maturity = None

            text = self._service_response(
                intent,
                state=state,
                sentiment=sentiment,
                follow_up=follow_up,
            )
            return ChatResponse(
                text=text,
                intent=intent,
                confidence=confidence,
                source=source,
            )

        text = self._response_for_intent(
            intent,
            state,
            sentiment=sentiment,
            follow_up=follow_up,
        )
        return ChatResponse(
            text=text,
            intent=intent,
            confidence=confidence,
            source=source,
        )

    def _route_service_rules(self, message: str) -> str | None:
        scores = {
            key: count_matches(
                message,
                list(service.get("keywords", [])),
            )
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
        elif self._is_contact_request(normalized):
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

        facet_match = (
            self._detect_facet(normalized, primary_intent=intent)
            if intent in self._SERVICE_INTENTS
            else None
        )
        return self._build_response(
            intent=intent,
            confidence=confidence,
            source="rules",
            state=state,
            sentiment=sentiment,
            facet_match=facet_match,
        )

    def _is_contact_request(self, normalized: str) -> bool:
        """Distingue pedir contacto de mencionar un canal de integración."""
        direct_contact_terms = [
            "contacto", "contactar", "contactarlos", "comunicarme", "correo",
            "email", "hablar con alguien", "sitio web", "pagina web",
            "página web", "website", "sitio oficial", "web de contacto",
            "direccion web", "dirección web",
        ]
        if contains_any(normalized, direct_contact_terms):
            return True

        if "whatsapp" in normalized:
            return contains_any(
                normalized,
                [
                    "cual es", "cuál es", "numero", "número", "contacto",
                    "contactar", "contactarlos", "comunicarme", "escribirles",
                    "hablar con", "donde los contacto", "dónde los contacto",
                ],
            )

        return (
            " web " in f" {normalized} "
            and contains_any(
                normalized,
                [
                    "cual es", "cuál es", "donde", "dónde", "direccion",
                    "dirección", "pagina", "página", "sitio",
                ],
            )
        )

    def _is_predictive_request(self, normalized: str) -> bool:
        """Detecta señales inequívocas de intención predictiva."""
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
        """Genera respuesta con contexto, intención, faceta y fallback seguro."""
        normalized = normalize_text(message)
        has_follow_up = contains_any(normalized, self._FOLLOW_UP)

        # Una petición explícita en el mismo mensaje tiene prioridad sobre
        # el contexto previo. Esto evita que "motor de reglas. Cuéntame más"
        # herede accidentalmente la faceta de un turno anterior.
        explicit_facet = self._detect_facet(normalized)

        # Contacto explícito conserva prioridad incluso si el mensaje incluye
        # una expresión de seguimiento.
        if self._is_contact_request(normalized):
            intent = "contacto"
            response = self._build_response(
                intent=intent,
                confidence=1.0,
                source="policy",
                state=state,
                sentiment=sentiment,
            )
            state.register_turn(intent)
            return response

        # Pronóstico inequívoco conserva la política estable.
        if self._is_predictive_request(normalized):
            intent = "ia_aplicada"
            response = self._build_response(
                intent=intent,
                confidence=0.90,
                source="policy",
                state=state,
                sentiment=sentiment,
                facet_match=self._detect_facet(
                    normalized,
                    primary_intent=intent,
                ),
            )
            state.register_turn(intent)
            return response

        # Sprint 8: términos técnicos inequívocos se mapean a una intención
        # EXISTENTE; no se crean clases nuevas ni se reentrena el baseline.
        capability_match = explicit_facet
        if (
            capability_match is not None
            and (
                capability_match.capability is not None
                or (
                    self._service_composer is not None
                    and self._service_composer.supports_facet(
                        capability_match.facet
                    )
                )
            )
            and capability_match.confidence
            >= self._FACET_POLICY_MIN_CONFIDENCE
        ):
            intent = capability_match.suggested_intent
            response = self._build_response(
                intent=intent,
                confidence=capability_match.confidence,
                source="facet_policy",
                state=state,
                sentiment=sentiment,
                facet_match=capability_match,
                follow_up=has_follow_up,
            )
            state.register_turn(intent)
            return response

        # Una referencia explícita a otro servicio también prevalece sobre el
        # contexto previo cuando aparece junto con "cuéntame más".
        if has_follow_up:
            explicit_service = self._route_service_rules(normalized)
            if explicit_service:
                facet_match = self._detect_facet(
                    normalized,
                    primary_intent=explicit_service,
                )
                response = self._build_response(
                    intent=explicit_service,
                    confidence=0.90,
                    source="rules",
                    state=state,
                    sentiment=sentiment,
                    facet_match=facet_match,
                    follow_up=True,
                )
                state.register_turn(explicit_service)
                return response

        # Solo un seguimiento sin tema nuevo reutiliza el contexto vigente.
        if has_follow_up and state.service_interest:
            intent = state.service_interest
            facet_match = self._context_facet(state)
            response = self._build_response(
                intent=intent,
                confidence=0.85,
                source="context",
                state=state,
                sentiment=sentiment,
                facet_match=facet_match,
                follow_up=True,
            )
            state.register_turn(intent)
            return response

        if self._intent_predictor is not None:
            prediction = self._intent_predictor.predict(message)
            label = str(getattr(prediction, "label"))
            confidence = float(getattr(prediction, "confidence"))
            if confidence >= self._ml_threshold:
                facet_match = (
                    self._detect_facet(
                        normalized,
                        primary_intent=label,
                    )
                    if label in self._SERVICE_INTENTS
                    else None
                )
                response = self._build_response(
                    intent=label,
                    confidence=confidence,
                    source="ml",
                    state=state,
                    sentiment=sentiment,
                    facet_match=facet_match,
                )
                state.register_turn(label)
                return response

        response = self._reply_with_rules(
            normalized,
            state,
            sentiment=sentiment,
        )
        state.register_turn(response.intent)
        return response
