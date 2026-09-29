"""Composición controlada de respuestas enriquecidas por faceta."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

from src.services.facet_detector import FacetMatch
from src.services.service_knowledge import ServiceKnowledge


_FACET_SUMMARIES = MappingProxyType(
    {
        "pln_chatbot": (
            "Podemos diseñar un chatbot de dominio para atención, preguntas "
            "frecuentes, citas, derivación a una persona e integración con "
            "canales como web o WhatsApp, según el alcance."
        ),
        "deep_learning": (
            "Podemos evaluar un prototipo neuronal y compararlo con un "
            "baseline antes de justificar mayor complejidad."
        ),
        "vision_computacional": (
            "Podemos evaluar clasificación, detección o segmentación de "
            "imágenes según el resultado que necesites."
        ),
        "pln_profundo": (
            "Podemos evaluar procesamiento de texto, generación controlada "
            "o representación semántica según el caso."
        ),
        "rnn_lstm_gru": (
            "Podemos evaluar modelos recurrentes para secuencias y comparar "
            "RNN, LSTM o GRU con una línea base."
        ),
        "transfer_learning": (
            "Podemos evaluar la adaptación de un modelo preentrenado frente "
            "a entrenar desde cero."
        ),
        "optimizacion_modelos": (
            "Podemos estructurar una búsqueda reproducible de "
            "hiperparámetros y medir su efecto en validación."
        ),
        "representacion_conocimiento": (
            "Podemos representar reglas, relaciones y conocimiento explícito "
            "para apoyar inferencias trazables."
        ),
        "sistemas_reglas": (
            "Podemos prototipar un motor de reglas con condiciones, "
            "excepciones y trazabilidad de decisiones."
        ),
        "busqueda_heuristica": (
            "Podemos modelar estados y alternativas para explorar soluciones "
            "con métodos de búsqueda y heurísticas."
        ),
        "planificacion_strips": (
            "Podemos representar estado inicial, objetivo, precondiciones y "
            "efectos para generar o validar secuencias de acciones."
        ),
        "cbr": (
            "Podemos evaluar recuperación de casos previos para reutilizar "
            "experiencias y apoyar recomendaciones."
        ),
        "decision_explicable": (
            "Podemos combinar conocimiento explícito, reglas o casos para "
            "hacer más trazable el apoyo a una decisión."
        ),
    }
)

_FACET_QUESTIONS = MappingProxyType(
    {
        "pln_chatbot": (
            "¿Qué tareas debe resolver el chatbot, qué fuentes de información "
            "usará y en qué canal (web, WhatsApp u otro) debe operar?"
        ),
        "deep_learning": (
            "¿Qué tipo de datos tienes y qué métrica definiría que una red "
            "neuronal aporta valor frente a un baseline?"
        ),
        "vision_computacional": (
            "¿Qué necesitas identificar en las imágenes y cuentas con "
            "ejemplos etiquetados?"
        ),
        "pln_profundo": (
            "¿Qué tipo de texto quieres procesar y cuál debe ser la salida "
            "esperada del sistema?"
        ),
        "rnn_lstm_gru": (
            "¿Qué secuencia quieres modelar y cómo medirías el resultado "
            "útil para el proyecto?"
        ),
        "transfer_learning": (
            "¿Qué tarea y dataset tienes, y existe un modelo preentrenado "
            "relacionado con ese dominio?"
        ),
        "optimizacion_modelos": (
            "¿Qué modelo, métrica y conjunto de validación usarías para "
            "comparar configuraciones?"
        ),
        "representacion_conocimiento": (
            "¿Qué hechos, relaciones o reglas del dominio necesitan quedar "
            "representados explícitamente?"
        ),
        "sistemas_reglas": (
            "¿Qué reglas de negocio y excepciones deben respetarse y cómo "
            "se valida hoy una decisión?"
        ),
        "busqueda_heuristica": (
            "¿Cuál es el estado inicial, qué objetivo buscas y qué "
            "restricciones limitan las alternativas?"
        ),
        "planificacion_strips": (
            "¿Cuál es el estado inicial, el objetivo y las acciones con sus "
            "precondiciones y efectos?"
        ),
        "cbr": (
            "¿Existen casos históricos comparables y qué resultado de cada "
            "caso debería reutilizarse?"
        ),
        "decision_explicable": (
            "¿Qué parte de la decisión debe poder justificarse con reglas, "
            "casos o pasos verificables?"
        ),
    }

)

_DOMAIN_SUMMARIES = MappingProxyType(
    {
        ("pln_chatbot", "clinico"): (
            "Podemos evaluar un chatbot de dominio para atención administrativa, "
            "preguntas "
            "frecuentes, citas y derivación a personal, con integración web o "
            "WhatsApp. Si manejará información clínica o de pacientes, el alcance "
            "requiere controles adicionales."
        ),
    }
)

_DOMAIN_QUESTIONS = MappingProxyType(
    {
        ("pln_chatbot", "clinico"): (
            "¿El chatbot se limitará a citas, horarios y preguntas frecuentes, "
            "o también procesará información clínica o datos de pacientes?"
        ),
    }
)


class ServiceResponseComposer:
    """Genera respuestas breves basadas en conocimiento aprobado."""

    def __init__(
        self,
        base_knowledge: dict[str, Any],
        service_knowledge: ServiceKnowledge,
    ) -> None:
        self._base = base_knowledge
        self._services = service_knowledge

    def supports_facet(self, facet: str) -> bool:
        """Indica si existe una respuesta específica aprobada para la faceta."""
        return facet in _FACET_SUMMARIES or facet in _FACET_QUESTIONS

    def _service_title(self, intent: str) -> str:
        service = self._base.get("services", {}).get(intent, {})
        return str(service.get("title", intent.replace("_", " ").title()))

    def _negative_preamble(self, sentiment: str | None) -> str:
        if sentiment == "negativo":
            return (
                "La prioridad es identificar el punto que está generando "
                "la dificultad. "
            )
        return ""

    def _question(
        self,
        match: FacetMatch,
        follow_up: bool,
    ) -> str:
        domain_question = _DOMAIN_QUESTIONS.get(
            (match.facet, match.domain)
        )
        if domain_question:
            return domain_question

        if not follow_up:
            direct = _FACET_QUESTIONS.get(match.facet)
            if direct:
                return direct

        profile = self._services.capability_for_facet(
            match.facet,
            intent=match.suggested_intent,
        )
        if profile is not None and profile.diagnostic_questions:
            index = 1 if follow_up and len(
                profile.diagnostic_questions
            ) > 1 else 0
            return profile.diagnostic_questions[index]

        questions = self._services.diagnostic_questions(
            match.suggested_intent
        )
        if questions:
            return questions[0]
        return "¿Qué resultado esperas obtener?"

    def _maturity_note(self, match: FacetMatch) -> str:
        if (
            match.maturity == "requiere_validacion_especifica"
            and match.domain == "clinico"
        ):
            return (
                "_Alcance clínico: requiere validación específica, protección "
                "de datos sensibles y derivación humana. El chatbot no debe "
                "asumir diagnóstico, prescripción ni decisiones clínicas "
                "autónomas._"
            )
        if match.maturity == "requiere_validacion_especifica":
            return (
                "_Alcance: este caso requiere validación específica y "
                "revisión humana antes de asumir automatización o puesta "
                "en producción._"
            )
        if match.maturity == "capacidad_tecnica_y_prototipado":
            return (
                "_Alcance: capacidad técnica y prototipado; una integración "
                "productiva requiere validar datos, métricas, recursos y riesgos._"
            )
        return ""

    def compose(
        self,
        match: FacetMatch,
        sentiment: str | None = None,
        follow_up: bool = False,
    ) -> str:
        """Compone orientación + pregunta diagnóstica + límite."""
        service_title = self._service_title(
            match.suggested_intent
        )
        question = self._question(match, follow_up)
        note = self._maturity_note(match)

        if follow_up:
            parts = [
                (
                    self._negative_preamble(sentiment)
                    + f"Sigamos con **{service_title} · {match.label}**."
                ),
                f"Para orientarte mejor: {question}",
            ]
            if note:
                parts.append(note)
            return "\n\n".join(parts)

        summary = _DOMAIN_SUMMARIES.get(
            (match.facet, match.domain)
        )
        if not summary:
            summary = _FACET_SUMMARIES.get(match.facet)
        if not summary:
            profile = self._services.capability_for_facet(
                match.facet,
                intent=match.suggested_intent,
            )
            if profile is not None and profile.areas_of_attention:
                summary = (
                    "Podemos evaluar "
                    + profile.areas_of_attention[0]
                    + "."
                )
            else:
                summary = (
                    "Podemos revisar el alcance técnico y definir un "
                    "prototipo verificable."
                )

        parts = [
            (
                self._negative_preamble(sentiment)
                + f"**{service_title} · {match.label}**"
            ),
            summary,
            f"Para orientarte mejor: {question}",
        ]
        if note:
            parts.append(note)
        return "\n\n".join(parts)
