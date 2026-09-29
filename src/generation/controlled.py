"""Generación controlada de respuestas basada en conocimiento aprobado."""
from __future__ import annotations

from typing import Any

from src.domain import ConversationState


class ControlledGenerationError(RuntimeError):
    """Error controlado al componer una respuesta."""


class ControlledResponseGenerator:
    """Compone respuestas verificables a partir de intención y conocimiento.

    La clase no inventa hechos ni consulta servicios externos. La variación se
    limita a estructura, preguntas diagnósticas y una breve adaptación tonal
    cuando el sentimiento negativo es explícito.
    """

    _SERVICE_INTENTS = {
        "ia_aplicada",
        "ciencia_datos",
        "automatizacion",
        "aplicaciones_empresariales",
    }

    _FOLLOW_UP_QUESTIONS = {
        "ia_aplicada": (
            "¿Qué datos históricos tienes disponibles y cómo medirías que la "
            "predicción o clasificación es útil?"
        ),
        "ciencia_datos": (
            "¿Qué fuentes de datos necesitas combinar y qué indicador o decisión "
            "quieres obtener al final?"
        ),
        "automatizacion": (
            "¿Qué pasos realiza hoy una persona y cuáles podrían ejecutarse de "
            "forma repetible sin intervención manual?"
        ),
        "aplicaciones_empresariales": (
            "¿Qué usuarios utilizarían la aplicación y qué información o acción "
            "deben tener disponible en la pantalla principal?"
        ),
    }

    def __init__(self, knowledge: dict[str, Any]) -> None:
        if not isinstance(knowledge, dict) or not knowledge:
            raise ControlledGenerationError(
                "La base de conocimiento debe ser un diccionario no vacío."
            )
        self._knowledge = knowledge

    def _negative_preamble(self, sentiment: str | None) -> str:
        """Devuelve un preámbulo breve solo ante sentimiento negativo."""
        if sentiment == "negativo":
            return (
                "La prioridad es identificar el punto del proceso que está "
                "generando la dificultad. "
            )
        return ""

    def service_response(
        self,
        key: str,
        state: ConversationState,
        sentiment: str | None = None,
        follow_up: bool = False,
    ) -> str:
        """Genera respuesta de servicio con pregunta diagnóstica verificable."""
        services = self._knowledge.get("services", {})
        if key not in services:
            raise ControlledGenerationError(f"Servicio desconocido: {key}")

        service = services[key]
        state.service_interest = key

        question = str(service["diagnostic_question"])

        if follow_up:
            follow_up_question = self._FOLLOW_UP_QUESTIONS.get(key, question)
            return (
                self._negative_preamble(sentiment)
                + f"Sigamos con **{service['title']}**.\n\n"
                + f"Para orientarte mejor: {follow_up_question}"
            )

        return (
            self._negative_preamble(sentiment)
            + f"**{service['title']}**: {service['description']}\n\n"
            + f"Para orientarte mejor: {question}"
        )

    def methodology_response(self) -> str:
        """Compone la metodología desde las etapas presentes en conocimiento."""
        steps = self._knowledge.get("methodology", [])
        if not steps:
            raise ControlledGenerationError("No hay metodología disponible.")
        flow = " → ".join(
            f"{step['step']}. {step['name']}" for step in steps
        )
        return (
            "La metodología de Sképsis Apps se organiza en cuatro etapas: "
            f"**{flow}**. Se parte del problema, se define la arquitectura, "
            "se construye un activo funcional y se valida el resultado."
        )

    def contact_response(self) -> str:
        """Compone los canales de contacto desde la base de conocimiento."""
        contact = self._knowledge.get("contact", {})
        required = {
            "email",
            "whatsapp_mexico",
            "whatsapp_venezuela",
            "website",
        }
        if not required.issubset(contact):
            raise ControlledGenerationError(
                "La información de contacto está incompleta."
            )

        mexico_digits = "".join(
            char for char in str(contact["whatsapp_mexico"])
            if char.isdigit()
        )
        venezuela_digits = "".join(
            char for char in str(contact["whatsapp_venezuela"])
            if char.isdigit()
        )
        return (
            "Puedes contactar a Sképsis Apps por:\n\n"
            f"- Correo: [**{contact['email']}**](mailto:{contact['email']})\n"
            "- WhatsApp México:  \n"
            f"  [**{contact['whatsapp_mexico']}**](https://wa.me/{mexico_digits})\n"
            "- WhatsApp Venezuela:  \n"
            f"  [**{contact['whatsapp_venezuela']}**](https://wa.me/{venezuela_digits})\n"
            f"- Sitio: [**Sképsis Apps**]({contact['website']})"
        )

    def services_overview(self) -> str:
        """Lista servicios exactamente desde el conocimiento disponible."""
        services = self._knowledge.get("services", {})
        names = [str(value["title"]) for value in services.values()]
        if not names:
            raise ControlledGenerationError("No hay servicios disponibles.")
        return (
            "Puedo orientarte principalmente en **"
            + "**, **".join(names)
            + "**. Describe tu problema y trataré de ubicarlo en una de estas áreas."
        )

    def consulting_response(self) -> str:
        """Genera una invitación diagnóstica sin afirmar una solución aún."""
        return (
            "Podemos comenzar con un **diagnóstico preliminar**. "
            "Describe el problema que quieres resolver, quién realiza actualmente "
            "el proceso y qué resultado esperas obtener."
        )

    def generate(
        self,
        intent: str,
        state: ConversationState,
        sentiment: str | None = None,
        follow_up: bool = False,
    ) -> str:
        """Genera la respuesta final a partir de intención y estado."""
        if intent in self._SERVICE_INTENTS:
            return self.service_response(
                key=intent,
                state=state,
                sentiment=sentiment,
                follow_up=follow_up,
            )
        if intent == "saludo":
            return (
                "Hola. Soy **Sképsis Assistant**. Puedo orientarte sobre IA aplicada, "
                "ciencia de datos, automatización, aplicaciones empresariales y "
                "diagnóstico inicial."
            )
        if intent == "despedida":
            return (
                "Gracias por conversar conmigo. Cuando quieras, podemos continuar "
                "el diagnóstico."
            )
        if intent == "contacto":
            return self.contact_response()
        if intent == "metodologia":
            return self.methodology_response()
        if intent == "tecnologias":
            technologies = self._knowledge.get("technologies", [])
            return "Tecnologías publicadas: " + ", ".join(technologies) + "."
        if intent == "servicios":
            return self.services_overview()
        if intent == "consultoria":
            return self.consulting_response()

        guardrail = self._knowledge.get("guardrail", {}).get("out_of_scope")
        if not guardrail:
            raise ControlledGenerationError(
                "No se definió respuesta fuera de dominio."
            )
        return str(guardrail)
