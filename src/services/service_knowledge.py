"""Carga y valida la expansión de conocimiento de servicios."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PRIMARY_INTENTS: frozenset[str] = frozenset(
    {
        "aplicaciones_empresariales",
        "automatizacion",
        "ciencia_datos",
        "consultoria",
        "contacto",
        "despedida",
        "fuera_dominio",
        "ia_aplicada",
        "metodologia",
        "saludo",
        "servicios",
        "tecnologias",
    }
)


class ServiceKnowledgeError(RuntimeError):
    """Error de carga o esquema del conocimiento de servicios."""


@dataclass(frozen=True, slots=True)
class RouteExample:
    """Ejemplo aprobado para detectar una faceta."""

    phrase: str
    intent: str
    facet: str


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Perfil operativo de una capacidad avanzada."""

    key: str
    label: str
    primary_intents: tuple[str, ...]
    maturity: str
    service_facets: tuple[str, ...]
    areas_of_attention: tuple[str, ...]
    diagnostic_questions: tuple[str, ...]
    communication_rule: str
    limits: tuple[str, ...]


class ServiceKnowledge:
    """Representa conocimiento de servicios validado y de solo lectura lógica."""

    def __init__(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict) or not data:
            raise ServiceKnowledgeError(
                "El conocimiento de servicios debe ser un objeto JSON no vacío."
            )
        self._data = deepcopy(data)
        self._validate()

    @classmethod
    def from_file(cls, path: Path) -> "ServiceKnowledge":
        """Carga el conocimiento desde JSON UTF-8."""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ServiceKnowledgeError(
                f"No existe el conocimiento de servicios: {path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ServiceKnowledgeError(
                f"No fue posible leer el conocimiento de servicios: {path}"
            ) from exc

        if not isinstance(value, dict):
            raise ServiceKnowledgeError(
                "La raíz del conocimiento de servicios debe ser un objeto."
            )
        return cls(value)

    def _validate(self) -> None:
        metadata = self._data.get("metadata")
        if not isinstance(metadata, dict):
            raise ServiceKnowledgeError("Falta metadata.")

        version = metadata.get("version")
        if not isinstance(version, str) or not version.startswith("2."):
            raise ServiceKnowledgeError(
                "Sprint 8 requiere conocimiento de servicios v2.x."
            )

        strategy = self._data.get("strategy")
        if not isinstance(strategy, dict) or not strategy.get(
            "keep_primary_intents"
        ):
            raise ServiceKnowledgeError(
                "El esquema debe conservar las intenciones principales."
            )

        facets = self._data.get("facets")
        if not isinstance(facets, dict) or not facets:
            raise ServiceKnowledgeError("Falta el catálogo de facetas.")

        for intent, values in facets.items():
            if intent not in PRIMARY_INTENTS:
                raise ServiceKnowledgeError(
                    f"Intención no permitida en facetas: {intent}"
                )
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item.strip()
                for item in values
            ):
                raise ServiceKnowledgeError(
                    f"Facetas inválidas para {intent}."
                )
            if len(values) != len(set(values)):
                raise ServiceKnowledgeError(
                    f"Hay facetas duplicadas para {intent}."
                )

        for route in self._data.get("routing_examples", []):
            if not isinstance(route, dict):
                raise ServiceKnowledgeError(
                    "routing_examples contiene un elemento inválido."
                )
            phrase = route.get("phrase")
            intent = route.get("intent")
            facet = route.get("facet")
            if (
                not isinstance(phrase, str)
                or not phrase.strip()
                or intent not in PRIMARY_INTENTS
                or not isinstance(facet, str)
            ):
                raise ServiceKnowledgeError(
                    "Ejemplo de enrutamiento incompleto."
                )
            if facet not in facets.get(intent, []):
                raise ServiceKnowledgeError(
                    f"La faceta {facet} no está registrada en {intent}."
                )

        extensions = self._data.get("capability_extensions", {})
        if not isinstance(extensions, dict):
            raise ServiceKnowledgeError(
                "capability_extensions debe ser un objeto."
            )

        for key, raw in extensions.items():
            if not isinstance(raw, dict):
                raise ServiceKnowledgeError(
                    f"Capacidad inválida: {key}"
                )
            intents = raw.get("primary_intents", [])
            service_facets = raw.get("service_facets", [])
            if not intents or not service_facets:
                raise ServiceKnowledgeError(
                    f"La capacidad {key} no tiene intención o facetas."
                )
            if any(intent not in PRIMARY_INTENTS for intent in intents):
                raise ServiceKnowledgeError(
                    f"La capacidad {key} usa una intención no permitida."
                )
            if not all(
                any(facet in facets.get(intent, []) for intent in intents)
                for facet in service_facets
            ):
                raise ServiceKnowledgeError(
                    f"La capacidad {key} contiene facetas no registradas."
                )

    @property
    def version(self) -> str:
        """Versión del conocimiento."""
        return str(self._data["metadata"]["version"])

    @property
    def keep_primary_intents(self) -> bool:
        """Confirma que no se crean clases de intención nuevas."""
        return bool(self._data["strategy"]["keep_primary_intents"])

    @property
    def routes(self) -> tuple[RouteExample, ...]:
        """Devuelve ejemplos de enrutamiento validados."""
        return tuple(
            RouteExample(
                phrase=str(item["phrase"]),
                intent=str(item["intent"]),
                facet=str(item["facet"]),
            )
            for item in self._data.get("routing_examples", [])
        )

    def facets_for_intent(self, intent: str) -> tuple[str, ...]:
        """Facetas permitidas para una intención."""
        values = self._data.get("facets", {}).get(intent, [])
        return tuple(str(value) for value in values)

    def preferred_intent_for_facet(self, facet: str) -> str | None:
        """Intención preferida según ejemplos y perfiles aprobados."""
        for route in self.routes:
            if route.facet == facet:
                return route.intent

        for profile in self.capability_profiles:
            if facet in profile.service_facets and profile.primary_intents:
                return profile.primary_intents[0]
        return None

    @property
    def capability_profiles(self) -> tuple[CapabilityProfile, ...]:
        """Perfiles de capacidades académicas aprobadas."""
        profiles: list[CapabilityProfile] = []
        for key, raw in self._data.get(
            "capability_extensions", {}
        ).items():
            profiles.append(
                CapabilityProfile(
                    key=str(key),
                    label=str(raw["label"]),
                    primary_intents=tuple(
                        str(value)
                        for value in raw.get("primary_intents", [])
                    ),
                    maturity=str(raw["maturity"]),
                    service_facets=tuple(
                        str(value)
                        for value in raw.get("service_facets", [])
                    ),
                    areas_of_attention=tuple(
                        str(value)
                        for value in raw.get("areas_of_attention", [])
                    ),
                    diagnostic_questions=tuple(
                        str(value)
                        for value in raw.get(
                            "diagnostic_questions", []
                        )
                    ),
                    communication_rule=str(
                        raw.get("communication_rule", "")
                    ),
                    limits=tuple(
                        str(value)
                        for value in raw.get("limits", [])
                    ),
                )
            )
        return tuple(profiles)

    def capability_for_facet(
        self,
        facet: str,
        intent: str | None = None,
    ) -> CapabilityProfile | None:
        """Busca el perfil que respalda una faceta."""
        for profile in self.capability_profiles:
            if facet not in profile.service_facets:
                continue
            if (
                intent is not None
                and intent not in profile.primary_intents
            ):
                continue
            return profile
        return None

    def maturity_description(self, maturity: str) -> str:
        """Descripción aprobada de un nivel de madurez."""
        value = self._data.get("maturity_model", {}).get(maturity, "")
        return str(value)

    def diagnostic_questions(
        self,
        category: str,
    ) -> tuple[str, ...]:
        """Preguntas diagnósticas de una categoría."""
        values = self._data.get(
            "diagnostic_questions", {}
        ).get(category, [])
        return tuple(str(value) for value in values)


def load_service_knowledge(path: Path) -> ServiceKnowledge:
    """Función pública para cargar conocimiento de servicios."""
    return ServiceKnowledge.from_file(path)
