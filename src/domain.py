"""Entidades del dominio conversacional."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class ConversationState:
    """Estado mínimo de una conversación, sin persistir texto del usuario."""

    last_intent: Optional[str] = None
    service_interest: Optional[str] = None
    service_facet: Optional[str] = None
    service_maturity: Optional[str] = None
    turn_count: int = 0
    history: list[str] = field(default_factory=list)

    def register_turn(self, intent: str) -> None:
        """Registra una intención sin persistir el texto del usuario."""
        self.last_intent = intent
        self.turn_count += 1
        self.history.append(intent)
        self.history = self.history[-10:]
