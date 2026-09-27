"""Transformaciones puras y reutilizables de texto."""
import re
import unicodedata

def normalize_text(text: str) -> str:
    """Normaliza texto para coincidencia determinista."""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    ).lower()
    without_accents = re.sub(r"[^a-z0-9ñ\s]", " ", without_accents)
    return re.sub(r"\s+", " ", without_accents).strip()

def contains_term(text: str, term: str) -> bool:
    """Busca palabra o frase completa evitando coincidencias parciales."""
    normalized_text = normalize_text(text)
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_term)}(?!\w)"
    return re.search(pattern, normalized_text) is not None

def contains_any(text: str, terms: list[str]) -> bool:
    """Indica si al menos un término completo aparece en el texto."""
    return any(contains_term(text, term) for term in terms)

def count_matches(text: str, terms: list[str]) -> int:
    """Cuenta términos distintos presentes en el texto."""
    return sum(1 for term in terms if contains_term(text, term))
