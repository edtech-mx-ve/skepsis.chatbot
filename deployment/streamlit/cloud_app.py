"""Sképsis Assistant — perfil Cloud Lite para Streamlit Community Cloud.

Este entrypoint conserva la funcionalidad estable del ChatBot:
clasificación de intención, sentimiento Word2Vec, contexto conversacional,
generación controlada y guardrails.

Los laboratorios de PyTorch/Transformers se omiten deliberadamente para reducir
memoria, tamaño de instalación y tiempos de arranque en el nivel gratuito.
"""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    APP_NAME,
    APP_VERSION,
    COMPANY_SITE,
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_MODEL_PATH,
    KNOWLEDGE_BASE_PATH,
    LOGO_LEFT_PATH,
    LOGO_RIGHT_PATH,
    MAX_CLOUD_MESSAGE_LENGTH,
    MAX_SESSION_MESSAGES,
    PAGE_TITLE,
    SENTIMENT_METADATA_PATH,
    SENTIMENT_MODEL_PATH,
    WORD2VEC_PATH,
)
from src.chatbot import ChatbotEngine  # noqa: E402
from src.cloud.sentiment_lite import (  # noqa: E402
    CloudSentimentAnalyzer,
    CloudSentimentError,
)
from src.domain import ConversationState  # noqa: E402
from src.knowledge import (  # noqa: E402
    KnowledgeBaseError,
    load_knowledge_base,
)
from src.logging_config import configure_logging  # noqa: E402
from src.ui.info_sections import get_info_sections  # noqa: E402
from src.ml.intent_classifier import (  # noqa: E402
    IntentClassifier,
    IntentModelError,
)
from src.validation import (  # noqa: E402
    InputValidationError,
    validate_user_message,
)


LOGGER = configure_logging()


@st.cache_resource
def get_engine() -> ChatbotEngine:
    knowledge = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    predictor = None

    if INTENT_MODEL_PATH.exists():
        try:
            predictor = IntentClassifier.from_file(INTENT_MODEL_PATH)
            LOGGER.info("cloud_intent_model_loaded")
        except IntentModelError as exc:
            LOGGER.warning(
                "cloud_intent_model_unavailable error=%s",
                exc,
            )

    return ChatbotEngine(
        knowledge=knowledge,
        intent_predictor=predictor,
        ml_threshold=INTENT_CONFIDENCE_THRESHOLD,
    )


@st.cache_resource
def get_sentiment() -> CloudSentimentAnalyzer | None:
    try:
        analyzer = CloudSentimentAnalyzer.from_files(
            model_path=SENTIMENT_MODEL_PATH,
            metadata_path=SENTIMENT_METADATA_PATH,
            word2vec_path=WORD2VEC_PATH,
        )
        LOGGER.info("cloud_sentiment_loaded")
        return analyzer
    except CloudSentimentError as exc:
        LOGGER.warning(
            "cloud_sentiment_unavailable error=%s",
            exc,
        )
        return None


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = ConversationState()


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.conversation_state = ConversationState()
    LOGGER.info("cloud_conversation_cleared")


def _append_message(
    role: str,
    content: str,
    analysis: str | None = None,
) -> None:
    item = {
        "role": role,
        "content": content,
    }
    if analysis:
        item["analysis"] = analysis

    st.session_state.messages.append(item)
    st.session_state.messages = st.session_state.messages[
        -MAX_SESSION_MESSAGES:
    ]


def render_sidebar() -> bool:
    with st.sidebar:
        if LOGO_LEFT_PATH.exists():
            st.image(
                str(LOGO_LEFT_PATH),
                width="stretch",
            )

        st.subheader("Perfil cloud")
        st.success("Cloud Lite · costo de app $0")
        st.caption(
            "Motor estable, sentimiento y generación controlada. "
            "Los laboratorios neuronales permanecen disponibles en la "
            "versión local."
        )

        st.subheader("Motor de intención")
        if INTENT_MODEL_PATH.exists():
            st.success("TF-IDF + Logistic Regression activo")
        else:
            st.warning("Fallback por reglas activo")

        st.subheader("Sentimiento")
        if (
            SENTIMENT_MODEL_PATH.exists()
            and WORD2VEC_PATH.exists()
        ):
            st.success("Word2Vec activo")
        else:
            st.info("No disponible")

        show_analysis = st.toggle(
            "Mostrar análisis PLN",
            value=False,
        )

        st.subheader("Generación")
        st.success("Controlada · conocimiento aprobado")

        st.subheader("Privacidad")
        st.write(
            "La conversación se conserva únicamente en la sesión actual. "
            "El texto del usuario no se escribe en logs."
        )
        st.button(
            "Nueva conversación",
            on_click=clear_conversation,
            width="stretch",
        )

        st.subheader("Información")
        for section in get_info_sections():
            with st.expander(section.title):
                st.markdown(section.body)

        st.caption(f"Versión {APP_VERSION} · Cloud Lite")

    return show_analysis


def render_header() -> None:
    """Replica en Streamlit la composición aprobada de la interfaz HTML."""
    col_symbol, col_text = st.columns(
        [1.0, 5.4],
        vertical_alignment="center",
    )

    with col_symbol:
        if LOGO_RIGHT_PATH.exists():
            st.image(
                str(LOGO_RIGHT_PATH),
                width="stretch",
            )

    with col_text:
        st.title(APP_NAME)
        st.caption(
            "Asistente especializado para orientación y diagnóstico preliminar "
            "de proyectos tecnológicos de Sképsis Apps."
        )


def render_history() -> None:
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("analysis"):
                st.caption(str(item["analysis"]))


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="💬",
        layout="centered",
    )

    initialize_session()
    render_header()
    show_analysis = render_sidebar()

    try:
        engine = get_engine()
    except KnowledgeBaseError:
        LOGGER.exception("cloud_knowledge_base_load_failed")
        st.error("No fue posible cargar el contexto del asistente.")
        st.stop()

    sentiment_analyzer = get_sentiment()
    render_history()

    prompt = st.chat_input(
        "Describe tu necesidad tecnológica...",
        max_chars=MAX_CLOUD_MESSAGE_LENGTH,
    )

    if prompt is None:
        st.caption(
            "Ejemplos: “Quiero automatizar reportes” · "
            "“Necesito un dashboard” · “¿Cómo trabajan?”"
        )
        return

    try:
        cleaned = validate_user_message(
            prompt,
            MAX_CLOUD_MESSAGE_LENGTH,
        )
    except InputValidationError as exc:
        st.warning(str(exc))
        LOGGER.warning("cloud_invalid_input_rejected")
        return

    _append_message("user", cleaned)

    with st.chat_message("user"):
        st.markdown(cleaned)

    sentiment = None
    if sentiment_analyzer is not None:
        try:
            sentiment = sentiment_analyzer.predict(cleaned)
        except CloudSentimentError as exc:
            LOGGER.warning(
                "cloud_sentiment_inference_failed error=%s",
                exc,
            )

    response = engine.reply(
        cleaned,
        st.session_state.conversation_state,
        sentiment=(
            sentiment.label
            if sentiment is not None
            else None
        ),
    )

    analysis_text = None
    if show_analysis:
        sentiment_text = (
            f"{sentiment.label} ({sentiment.confidence:.2f})"
            if sentiment is not None
            else "no disponible"
        )
        analysis_text = (
            f"Intención: {response.intent} "
            f"({response.confidence:.2f}) · "
            f"Sentimiento: {sentiment_text} · "
            "Generación: controlada"
        )

    _append_message(
        "assistant",
        response.text,
        analysis=analysis_text,
    )

    with st.chat_message("assistant"):
        st.markdown(response.text)
        if analysis_text:
            st.caption(analysis_text)

    LOGGER.info(
        "cloud_turn_processed intent=%s confidence=%.3f "
        "source=%s sentiment=%s turn=%d",
        response.intent,
        response.confidence,
        response.source,
        (
            sentiment.label
            if sentiment is not None
            else "unavailable"
        ),
        st.session_state.conversation_state.turn_count,
    )

    st.divider()
    st.caption(
        f"Powered by Sképsis Apps · © 2026 Sképsis Apps · {COMPANY_SITE}"
    )


if __name__ == "__main__":
    main()
