"""Sképsis Assistant — perfil Cloud Lite para Streamlit Community Cloud.

Este entrypoint conserva la funcionalidad estable del ChatBot:
clasificación de intención, facetas de servicio, sentimiento Word2Vec,
contexto conversacional, generación controlada y guardrails.

Los laboratorios de PyTorch/Transformers se omiten deliberadamente para reducir
memoria, tamaño de instalación y tiempos de arranque en el nivel gratuito.
"""
from __future__ import annotations

from pathlib import Path
import base64
import html
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
    SERVICE_KNOWLEDGE_PATH,
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
from src.services.service_knowledge import (  # noqa: E402
    ServiceKnowledgeError,
    load_service_knowledge,
)
from src.logging_config import configure_logging  # noqa: E402
from src.ui.analysis import format_analysis_text  # noqa: E402
from src.ui.info_sections import (  # noqa: E402
    get_info_sections,
    next_open_info_section,
)
from src.ui.suggested_questions import (  # noqa: E402
    SUGGESTED_QUESTIONS,
)
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
    service_knowledge = None
    if SERVICE_KNOWLEDGE_PATH.exists():
        try:
            service_knowledge = load_service_knowledge(
                SERVICE_KNOWLEDGE_PATH
            )
            LOGGER.info(
                "service_knowledge_loaded version=%s",
                service_knowledge.version,
            )
        except ServiceKnowledgeError as exc:
            LOGGER.warning(
                "service_knowledge_unavailable error=%s",
                exc,
            )
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
        service_knowledge=service_knowledge,
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
    if "open_info_section" not in st.session_state:
        st.session_state.open_info_section = None
    if "pending_suggested_question" not in st.session_state:
        st.session_state.pending_suggested_question = None


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.conversation_state = ConversationState()
    st.session_state.pending_suggested_question = None
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



def _image_data_uri(path: Path) -> str:
    """Convierte un recurso local aprobado a data URI para el encabezado."""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_app_styles() -> None:
    """Aplica estilos responsivos y accesibles de la interfaz."""
    st.markdown(
        """
        <style>
        .skepsis-main-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 0.2rem 0 1.1rem 0;
        }

        .skepsis-main-logo {
            width: 92px;
            height: auto;
            flex: 0 0 auto;
            border-radius: 10px;
        }

        .skepsis-main-title {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.1;
            font-weight: 700;
        }

        .skepsis-main-subtitle {
            margin-top: 0.6rem;
            opacity: 0.72;
            line-height: 1.5;
        }

        .st-key-info_accordion button {
            justify-content: flex-start;
            text-align: left;
        }

        .st-key-info_accordion [data-testid="stMarkdownContainer"] {
            text-align: left;
        }

        .st-key-chat_controls {
            position: sticky;
            bottom: 0;
            z-index: 20;
            padding: 0.35rem 0 0.45rem 0;
            background: var(--background-color, #0e1117);
        }

        .st-key-chat_controls [data-testid="stExpander"] {
            margin-top: 0.25rem;
        }

        .st-key-chat_controls [data-testid="stExpanderDetails"] {
            max-height: 45vh;
            overflow-y: auto;
        }

        .st-key-chat_controls [data-testid="stExpanderDetails"] button {
            justify-content: flex-start;
            text-align: left;
            white-space: normal;
            height: auto;
        }

        @media (max-width: 1024px) {
            .skepsis-main-logo {
                width: 69px;
            }

            .skepsis-main-header {
                gap: 0.7rem;
                align-items: flex-start;
            }

            .skepsis-main-title {
                font-size: clamp(1.8rem, 7vw, 2.5rem);
            }

            .skepsis-main-subtitle {
                margin-top: 0.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _toggle_info_section(title: str) -> None:
    """Abre/cierra una sección y garantiza exclusividad."""
    st.session_state.open_info_section = next_open_info_section(
        st.session_state.get("open_info_section"),
        title,
    )


def render_info_accordion() -> None:
    """Renderiza Información como acordeón exclusivo."""
    with st.container(key="info_accordion"):
        for index, section in enumerate(get_info_sections()):
            is_open = (
                st.session_state.get("open_info_section")
                == section.title
            )
            symbol = "⌄" if is_open else "›"
            st.button(
                f"{symbol}  {section.title}",
                key=f"info_section_{index}",
                on_click=_toggle_info_section,
                args=(section.title,),
                width="stretch",
            )
            if is_open:
                st.markdown(section.body)


def _select_suggested_question(question: str) -> None:
    """Envía una pregunta sugerida al flujo normal del chatbot."""
    st.session_state.pending_suggested_question = question


def render_chat_controls() -> str | None:
    """Muestra el chat y 30 preguntas que se envían al seleccionarlas."""
    pending = st.session_state.get(
        "pending_suggested_question"
    )
    st.session_state.pending_suggested_question = None

    with st.container(key="chat_controls"):
        typed_prompt = st.chat_input(
            "Describe tu necesidad tecnológica...",
            max_chars=MAX_CLOUD_MESSAGE_LENGTH,
        )
        with st.expander(
            "30 preguntas sugeridas para mejor uso del chatbot",
            expanded=False,
        ):
            st.caption(
                "Selecciona una pregunta para enviarla directamente al chatbot."
            )
            for index, question in enumerate(
                SUGGESTED_QUESTIONS,
                start=1,
            ):
                st.button(
                    f"{index}. {question}",
                    key=f"suggested_question_{index}",
                    on_click=_select_suggested_question,
                    args=(question,),
                    width="stretch",
                )

    if pending:
        return str(pending)
    return typed_prompt

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
        render_info_accordion()

        st.caption(f"Versión {APP_VERSION} · Cloud Lite")

    return show_analysis


def render_header() -> None:
    """Renderiza encabezado responsivo con logo de 69 px en móvil/tablet."""
    logo_html = ""
    if LOGO_RIGHT_PATH.exists():
        logo_html = (
            '<img class="skepsis-main-logo" '
            f'src="{_image_data_uri(LOGO_RIGHT_PATH)}" '
            'alt="Símbolo de Sképsis Apps">'
        )

    subtitle = (
        "Asistente especializado para orientación y diagnóstico preliminar "
        "de proyectos tecnológicos de Sképsis Apps."
    )
    st.markdown(
        (
            '<div class="skepsis-main-header">'
            f"{logo_html}"
            '<div class="skepsis-main-copy">'
            f'<h1 class="skepsis-main-title">{html.escape(APP_NAME)}</h1>'
            f'<div class="skepsis-main-subtitle">{html.escape(subtitle)}</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
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
    render_app_styles()
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

    st.divider()
    st.caption(
        f"Powered by Sképsis Apps · © 2026 Sképsis Apps · {COMPANY_SITE}"
    )

    prompt = render_chat_controls()

    if prompt is None:
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
        analysis_text = format_analysis_text(
            response=response,
            sentiment_label=(
                sentiment.label
                if sentiment is not None
                else None
            ),
            sentiment_confidence=(
                sentiment.confidence
                if sentiment is not None
                else None
            ),
        )

    _append_message(
        "assistant",
        response.text,
        analysis=analysis_text,
    )

    LOGGER.info(
        "cloud_turn_processed intent=%s confidence=%.3f "
        "source=%s facet=%s maturity=%s sentiment=%s turn=%d",
        response.intent,
        response.confidence,
        response.source,
        response.facet or "none",
        response.maturity or "none",
        (
            sentiment.label
            if sentiment is not None
            else "unavailable"
        ),
        st.session_state.conversation_state.turn_count,
    )

    st.rerun()


if __name__ == "__main__":
    main()
