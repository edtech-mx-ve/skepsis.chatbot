"""Contenido didáctico breve para la barra lateral."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InfoSection:
    """Sección informativa inmutable de la interfaz."""

    title: str
    body: str


INFO_SECTIONS: tuple[InfoSection, ...] = (
    InfoSection(
        title="Ayuda",
        body=(
            "**Cómo usar la app**\n\n"
            "1. Escribe tu necesidad tecnológica.\n"
            "2. La app detecta la intención y el tono.\n"
            "3. Responde con conocimiento aprobado de Sképsis Apps.\n"
            "4. Activa **Mostrar análisis PLN** para ver intención, faceta, madurez y sentimiento.\n\n"
            "**Ejemplos:** automatización, IA, datos, APIs, metodología y contacto."
        ),
    ),
    InfoSection(
        title="Modelo",
        body=(
            "**Arquitectura híbrida**\n\n"
            "- Producción: **TF-IDF + Logistic Regression** para intención.\n"
            "- Deep Learning: **RNN, LSTM y GRU** implementadas en PyTorch CPU.\n"
            "- Intención recurrente: **LSTM** seleccionada como mejor experimento.\n"
            "- Sentimiento: **Word2Vec + reglas**; GloVe queda como comparación.\n"
            "- Servicios: **facetas deterministas** enriquecen el diagnóstico sin reentrenar las 12 intenciones.\n"
            "- Generación oficial: **controlada**; GRU y FLAN-T5 quedan como laboratorios locales.\n\n"
            "Cloud Lite usa solo los componentes estables y ligeros."
        ),
    ),
    InfoSection(
        title="Evaluación",
        body=(
            "**Resultados principales**\n\n"
            "- Baseline intención: accuracy **0.975** y macro-F1 **0.973** en test.\n"
            "- LSTM recurrente: macro-F1 **0.890** en test; se mantiene experimental.\n"
            "- Sentimiento Word2Vec: macro-F1 **1.000** en test controlado; híbrido **30/30** en evaluación externa.\n"
            "- Generación GRU experimental: perplejidad **2.208** y accuracy token **0.780** en test.\n"
            "- Cloud Lite: **11/11** controles funcionales y robustez sin fallos.\n\n"
            "Las métricas provienen de datasets pequeños y especializados del proyecto."
        ),
    ),
    InfoSection(
        title="Acerca de",
        body=(
            "**Sképsis Apps** ofrece IA aplicada, ciencia de datos, automatización, "
            "aplicaciones empresariales, dashboards, APIs y diagnóstico tecnológico.\n\n"
            "**Contacto**\n"
            "- Correo: [**skepsis.apps@gmail.com**](mailto:skepsis.apps@gmail.com)\n"
            "- WhatsApp México:  \n"
            "  [**+52 55 6574 1576**](https://wa.me/525565741576)\n"
            "- WhatsApp Venezuela:  \n"
            "  [**+58 424 403 55 99**](https://wa.me/584244035599)\n"
            "- Sitio: [**Sképsis Apps**](https://skepsis-apps.github.io/landing_page/)"
        ),
    ),
    InfoSection(
        title="Trivia",
        body=(
            "**Sképsis** viene del griego **σκέψις (sképsis)**, asociado con "
            "examinar, considerar y reflexionar.\n\n"
            "La idea resume la filosofía del producto: **entender el problema antes de proponer la solución**."
        ),
    ),
    InfoSection(
        title="Glosario",
        body=(
            "**Consulta rápida**\n\n"
            "- **IA aplicada:** uso de IA para clasificar, predecir, recomendar o automatizar.\n"
            "- **PLN:** procesamiento de lenguaje natural para analizar texto humano.\n"
            "- **Intención:** objetivo principal detectado en el mensaje.\n"
            "- **Sentimiento:** tono estimado: positivo, neutral o negativo.\n"
            "- **TF-IDF:** representación que pondera palabras relevantes en un texto.\n"
            "- **Logistic Regression:** clasificador estable usado para intención.\n"
            "- **Embedding:** representación vectorial de palabras o textos.\n"
            "- **Word2Vec / GloVe:** métodos de embeddings usados en el proyecto.\n"
            "- **RNN:** red neuronal recurrente para secuencias.\n"
            "- **LSTM / GRU:** variantes recurrentes con mecanismos de memoria.\n"
            "- **Generación controlada:** respuesta limitada a conocimiento aprobado.\n"
            "- **Contexto:** memoria mínima para relacionar turnos consecutivos.\n"
            "- **Grounding:** verificación de que la salida esté respaldada por contexto autorizado.\n"
            "- **Fallback:** respuesta segura alternativa cuando una salida no es confiable.\n"
            "- **FLAN-T5:** modelo local usado como laboratorio experimental.\n"
            "- **Cloud Lite:** versión ligera para despliegue web.\n"
            "- **Accuracy:** proporción total de predicciones correctas.\n"
            "- **Macro-F1:** promedio equilibrado del F1 entre clases.\n"
            "- **Perplejidad:** métrica de generación; menor suele indicar mejor ajuste al texto."
        ),
    ),
)


def get_info_sections() -> tuple[InfoSection, ...]:
    """Devuelve las secciones informativas en el orden de presentación."""
    return INFO_SECTIONS
