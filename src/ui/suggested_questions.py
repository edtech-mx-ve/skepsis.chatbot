"""Preguntas sugeridas para orientar el uso de Sképsis Assistant."""
from __future__ import annotations


SUGGESTED_QUESTIONS: tuple[str, ...] = (
    "¿Qué servicios ofrece Sképsis Apps?",
    "No sé qué tecnología necesito para mi proyecto. ¿Cómo pueden orientarme?",
    "¿Cómo trabajan los proyectos desde el diagnóstico hasta la validación?",
    "Quiero automatizar reportes de Excel. ¿Qué información necesitan?",
    "Necesito un dashboard para indicadores de ventas. ¿Cómo definimos el alcance?",
    "Necesito integrar dos sistemas mediante una API. ¿Qué datos y procesos deben conectarse?",
    "Quiero una aplicación interna para gestionar operaciones. ¿Por dónde empezamos?",
    "Tengo datos dispersos y quiero convertirlos en información útil. ¿Qué debería preparar?",
    "Quiero analizar las ventas de mi empresa. ¿Qué datos necesito?",
    "Quiero predecir la demanda del próximo trimestre. ¿Qué histórico conviene tener?",
    "Necesito detectar anomalías en datos operativos. ¿Cómo se puede evaluar?",
    "Necesito clasificar imágenes con una CNN. ¿Qué dataset necesito?",
    "Quiero detectar objetos en imágenes. ¿Cómo definimos las clases y métricas?",
    "Necesito segmentar regiones en imágenes. ¿Qué etiquetado se requiere?",
    "Quiero adaptar un modelo con transfer learning. ¿Qué modelo base y dataset convienen?",
    "Necesito optimizar hiperparámetros. ¿Cómo evitamos sobreajuste?",
    "Quiero analizar secuencias con LSTM o GRU. ¿Cómo elegimos entre ambas?",
    "Quiero analizar sentimientos en textos. ¿Qué datos y métricas necesito?",
    "Quiero usar embeddings para representar texto. ¿Qué opciones conviene comparar?",
    "Quiero un chatbot para atender preguntas frecuentes. ¿Qué información debe conocer?",
    "Quiero que el chatbot funcione desde WhatsApp. ¿Qué alcance debemos definir?",
    "Quiero un chatbot para un consultorio clínico. ¿Qué límites y validaciones necesita?",
    "Quiero representar conocimiento con lógica de predicados. ¿Cómo estructuramos hechos y reglas?",
    "Necesito un motor de reglas con excepciones. ¿Cómo documentamos las decisiones?",
    "Quiero usar búsqueda heurística. ¿Cómo definimos estados, objetivo y costo?",
    "Necesito planificar acciones con STRIPS. ¿Cómo representamos precondiciones y efectos?",
    "Quiero reutilizar casos anteriores con CBR. ¿Qué información debe conservar cada caso?",
    "Necesito explicar por qué el sistema tomó una decisión. ¿Qué trazabilidad se puede ofrecer?",
    "Quiero automatizar un proceso crítico. ¿Qué controles y supervisión humana se requieren?",
    "¿Cómo puedo contactar a Sképsis Apps y cuál es su sitio oficial?",
)


def questions_markdown() -> str:
    """Devuelve las 30 preguntas como lista numerada en Markdown."""
    return "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(SUGGESTED_QUESTIONS, start=1)
    )
