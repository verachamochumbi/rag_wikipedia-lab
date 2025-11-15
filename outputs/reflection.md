# 🧩 Multi-Agent Workflow vs. RAG Summarizer
**1. Manejo de ambigüedad y contradicciones (Multi-Agent Workflow)**

El enfoque multi-agente (Investigador → Redactor → Revisor) muestra un comportamiento más interpretativo.
- Cuando las fuentes son escasas o contradictorias, el Investigador intenta compensar ampliando la búsqueda.
- El Redactor sintetiza incluso si el material es limitado, generando un texto coherente aunque deba llenar vacíos.
- El Revisor detecta incoherencias o falta de estructura, aportando correcciones de estilo u organización.

Este flujo permite manejar ambigüedad de forma flexible, pero no garantiza factualidad estricta, porque los agentes dependen de las fuentes recuperadas y de su capacidad de interpretación.

**2. Factualidad y cobertura de recuperación (RAG Approach)**

El sistema RAG se basa exclusivamente en:
- Chunks embebidos del corpus real (Wikipedia)
- Recuperación vectorial precisa con ChromaDB
- Generación condicionada por los documentos recuperados

Esto genera:
- Mayor factualidad, porque el modelo solo responde con información presente en el corpus.
- Cobertura dependiente de la base de datos: si un dato no está en los chunks embebidos, el sistema no puede inventarlo (lo que evita alucinaciones).

El RAG es menos creativo, pero más confiable para respuestas basadas en hechos verificables.

**3. ¿Qué enfoque es mejor para cada tipo de pregunta?**
| Tipo de pregunta                                                 | Mejor enfoque   | Razón                                                                |
| ---------------------------------------------------------------- | --------------- | -------------------------------------------------------------------- |
| **Abiertas, interpretativas, ensayísticas**                      | **Multi-Agent** | Capacidad de síntesis, redacción extendida, razonamiento flexible.   |
| **Factuales, específicas y basadas en conocimiento verificable** | **RAG**         | Recuperación directa de información documentada, mínima alucinación. |

# Conclusión:
- El Multi-Agent Workflow es ideal cuando se requiere creatividad, integración de múltiples fuentes o manejo de ideas difusas.
- El RAG Summarizer es superior cuando la prioridad es precisión, verificación y trazabilidad de las fuentes.
