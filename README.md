# CTI-RAG-Mapper (TFM)

Este repositorio contiene el código base y la infraestructura del Trabajo de Fin de Máster (TFM) desarrollado para la extracción automatizada de Tácticas, Técnicas y Procedimientos (TTPs) del framework MITRE ATT&CK a partir de reportes de inteligencia de amenazas (CTI - Cyber Threat Intelligence) no estructurados.

El sistema emplea Modelos de Lenguaje Grande (LLMs) y técnicas de Recuperación Aumentada por Generación (RAG). El proyecto contrasta un enfoque de extracción secuencial frente a un ecosistema basado en agentes autónomos.

## Arquitectura y Enfoques

El repositorio implementa dos aproximaciones para la extracción de TTPs:

1. **Pipeline Secuencial (LangChain)**: Enfoque RAG híbrido basado en un flujo secuencial (fases de ingestión, indexación, recuperación y map-reduce de inferencia).
2. **Arquitectura Multi-Agente (LangGraph)**: Enfoque basado en agentes autónomos colaborativos que evalúan la información y aplican lógica de consolidación semántica.

## Estructura del Repositorio

El proyecto está organizado de la siguiente manera:

```text
/
├── data/                    # Reportes CTI crudos (PDF/TXT), base de datos de MITRE ATT&CK (STIX/JSON) y resultados de evaluaciones.
├── dashboard/               # Interfaz de usuario en Streamlit para la inspección de TTPs extraídos.
├── evaluation/              # Scripts para el procesamiento y evaluación de los conjuntos de datos CTI-HAL, CTIBench y APT_REPORT.
├── infrastructure/          # Archivos de despliegue Docker Compose para la base de datos vectorial Qdrant y modelos.
├── src/                     # Código fuente de la aplicación.
│   ├── core/                # Elementos comunes (enmascaramiento de IoCs, esquemas Pydantic, factorías de modelos y embeddings).
│   ├── langchain_pipeline/  # Implementación del enfoque secuencial en 4 fases.
│   ├── langgraph_agents/    # Implementación multi-agente basada en grafos de estado.
│   └── scripts/             # Scripts adicionales.
├── ARCHITECTURE.md          # Especificaciones de los componentes de software.
├── run_all_evals.sh         # Script para la ejecución de evaluaciones comparativas.
└── requirements.txt         # Dependencias del entorno Python.
```

## Tecnologías Principales

- **Gestión LLM:** LangChain, LangGraph.
- **Modelos de Lenguaje:** Ollama (local), OpenAI, Google GenAI (Gemini).
- **Procesamiento Vectorial y Recuperación:** Qdrant Client, FastEmbed, Sentence-Transformers.
- **Procesamiento de Documentos:** PyMuPDF, EasyOCR, Docling, Tesseract.
- **Interfaz Gráfica:** Streamlit.

## Instalación y Uso

### 1. Preparación del Entorno

Se requiere Python 3.9 o superior. Se recomienda el uso de un entorno virtual (venv o conda).

```bash
# Instalación de dependencias del sistema para OCR (ejemplo Ubuntu/Debian)
sudo apt-get install -y tesseract-ocr poppler-utils

# Instalación de dependencias de Python
pip install -r requirements.txt
```

### 2. Configuración de Variables de Entorno

Copiar la plantilla de configuración (si existe `.env.example`) a un archivo `.env` y establecer las claves de API y parámetros de los servicios empleados (OpenAI, Gemini, Qdrant).

### 3. Ejecución de la Infraestructura Local

El entorno requiere la base de datos vectorial Qdrant. La gestión de contenedores locales (incluyendo Ollama, en su caso) se realiza mediante Docker Compose.

```bash
cd infrastructure
docker-compose up -d
```

### 4. Ejecución de la Interfaz Visual (Dashboard)

El repositorio incluye un visor para la revisión de las extracciones obtenidas.

```bash
streamlit run dashboard/cti_inspector.py
```

### 5. Ejecución de Evaluaciones

Las evaluaciones comparativas (LangChain vs LangGraph) en sus distintas configuraciones (activación de VLM y Repetición de Prompt) se pueden lanzar a través del script proporcionado:

```bash
# Ejecución empleando el script run_eval.py por defecto
bash run_all_evals.sh

# Ejecución especificando un conjunto de datos particular
bash run_all_evals.sh "evaluation/run_eval_ctihal.py"
```

Los resultados finales, tiempos de ejecución y recuento de tokens se exportan en formato JSON dentro del directorio `data/output/evaluations/`.
