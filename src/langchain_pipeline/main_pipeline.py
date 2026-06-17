import sys
import os
import subprocess
import time

# Add the project root to the python path so imports from src work smoothly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def bootstrap_environment():
    """Instala dependencias e inicia los servicios requeridos (Ollama y Qdrant)."""
    print("[INFO] Configuring execution environment...")

    # Pip install step removed to avoid console spam.
    # User is expected to have installed requirements.txt manually.

    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if llm_provider == "ollama":
        print("[INFO] Initializing Ollama service...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("[WARNING] 'ollama' is not installed or not in PATH.")

    print("[INFO] Waiting 5 seconds for services to be ready...\n")
    time.sleep(5)


bootstrap_environment()

import json
import logging

from dotenv import load_dotenv

load_dotenv()

from src.langchain_pipeline.phase1_ingestion import ReportIngestor

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from src.langchain_pipeline.phase4_inference import TTPAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("langsmith.client").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


def run_cti_extraction(pdf_path: str) -> str:
    """
    Ejecutar el flujo original y secuencial de LangChain (Fases 1 a 4).
    A diferencia de LangGraph, aquí la inferencia (Fase 4) se hace chunk por chunk
    y se vuelca directamente en JSON, sin agentes de validación intermedios.
    Se utiliza como baseline para comparar el rendimiento con LangGraph en las evaluaciones.
    """
    logger.info(f"[INFO] Starting CTI extraction for: {pdf_path}")

    # ------------------------------------------------------------------
    # PHASE 1: Ingesta y Particionado Semántico
    # ------------------------------------------------------------------
    logger.info("[INFO] Phase 1: Ingesting and sanitizing the report...")
    t0 = time.time()

    use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in (
        "true",
        "1",
        "yes",
    )
    ingestor = ReportIngestor(chunk_size=3500, chunk_overlap=500, use_vlm=use_vlm_env)

    report_chunks = ingestor.process_report(pdf_path)
    p1_time = time.time() - t0
    logger.info(f"[INFO] Phase 1: Generated {len(report_chunks)} masked chunks.")

    # ------------------------------------------------------------------
    # PHASE 2/3 SETUP: Conexión a Base de Datos Vectorial (MITRE)
    # ------------------------------------------------------------------
    logger.info("[INFO] Setup: Connecting to Qdrant Vector Database...")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
    collection_name = "mitre_attack"

    try:
        client = QdrantClient(url=qdrant_url)

        if not client.collection_exists(collection_name):
            logger.info(
                f"[INFO] Setup: Collection '{collection_name}' does not exist. Running Indexer (Phase 2)..."
            )
            from src.langchain_pipeline.phase2_indexer import setup_mitre_index

            setup_mitre_index(qdrant_url, collection_name)
            logger.info("[INFO] Setup: MITRE ATT&CK collection built successfully.")
            client = QdrantClient(url=qdrant_url)

        from src.core.embedding_factory import get_embeddings

        embeddings = get_embeddings()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    except Exception as e:
        logger.error(
            f"[ERROR] Critical failure connecting to Qdrant or initializing DB. Is the container running? Error: {e}"
        )
        timing_metrics = {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": 0.0,
            "phase4_inference_seconds": 0.0,
        }
        return [
            {"error": "Qdrant connection or initialization failure", "detail": str(e)}
        ], timing_metrics

    # ------------------------------------------------------------------
    # PHASE 3: Hybrid Retrieval y Cross-Encoder Reranking
    # ------------------------------------------------------------------
    logger.info("[INFO] Phase 3: Retrieving and re-evaluating MITRE candidates...")
    retriever = CandidateRetriever(vector_store=vector_store)
    t1 = time.time()
    filtered_candidates = retriever.get_filtered_mitre_candidates(
        report_chunks, threshold=0.2
    )
    p3_time = time.time() - t1

    if not filtered_candidates:
        logger.warning(
            "[WARNING] Phase 3: No high-confidence candidates found. End of pipeline."
        )
        timing_metrics = {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": round(p3_time, 2),
            "phase4_inference_seconds": 0.0,
        }
        return [], timing_metrics

    # ------------------------------------------------------------------
    # PHASE 4: Inferencia LLM y Salida Estructurada
    # ------------------------------------------------------------------
    logger.info("[INFO] Phase 4: Querying LLM for structured confirmation...")

    logger.info("[INFO] Phase 4: Configuring LLM for inference using llm_factory...")
    from src.core.llm_factory import get_llm

    llm = get_llm(temperature=0.0)

    analyzer = TTPAnalyzer(llm=llm)
    t2 = time.time()
    confirmed_ttps, artificial_delay, tokens = analyzer.analyze_candidates(filtered_candidates, cache_key=pdf_path)
    p4_time = (time.time() - t2) - artificial_delay

    # ------------------------------------------------------------------
    # EXTRACCIÓN DE RESULTADOS
    # ------------------------------------------------------------------
    output_dict_list = [
        ttp.model_dump(exclude={"is_present"}) for ttp in confirmed_ttps
    ]

    logger.info("[INFO] Pipeline execution completed successfully.")
    timing_metrics = {
        "phase1_ingestion_seconds": round(p1_time, 2),
        "phase3_retrieval_seconds": round(p3_time, 2),
        "phase4_inference_seconds": round(p4_time, 2),
        "input_tokens": tokens.get("input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "api_crashed": tokens.get("api_crashed", False),
    }
    return output_dict_list, timing_metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.langchain_pipeline.main_pipeline <path_to_pdf>")
        sys.exit(1)

    target_pdf = sys.argv[1]

    if not os.path.exists(target_pdf):
        print(f"Error: The file '{target_pdf}' does not exist or is not accessible.")
        sys.exit(1)

    start_time = time.time()

    # run_cti_extraction ahora devuelve una lista de diccionarios y métricas de tiempo
    resultados_lista, timing_metrics = run_cti_extraction(target_pdf)

    end_time = time.time()
    execution_time = end_time - start_time

    # Crear el JSON final incluyendo métricas
    output_data = {
        "source_file": target_pdf,
        "total_execution_time_seconds": round(execution_time, 2),
        "total_execution_time_minutes": round(execution_time / 60, 2),
        "timing_breakdown": timing_metrics,
        "extracted_ttps": resultados_lista,
    }

    resultados_json = json.dumps(output_data, indent=4, ensure_ascii=False)

    # Asegurar que existe el directorio de salida
    output_dir = os.path.join(os.path.dirname(__file__), "../../data/output")
    os.makedirs(output_dir, exist_ok=True)

    # Generar nombre de archivo basado en el PDF original
    base_name = os.path.basename(target_pdf)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(output_dir, f"{file_name_without_ext}_results.json")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(resultados_json)

    print("\n" + "=" * 60)
    print("FINAL RESULTS: EXTRACTED TTPs")
    print("=" * 60)
    print(f"Execution time: {round(execution_time / 60, 2)} minutes")
    print(f"Results successfully saved to: {output_file_path}")
    print("=" * 60)
