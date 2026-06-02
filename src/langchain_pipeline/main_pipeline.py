import sys
import os
import subprocess
import time

# Add the project root to the python path so imports from src work smoothly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def bootstrap_environment():
    """Instala dependencias e inicia los servicios requeridos (Ollama y Qdrant)."""
    print("[INFO] Configuring execution environment...")
    
    print("[INFO] Installing dependencies from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    except Exception as e:
        print(f"[ERROR] Error installing dependencies: {e}")

    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "ollama":
        print("[INFO] Initializing Ollama service...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[WARNING] 'ollama' is not installed or not in PATH.")

    print("[INFO] Waiting 5 seconds for services to be ready...\n")
    time.sleep(5)

bootstrap_environment()

import json
import logging
from typing import List

from dotenv import load_dotenv
load_dotenv(override=True)

from src.langchain_pipeline.phase1_ingestion import ReportIngestor

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from src.langchain_pipeline.phase4_inference import TTPAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("langsmith.client").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

def run_cti_extraction(pdf_path: str) -> str:
    """
    Orquesta el flujo completo de 4 Fases de LangChain para extraer 
    las tácticas y técnicas de MITRE ATT&CK desde un reporte CTI.
    
    Args:
        pdf_path (str): Ruta absoluta o relativa al reporte en formato PDF.
        
    Returns:
        str: Cadena JSON formateada con las detecciones TTP confirmadas.
    """
    logger.info(f"[INFO] Starting CTI extraction for: {pdf_path}")
    
    # ------------------------------------------------------------------
    # PHASE 1: Ingesta y Particionado Semántico
    # ------------------------------------------------------------------
    logger.info("[INFO] Phase 1: Ingesting and sanitizing the report...")
    t0 = time.time()
    
    use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in ("true", "1", "yes")
    ingestor = ReportIngestor(chunk_size=1500, chunk_overlap=300, use_vlm=use_vlm_env)
    
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
            logger.info(f"[INFO] Setup: Collection '{collection_name}' does not exist. Running Indexer (Phase 2)...")
            from src.langchain_pipeline.phase2_indexer import setup_mitre_index
            setup_mitre_index(qdrant_url, collection_name)
            logger.info("[INFO] Setup: MITRE ATT&CK collection built successfully.")
            client = QdrantClient(url=qdrant_url)
            
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
    except Exception as e:
        logger.error(f"[ERROR] Critical failure connecting to Qdrant or initializing DB. Is the container running? Error: {e}")
        timing_metrics = {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": 0.0,
            "phase4_inference_seconds": 0.0
        }
        return [{"error": "Qdrant connection or initialization failure", "detail": str(e)}], timing_metrics

    # ------------------------------------------------------------------
    # PHASE 3: Hybrid Retrieval y Cross-Encoder Reranking
    # ------------------------------------------------------------------
    logger.info("[INFO] Phase 3: Retrieving and re-evaluating MITRE candidates...")
    retriever = CandidateRetriever(vector_store=vector_store)
    t1 = time.time()
    filtered_candidates = retriever.get_filtered_mitre_candidates(report_chunks, threshold=0.4)
    p3_time = time.time() - t1
    
    if not filtered_candidates:
        logger.warning("[WARNING] Phase 3: No high-confidence candidates found. End of pipeline.")
        timing_metrics = {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": round(p3_time, 2),
            "phase4_inference_seconds": 0.0
        }
        return [], timing_metrics
        
    # ------------------------------------------------------------------
    # PHASE 4: Inferencia LLM y Salida Estructurada
    # ------------------------------------------------------------------
    logger.info("[INFO] Phase 4: Querying LLM for structured confirmation...")
    
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "gemini":
        logger.info("[INFO] Phase 4: Configuring Google Gemini for inference...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        
        if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
            logger.error("[ERROR] Phase 4: Missing GOOGLE_API_KEY in the environment for Gemini.")
            timing_metrics = {
                "phase1_ingestion_seconds": round(p1_time, 2),
                "phase3_retrieval_seconds": round(p3_time, 2),
                "phase4_inference_seconds": 0.0
            }
            return [{"error": "Missing GOOGLE_API_KEY in .env"}], timing_metrics
            
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0)
    else:
        logger.info("[INFO] Phase 4: Configuring local Ollama for inference...")
        model_name = os.getenv("LLM_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)
        
        try:
            logger.info(f"[INFO] Phase 4: Checking availability of model {model_name}...")
            check_result = subprocess.run(["ollama", "show", model_name], capture_output=True)
            
            if check_result.returncode != 0:
                logger.info(f"[INFO] Phase 4: Model {model_name} is not installed. Downloading...")
                subprocess.run(["ollama", "pull", model_name], check=True)
                logger.info(f"[INFO] Phase 4: Model {model_name} downloaded and ready to use.")
            else:
                logger.info(f"[INFO] Phase 4: Model {model_name} is already available in the system.")
                
        except Exception as e:
            logger.warning(f"[WARNING] Phase 4: Could not automatically verify or download the model via CLI. Error: {e}")
            
    analyzer = TTPAnalyzer(llm=llm)
    t2 = time.time()
    confirmed_ttps = analyzer.analyze_candidates(filtered_candidates)
    p4_time = time.time() - t2
    
    # ------------------------------------------------------------------
    # EXTRACCIÓN DE RESULTADOS
    # ------------------------------------------------------------------
    output_dict_list = [ttp.model_dump(exclude={'is_present'}) for ttp in confirmed_ttps]
    
    logger.info("[INFO] Pipeline execution completed successfully.")
    timing_metrics = {
        "phase1_ingestion_seconds": round(p1_time, 2),
        "phase3_retrieval_seconds": round(p3_time, 2),
        "phase4_inference_seconds": round(p4_time, 2)
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
        "extracted_ttps": resultados_lista
    }
    
    resultados_json = json.dumps(output_data, indent=4, ensure_ascii=False)
    
    # Asegurar que existe el directorio de salida
    output_dir = os.path.join(os.path.dirname(__file__), '../../data/output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombre de archivo basado en el PDF original
    base_name = os.path.basename(target_pdf)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(output_dir, f"{file_name_without_ext}_results.json")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(resultados_json)
    
    print("\n" + "="*60)
    print("FINAL RESULTS: EXTRACTED TTPs")
    print("="*60)
    print(f"Execution time: {round(execution_time / 60, 2)} minutes")
    print(f"Results successfully saved to: {output_file_path}")
    print("="*60)
