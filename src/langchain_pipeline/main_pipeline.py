import sys
import os
import subprocess
import time

# Add the project root to the python path so imports from src work smoothly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def bootstrap_environment():
    """Instala dependencias e inicia los servicios requeridos (Ollama y Qdrant)."""
    print("🚀 [BOOTSTRAP] Configurando el entorno de ejecución...")
    
    # 1. Instalar dependencias
    print("📦 [BOOTSTRAP] Instalando dependencias de requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    except Exception as e:
        print(f"⚠️ [BOOTSTRAP] Error instalando dependencias: {e}")

    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "ollama":
        # 2. Iniciar Ollama en segundo plano
        print("🦙 [BOOTSTRAP] Iniciando servicio Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("⚠️ [BOOTSTRAP] 'ollama' no está instalado o no se encuentra en el PATH.")

    print("⏳ [BOOTSTRAP] Esperando 5 segundos para que los servicios estén listos...\n")
    time.sleep(5)

# Ejecutamos el bootstrap ANTES de importar librerías de terceros
# para evitar el error ModuleNotFoundError
bootstrap_environment()

import json
import logging
from typing import List

from dotenv import load_dotenv
# Cargar variables de entorno ANTES de importar LangChain (forzando override para que actualice la API key si cambió)
load_dotenv(override=True)

# Import pipeline phases (Ingestion FIRST to prevent OpenMP/segfault conflicts with torch/onnxruntime)
from src.langchain_pipeline.phase1_ingestion import ReportIngestor

# Se importarán dinámicamente según la configuración
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from src.langchain_pipeline.phase4_inference import TTPAnalyzer

# Configuración básica de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silenciar los molestos errores de conexión en segundo plano de LangSmith/HTTP
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
    logger.info(f"--- Iniciando extracción CTI para: {pdf_path} ---")
    
    # ------------------------------------------------------------------
    # PHASE 1: Ingesta y Particionado Semántico
    # ------------------------------------------------------------------
    logger.info("[FASE 1] Ingestando y sanitizando el reporte...")
    t0 = time.time()
    
    # Selector de Ingesta (Pruebas A/B TFM)
    use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in ("true", "1", "yes")
    ingestor = ReportIngestor(chunk_size=1500, chunk_overlap=300, use_vlm=use_vlm_env)
    
    report_chunks = ingestor.process_report(pdf_path)
    p1_time = time.time() - t0
    logger.info(f"[FASE 1] Se generaron {len(report_chunks)} chunks enmascarados.")
    
    # ------------------------------------------------------------------
    # PHASE 2/3 SETUP: Conexión a Base de Datos Vectorial (MITRE)
    # ------------------------------------------------------------------
    logger.info("[SETUP] Conectando a la Base Vectorial Qdrant...")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
    collection_name = "mitre_attack"
    
    try:
        client = QdrantClient(url=qdrant_url)
        
        # Verificamos si la colección existe; si no, la creamos
        if not client.collection_exists(collection_name):
            logger.info(f"[SETUP] La colección '{collection_name}' no existe. Ejecutando Indexer (Fase 2)...")
            from src.langchain_pipeline.phase2_indexer import setup_mitre_index
            setup_mitre_index(qdrant_url, collection_name)
            logger.info("[SETUP] Colección MITRE ATT&CK construida exitosamente.")
            # IMPORTANTE: Re-instanciar el cliente Qdrant. La indexación tarda mucho tiempo
            # y el pool de conexiones HTTP original puede cerrarse por inactividad.
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
        logger.error(f"Fallo crítico al conectar con Qdrant o inicializar la DB. ¿Está el contenedor corriendo? Error: {e}")
        timing_metrics = {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": 0.0,
            "phase4_inference_seconds": 0.0
        }
        return [{"error": "Fallo de conexión o inicialización en Qdrant", "detalle": str(e)}], timing_metrics

    # ------------------------------------------------------------------
    # PHASE 3: Hybrid Retrieval y Cross-Encoder Reranking
    # ------------------------------------------------------------------
    logger.info("[FASE 3] Recuperando y re-evaluando candidatos de MITRE...")
    retriever = CandidateRetriever(vector_store=vector_store)
    # top_k y threshold pueden ser ajustados empíricamente
    t1 = time.time()
    filtered_candidates = retriever.get_filtered_mitre_candidates(report_chunks, threshold=0.4)
    p3_time = time.time() - t1
    
    if not filtered_candidates:
        logger.warning("[FASE 3] No se encontraron candidatos con alto grado de confianza. Fin de pipeline.")
        timing_metrics = {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": round(p3_time, 2),
            "phase4_inference_seconds": 0.0
        }
        return [], timing_metrics
        
    # ------------------------------------------------------------------
    # PHASE 4: Inferencia LLM y Salida Estructurada
    # ------------------------------------------------------------------
    logger.info("[FASE 4] Consultando al LLM para confirmación estructurada...")
    
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "gemini":
        logger.info("[FASE 4] Configurando Google Gemini para inferencia (más rápido)...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Usamos gemini-3.5-flash por defecto (1.5 está deprecado)
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        
        if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
            logger.error("⚠️ [FASE 4] Falta la GOOGLE_API_KEY en el entorno para usar Gemini.")
            timing_metrics = {
                "phase1_ingestion_seconds": round(p1_time, 2),
                "phase3_retrieval_seconds": round(p3_time, 2),
                "phase4_inference_seconds": 0.0
            }
            return [{"error": "Falta GOOGLE_API_KEY en el .env"}], timing_metrics
            
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0)
    else:
        logger.info("[FASE 4] Configurando Ollama local para inferencia...")
        model_name = os.getenv("LLM_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        from langchain_ollama import ChatOllama
        # Instanciamos ChatOllama forzando temperature=0 para obtener máxima determinística analítica
        llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)
        
        # Verificamos si el modelo está instalado consultando la CLI de Ollama directamente
        try:
            logger.info(f"[FASE 4] Comprobando disponibilidad del modelo {model_name}...")
            check_result = subprocess.run(["ollama", "show", model_name], capture_output=True)
            
            if check_result.returncode != 0:
                logger.info(f"[FASE 4] El modelo {model_name} no está instalado. Descargando...")
                subprocess.run(["ollama", "pull", model_name], check=True)
                logger.info(f"[FASE 4] Modelo {model_name} descargado y listo para usar.")
            else:
                logger.info(f"[FASE 4] El modelo {model_name} ya está disponible en el sistema.")
                
        except Exception as e:
            logger.warning(f"[FASE 4] No se pudo verificar o descargar el modelo automáticamente mediante la CLI. Error: {e}")
            
    analyzer = TTPAnalyzer(llm=llm)
    t2 = time.time()
    confirmed_ttps = analyzer.analyze_candidates(filtered_candidates)
    p4_time = time.time() - t2
    
    # ------------------------------------------------------------------
    # EXTRACCIÓN DE RESULTADOS
    # ------------------------------------------------------------------
    # Convertimos los objetos de Pydantic a diccionarios excluyendo 'is_present' para limpiar el output final
    output_dict_list = [ttp.model_dump(exclude={'is_present'}) for ttp in confirmed_ttps]
    
    logger.info("--- Pipeline Finalizado con Éxito ---")
    timing_metrics = {
        "phase1_ingestion_seconds": round(p1_time, 2),
        "phase3_retrieval_seconds": round(p3_time, 2),
        "phase4_inference_seconds": round(p4_time, 2)
    }
    return output_dict_list, timing_metrics


if __name__ == "__main__":
    # Soporte para ejecución mediante terminal
    if len(sys.argv) < 2:
        print("Uso: python -m src.langchain_pipeline.main_pipeline <ruta_al_pdf>")
        sys.exit(1)
        
    target_pdf = sys.argv[1]
    
    if not os.path.exists(target_pdf):
        print(f"Error: El archivo '{target_pdf}' no existe o no es accesible.")
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
    print("🎯 RESULTADOS FINALES: TTPs EXTRAÍDOS")
    print("="*60)
    print(f"Tiempo de ejecución: {round(execution_time / 60, 2)} minutos")
    print(f"Resultados guardados exitosamente en: {output_file_path}")
    print("="*60)
