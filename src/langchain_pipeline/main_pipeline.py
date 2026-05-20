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

    # 2. Iniciar Ollama en segundo plano
    print("🦙 [BOOTSTRAP] Iniciando servicio Ollama...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("⚠️ [BOOTSTRAP] 'ollama' no está instalado o no se encuentra en el PATH.")

    # 3. Iniciar Qdrant en Docker
    print("🗄️ [BOOTSTRAP] Levantando contenedor de Qdrant vía Docker...")
    try:
        # Usamos subprocess.run sin check=True para no detenernos si ya está corriendo
        subprocess.run(
            ["docker", "run", "-d", "-p", "6333:6333", "-p", "6334:6334", "qdrant/qdrant"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("⚠️ [BOOTSTRAP] 'docker' no encontrado. Asegúrate de tener Docker instalado.")

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

from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Import pipeline phases
from src.langchain_pipeline.phase1_ingestion import ReportIngestor
from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from src.langchain_pipeline.phase4_inference import TTPAnalyzer

# Configuración básica de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    ingestor = ReportIngestor(chunk_size=1500, chunk_overlap=300)
    report_chunks = ingestor.process_report(pdf_path)
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
        # Se asume que el Indexer (Fase 2) ya construyó la DB previamente
        client = QdrantClient(url=qdrant_url)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
    except Exception as e:
        logger.error(f"Fallo crítico al conectar con Qdrant. ¿Está el contenedor corriendo? Error: {e}")
        return json.dumps({"error": "Fallo de conexión a Base de Datos Vectorial", "detalle": str(e)})

    # ------------------------------------------------------------------
    # PHASE 3: Hybrid Retrieval y Cross-Encoder Reranking
    # ------------------------------------------------------------------
    logger.info("[FASE 3] Recuperando y re-evaluando candidatos de MITRE...")
    retriever = CandidateRetriever(vector_store=vector_store)
    # top_k y threshold pueden ser ajustados empíricamente
    filtered_candidates = retriever.get_filtered_mitre_candidates(report_chunks, threshold=0.45)
    
    if not filtered_candidates:
        logger.warning("[FASE 3] No se encontraron candidatos con alto grado de confianza. Fin de pipeline.")
        return json.dumps([])
        
    # ------------------------------------------------------------------
    # PHASE 4: Inferencia LLM y Salida Estructurada
    # ------------------------------------------------------------------
    logger.info("[FASE 4] Consultando al LLM para confirmación estructurada...")
    
    model_name = os.getenv("LLM_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Instanciamos ChatOllama forzando temperature=0 para obtener máxima determinística analítica
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)
    
    analyzer = TTPAnalyzer(llm=llm)
    confirmed_ttps = analyzer.analyze_candidates(filtered_candidates)
    
    # ------------------------------------------------------------------
    # EXTRACCIÓN DE RESULTADOS
    # ------------------------------------------------------------------
    # Convertimos los objetos de Pydantic a diccionarios y luego a JSON
    output_dict_list = [ttp.model_dump() for ttp in confirmed_ttps]
    
    logger.info("--- Pipeline Finalizado con Éxito ---")
    return output_dict_list


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
    
    # run_cti_extraction ahora devuelve una lista de diccionarios
    resultados_lista = run_cti_extraction(target_pdf)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Crear el JSON final incluyendo métricas
    output_data = {
        "source_file": target_pdf,
        "execution_time_seconds": round(execution_time, 2),
        "execution_time_minutes": round(execution_time / 60, 2),
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
