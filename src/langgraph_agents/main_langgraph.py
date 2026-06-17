import os
import sys
import time
import json
from typing import List, Any

# Add project root to sys.path so we can run this as a standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import subprocess


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

from dotenv import load_dotenv

load_dotenv()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_pipeline.phase1_ingestion import ReportIngestor
from src.langgraph_agents.graph_builder import process_full_report
from src.core.llm_factory import get_llm


def generate_global_context(chunks: List[Any], llm: BaseChatModel) -> str:
    """
    Objetivo: Generar un resumen de alto nivel (Threat Actor, Malware, Target) a partir de la
    introducción del reporte. Este "contexto global" se inyectará después en todos los nodos
    de extracción para ayudar al LLM a resolver pronombres (ej: "ellos" -> APT29) y no perder
    el hilo conductor de la narrativa durante el procesamiento paralelo por chunks.
    """
    # Extract text from the first 4 chunks
    # Phase1_ingestion chunks are LangChain Document objects
    intro_texts = []
    for chunk in chunks[:4]:
        if hasattr(chunk, "page_content"):
            intro_texts.append(chunk.page_content)
        elif isinstance(chunk, dict) and "text" in chunk:
            intro_texts.append(chunk["text"])
        else:
            intro_texts.append(str(chunk))

    combined_intro = "\n\n".join(intro_texts)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert Cyber Threat Intelligence analyst."),
            (
                "human",
                "Read the following introductory sections of a CTI report.\n"
                "Write a 3-sentence summary identifying the Threat Actor, the primary Malware, and the Target industry/country. "
                "If unknown, state it explicitly.\n\n"
                "Text:\n{text}",
            ),
        ]
    )

    chain = prompt | llm
    print("[*] Generating Global Context...")
    try:
        response = chain.invoke({"text": combined_intro})

        # Extracción robusta del contenido
        if isinstance(response.content, str):
            context = response.content
        elif isinstance(response.content, list):
            text_parts = []
            for item in response.content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            context = " ".join(text_parts)
        else:
            context = str(response.content)

        print(f"[INFO] Global Context Generated:\n{context}\n")
        return context
    except Exception as e:
        print(f"[ERROR] Generating global context: {e}")
        return "Global context could not be generated."


def run_langgraph_extraction(pdf_path: str):
    """
    Objetivo: Actuar como punto de entrada limpio para invocar toda la arquitectura multi-agente
    sobre un PDF. Orquesta la Fase 1 (ingesta), extrae el contexto global y lanza el grafo.
    Devuelve los TTPs consolidados y las métricas de tiempo para la evaluación.
    """
    t0 = time.time()

    import hashlib
    file_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    cache_path = os.path.join("data", "output", "evaluations", f"cache_langgraph_{file_hash}.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[*] Resuming from Cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        sanitized_chunks = cache_data.get("pending_chunks", [])
        global_context = cache_data.get("global_context", "")
        completed_ttps = cache_data.get("completed_ttps", [])
        p1_time = cache_data.get("p1_time", 0.0)
        p2_time = cache_data.get("p2_time", 0.0)
        previous_timing = cache_data.get("timing_metrics", {})
        
        if not sanitized_chunks and not completed_ttps:
            print("[-] Cache exists but is empty. Restarting extraction.")
            os.remove(cache_path)
            return run_langgraph_extraction(pdf_path)
            
    else:
        use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in (
            "true",
            "1",
            "yes",
        )
        ingestor = ReportIngestor(chunk_size=3500, chunk_overlap=500, use_vlm=use_vlm_env)

        raw_chunks = ingestor.process_report(pdf_path)
        p1_time = time.time() - t0

        sanitized_chunks = []
        for idx, c in enumerate(raw_chunks):
            chunk_data = {}
            if hasattr(c, "page_content"):
                chunk_data = {"text": c.page_content, "metadata": c.metadata}
            elif isinstance(c, dict) and "text" in c:
                chunk_data = c
            else:
                chunk_data = {"text": str(c), "metadata": {}}
            
            chunk_data["chunk_id"] = f"chunk_{idx}"
            sanitized_chunks.append(chunk_data)

        if not sanitized_chunks:
            return [], {"error": "No text extracted"}

        t1 = time.time()
        llm = get_llm(temperature=0.0)
        global_context = generate_global_context(sanitized_chunks, llm)
        p2_time = time.time() - t1
        
        completed_ttps = []
        previous_timing = {}
        
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "pending_chunks": sanitized_chunks,
                "global_context": global_context,
                "completed_ttps": completed_ttps,
                "p1_time": p1_time,
                "p2_time": p2_time,
                "timing_metrics": previous_timing
            }, f, ensure_ascii=False, indent=2)

    t2 = time.time()
    result_dict = process_full_report(pdf_path, global_context, sanitized_chunks, cache_path, completed_ttps, previous_timing)

    t1 = time.time()
    p3_time = time.time() - t2
    extracted_ttps = result_dict.get("extracted_ttps", [])

    timing_metrics = {
        "phase1_ingestion_seconds": round(p1_time, 2),
        "phase3_retrieval_seconds": round(p2_time, 2),
        "phase4_inference_seconds": round(p3_time, 2),
        "langgraph_internal_breakdown": result_dict.get("timing_breakdown_phase3", {}),
        "input_tokens": result_dict.get("timing_breakdown_phase3", {}).get("input_tokens", 0),
        "output_tokens": result_dict.get("timing_breakdown_phase3", {}).get("output_tokens", 0),
        "api_crashed": result_dict.get("timing_breakdown_phase3", {}).get("api_crashed", False)
    }

    # Borrar caché si fue exitoso (no hubo crash de API)
    if not timing_metrics.get("api_crashed", False) and os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass

    return extracted_ttps, timing_metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.langgraph_agents.main_langgraph <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("LANGGRAPH MULTI-AGENT CTI EXTRACTION PIPELINE")
    print(f"{'=' * 60}\n")

    total_start_time = time.time()

    print(f"[*] [INFO] Phase 1: Ingesting PDF: {pdf_path}")
    t0 = time.time()

    import hashlib
    file_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    cache_path = os.path.join("data", "output", "evaluations", f"cache_langgraph_{file_hash}.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[*] Resuming from Cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        sanitized_chunks = cache_data.get("pending_chunks", [])
        global_context = cache_data.get("global_context", "")
        completed_ttps = cache_data.get("completed_ttps", [])
        p1_time = cache_data.get("p1_time", 0.0)
        p2_time = cache_data.get("p2_time", 0.0)
        previous_timing = cache_data.get("timing_metrics", {})
        
        if not sanitized_chunks and not completed_ttps:
            print("[-] Cache exists but is empty. Restarting extraction.")
            os.remove(cache_path)
            sys.exit(1)
            
    else:
        use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in (
            "true",
            "1",
            "yes",
        )
        ingestor = ReportIngestor(chunk_size=3500, chunk_overlap=500, use_vlm=use_vlm_env)

        raw_chunks = ingestor.process_report(pdf_path)
        p1_time = time.time() - t0
        print(f"[+] [INFO] Phase 1: Extracted {len(raw_chunks)} chunks.\n")

        sanitized_chunks = []
        for idx, c in enumerate(raw_chunks):
            chunk_data = {}
            if hasattr(c, "page_content"):
                chunk_data = {"text": c.page_content, "metadata": c.metadata}
            elif isinstance(c, dict) and "text" in c:
                chunk_data = c
            else:
                chunk_data = {"text": str(c), "metadata": {}}
            
            chunk_data["chunk_id"] = f"chunk_{idx}"
            sanitized_chunks.append(chunk_data)

        if not sanitized_chunks:
            print("[-] No text could be extracted from the PDF. Exiting.")
            sys.exit(1)

        # 2. Global Context Generation Phase
        print("[*] [INFO] Phase 2: Generating Global Context...")
        t1 = time.time()
        llm = get_llm(temperature=0.0)
        global_context = generate_global_context(sanitized_chunks, llm)
        p2_time = time.time() - t1
        
        completed_ttps = []
        previous_timing = {}
        
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "pending_chunks": sanitized_chunks,
                "global_context": global_context,
                "completed_ttps": completed_ttps,
                "p1_time": p1_time,
                "p2_time": p2_time,
                "timing_metrics": previous_timing
            }, f, ensure_ascii=False, indent=2)

    # 3. Execution Phase (Map-Reduce)
    print("\n[*] [INFO] Phase 4: Starting Graph Execution (Map-Reduce Inference)...")
    t2 = time.time()
    # === CAMBIO AQUI ===
    result_dict = process_full_report(pdf_path, global_context, sanitized_chunks, cache_path, completed_ttps, previous_timing)
    p3_time = time.time() - t2
    print("[+] [INFO] Phase 4: Graph Execution Completed.")

    # Extraer variables
    extracted_ttps = result_dict.get("extracted_ttps", [])
    p3_breakdown = result_dict.get("timing_breakdown_phase3", {})
    # ===================


    # 4. Construct Final JSON matching LangChain standard
    total_execution_time = time.time() - total_start_time

    output_data = {
        "source_file": pdf_path,
        "total_execution_time_seconds": round(total_execution_time, 2),
        "total_execution_time_minutes": round(total_execution_time / 60.0, 2),
        "timing_breakdown": {
            "phase1_ingestion_seconds": round(p1_time, 2),
            "phase3_retrieval_seconds": round(p2_time, 2),
            "phase4_inference_seconds": round(p3_time, 2),
            "langgraph_internal_breakdown": p3_breakdown,
        },
        "extracted_ttps": extracted_ttps,
    }

    # 5. Save Output
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/output")
    )
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(pdf_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(
        output_dir, f"{file_name_without_ext}_langgraph_results.json"
    )

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print("EXTRACTION SUCCESSFUL")
        print(f"Time Taken: {round(total_execution_time / 60.0, 2)} minutes")
        print(f"Results saved to: {output_file_path}")
        print(f"{'=' * 60}")
        
        if not p3_breakdown.get("api_crashed", False) and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
    except Exception as e:
        print(f"[!] Error saving output file: {e}")
