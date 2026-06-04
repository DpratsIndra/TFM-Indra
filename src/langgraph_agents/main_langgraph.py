import os
import sys
import time
import json
from typing import List, Dict, Any

# Add project root to sys.path so we can run this as a standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import subprocess

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

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_pipeline.phase1_ingestion import ReportIngestor
from src.langgraph_agents.graph_builder import process_full_report

def generate_global_context(chunks: List[Any], llm: ChatGoogleGenerativeAI) -> str:
    """
    Generates a high-level summary of the report to serve as global context.
    This helps the extractor agents resolve pronouns and understand the broader scope.
    Takes the first 4 chunks to extract Threat Actor, Malware, and Target.
    """
    # Extract text from the first 4 chunks
    # Phase1_ingestion chunks are LangChain Document objects
    intro_texts = []
    for chunk in chunks[:4]:
        if hasattr(chunk, 'page_content'):
            intro_texts.append(chunk.page_content)
        elif isinstance(chunk, dict) and "text" in chunk:
            intro_texts.append(chunk["text"])
        else:
            intro_texts.append(str(chunk))
            
    combined_intro = "\n\n".join(intro_texts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Cyber Threat Intelligence analyst."),
        ("human", "Read the following introductory sections of a CTI report.\n"
                  "Write a 3-sentence summary identifying the Threat Actor, the primary Malware, and the Target industry/country. "
                  "If unknown, state it explicitly.\n\n"
                  "Text:\n{text}")
    ])
    
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
    """Función de entrada limpia para invocar todo el pipeline de LangGraph desde scripts de evaluación."""
    import time
    t0 = time.time()
    
    use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in ("true", "1", "yes")
    ingestor = ReportIngestor(chunk_size=3500, chunk_overlap=500, use_vlm=use_vlm_env)
    
    raw_chunks = ingestor.process_report(pdf_path)
    p1_time = time.time() - t0
    
    sanitized_chunks = []
    for c in raw_chunks:
        if hasattr(c, 'page_content'):
            sanitized_chunks.append({"text": c.page_content, "metadata": c.metadata})
        elif isinstance(c, dict) and "text" in c:
            sanitized_chunks.append(c)
        else:
            sanitized_chunks.append({"text": str(c), "metadata": {}})
            
    if not sanitized_chunks:
        return [], {"error": "No text extracted"}

    t1 = time.time()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0)
    global_context = generate_global_context(sanitized_chunks, llm)
    p2_time = time.time() - t1
    
    t2 = time.time()
    result_dict = process_full_report(pdf_path, global_context, sanitized_chunks)
    p3_time = time.time() - t2
    extracted_ttps = result_dict.get("extracted_ttps", [])
    
    timing_metrics = {
        "phase1_ingestion_seconds": round(p1_time, 2),
        "phase2_context_seconds": round(p2_time, 2),
        "phase3_extraction_seconds": round(p3_time, 2)
    }
    
    return extracted_ttps, timing_metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.langgraph_agents.main_langgraph <path_to_pdf>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        sys.exit(1)
        
    print(f"\n{'='*60}")
    print(f"LANGGRAPH MULTI-AGENT CTI EXTRACTION PIPELINE")
    print(f"{'='*60}\n")
    
    total_start_time = time.time()
    
    print(f"[*] [INFO] Phase 1: Ingesting PDF: {pdf_path}")
    t0 = time.time()
    
    use_vlm_env = os.getenv("USE_VLM_EXTRACTION", "False").lower() in ("true", "1", "yes")
    ingestor = ReportIngestor(chunk_size=3500, chunk_overlap=500, use_vlm=use_vlm_env)
    
    raw_chunks = ingestor.process_report(pdf_path)
    p1_time = time.time() - t0
    print(f"[+] [INFO] Phase 1: Extracted {len(raw_chunks)} chunks.\n")
    
    sanitized_chunks = []
    for c in raw_chunks:
        if hasattr(c, 'page_content'):
            sanitized_chunks.append({"text": c.page_content, "metadata": c.metadata})
        elif isinstance(c, dict) and "text" in c:
            sanitized_chunks.append(c)
        else:
            sanitized_chunks.append({"text": str(c), "metadata": {}})
    
    if not sanitized_chunks:
        print("[-] No text could be extracted from the PDF. Exiting.")
        sys.exit(1)
    
    # 2. Global Context Generation Phase
    print("[*] [INFO] Phase 2: Generating Global Context...")
    t1 = time.time()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0)
    global_context = generate_global_context(sanitized_chunks, llm)
    p2_time = time.time() - t1
    
    # 3. Execution Phase (Map-Reduce)
    print("\n[*] [INFO] Phase 3: Starting Graph Execution (Map-Reduce)...")
    t2 = time.time()
    # === CAMBIO AQUI ===
    result_dict = process_full_report(pdf_path, global_context, sanitized_chunks)
    p3_time = time.time() - t2
    print("[+] [INFO] Phase 3: Graph Execution Completed.")
    
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
            "phase2_context_seconds": round(p2_time, 2),
            "phase3_extraction_seconds": round(p3_time, 2)
        },
        "extracted_ttps": extracted_ttps
    }
    
    # 5. Save Output
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/output'))
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(pdf_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(output_dir, f"{file_name_without_ext}_langgraph_results.json")
    
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n{'='*60}")
        print(f"EXTRACTION SUCCESSFUL")
        print(f"Time Taken: {round(total_execution_time / 60.0, 2)} minutes")
        print(f"Results saved to: {output_file_path}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"[!] Error saving output file: {e}")
