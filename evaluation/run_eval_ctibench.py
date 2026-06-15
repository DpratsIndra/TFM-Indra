import os
import sys
import json
import time
import argparse
import contextlib
from datetime import datetime
from tqdm import tqdm
from langchain.docstore.document import Document

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.data_loaders import CtibenchDataLoader
from evaluation.metrics_calculator import Evaluator
from src.langchain_pipeline.phase4_inference import TTPAnalyzer
from src.langgraph_agents.main_langgraph import process_full_report
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.retrieval_mode import RetrievalMode
from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv(override=True)

class FastEmbedSparse:
    def __init__(self, model_name="Qdrant/bm25"):
        self.model = SparseTextEmbedding(model_name=model_name)
    def embed_query(self, query: str):
        return list(self.model.embed([query]))[0]
    def embed_documents(self, texts):
        return list(self.model.embed(texts))

def setup_langchain_components():
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
    client = QdrantClient(url=qdrant_url)
    
    vector_store = QdrantVectorStore(
        client=client, collection_name="mitre_attack",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID
    )
    retriever = CandidateRetriever(vector_store=vector_store)
    from src.core.llm_factory import get_llm
    llm = get_llm(temperature=0.0)
    analyzer = TTPAnalyzer(llm=llm)
    
    return retriever, analyzer

def run_evaluation():
    """
    Ejecuta una evaluación sobre el dataset de CTIBench (textos cortos).
    Este dataset contiene descripciones de malware y herramientas APT.
    """
    parser = argparse.ArgumentParser(description="Run evaluation on CTIBench dataset")
    parser.add_argument("--dataset_path", type=str, default="data/eval_datasets/ctibench", help="Path to CTIBench dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of sentences to evaluate (for quick testing)")
    parser.add_argument("--pipeline", type=str, choices=["langchain", "langgraph"], default="langgraph", help="Which architecture to evaluate")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a previous JSON to resume execution")
    args = parser.parse_args()

    pipeline_type = args.pipeline

    print("\n" + "="*80)
    print("🚀 INICIANDO EVALUACIÓN CTIBENCH (TEXT-LEVEL)")
    print(f"   Pipeline:           {pipeline_type.upper()}")
    
    profile = os.getenv("EXECUTION_PROFILE", "LOCAL").upper()
    if profile == "REMOTE":
        use_gemma = os.getenv("USE_GEMMA4", "False").lower() in ("true", "1", "yes")
        model_used = os.getenv("VLLM_MODEL_NAME_GEMMA", "gemma4") if use_gemma else os.getenv("VLLM_MODEL_NAME", "gpt-oss-20b")
    else:
        model_used = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    
    print(f"   Model Used:         {model_used} ({profile})")
    print("="*80)

    print(f"\n[*] Loading CTIBench dataset from: {args.dataset_path}")
    loader = CtibenchDataLoader(base_path=args.dataset_path)
    df = loader.load()
    
    if df.empty:
        print("[!] Error: No data found. Ensure cti-ate.tsv exists.")
        sys.exit(1)
        
    limit = args.limit if args.limit else 200
    print(f"[*] Subsampling {limit} random records for testing...")
    df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        
    print(f"[*] Total sentences for evaluation: {len(df)}")
    
    if pipeline_type == "langchain":
        retriever, analyzer = setup_langchain_components()
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"ctibench_eval_{pipeline_type}_{model_used}_{timestamp}.json"
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[*] Resuming from checkpoint: {args.resume_from}")
        output_path = args.resume_from
        with open(args.resume_from, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        if "detailed_executions" in prev_data:
            detailed_results = prev_data["detailed_executions"]
            start_idx = len(detailed_results)
            hierarchy_stats = prev_data.get("hierarchy_analysis", hierarchy_stats)
            for d in detailed_results:
                predicted_labels.append(d.get("predicted_labels_normalized", []))
            print(f"[*] Resumed {start_idx} processed sentences. Continuing from index {start_idx}...")
        previous_execution_minutes = prev_data.get("total_execution_minutes", 0.0)
    else:
        previous_execution_minutes = 0.0
        output_path = os.path.join("data", "output", "evaluations", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    predicted_labels = []
    detailed_results = []
    hierarchy_stats = {"total_exact_matches": 0, "total_more_detailed": 0, "total_more_general": 0}
    
    start_time = time.time()
    
    start_idx_val = len(predicted_labels)
    df_remaining = df.iloc[start_idx_val:]
    
    for idx, row in tqdm(df_remaining.iterrows(), total=len(df), desc=f"Eval: {pipeline_type}", initial=start_idx_val):
            
        text = row['text']
        true_lbls = row['true_labels']
        predicted_ids = []
        
        try:
            if pipeline_type == "langchain":
                doc = Document(page_content=text, metadata={"chunk_index": idx, "page_number": 1})
                candidates = retriever.get_filtered_mitre_candidates([doc], threshold=0.2)
                if candidates:
                    detections, artificial_delay, tokens = analyzer.analyze_candidates(candidates)
                else:
                    detections, artificial_delay, tokens = [], 0.0, {"input_tokens": 0, "output_tokens": 0}
                
                extracted_payload = [d.model_dump() for d in detections if d.is_present]
                    
            elif pipeline_type == "langgraph":
                sanitized_chunks = [{"text": text, "metadata": {"chunk_index": idx}}]
                global_context = "This is a tool/malware description from CTIBench."
                
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        result = process_full_report(
                            source_file=f"ctibench_doc_{idx}",
                            global_context=global_context,
                            sanitized_chunks=sanitized_chunks
                        )
                        extracted_payload = result.get("extracted_ttps", [])
                        predicted_ids = [ttp.get("technique_id") for ttp in extracted_payload]
                        timing_ph3 = result.get("timing_breakdown_phase3", {})
                        tokens = {
                            "input_tokens": timing_ph3.get("input_tokens", 0),
                            "output_tokens": timing_ph3.get("output_tokens", 0)
                        }
                        
            # Normalización y Trazabilidad Jerárquica
            normalized_preds = []
            traceability_summary = {"exact": 0, "more_detailed": 0, "more_general": 0}
            
            for result_ttp in extracted_payload:
                pred_id = result_ttp.get("technique_id")
                if not pred_id: continue
                
                matched_true = None
                match_type = "false_positive"
                
                for t in true_lbls:
                    if pred_id == t:
                        matched_true = t
                        match_type = "exact"
                        break
                    elif "." not in t and pred_id.startswith(t + "."):
                        matched_true = t
                        match_type = "more_detailed"
                        break
                    elif "." not in pred_id and t.startswith(pred_id + "."):
                        matched_true = t
                        match_type = "more_general"
                        break
                        
                result_ttp["hierarchy_match"] = match_type
                if matched_true:
                    result_ttp["matched_true_label"] = matched_true
                    normalized_preds.append(matched_true)
                    traceability_summary[match_type] += 1
                    
                    if match_type == "exact": hierarchy_stats["total_exact_matches"] += 1
                    elif match_type == "more_detailed": hierarchy_stats["total_more_detailed"] += 1
                    elif match_type == "more_general": hierarchy_stats["total_more_general"] += 1
                else:
                    normalized_preds.append(pred_id)
            
            predicted_ids = list(set(normalized_preds))
            tp = list(set(true_lbls) & set(predicted_ids))
            fp = list(set(predicted_ids) - set(true_lbls))
            fn = list(set(true_lbls) - set(predicted_ids))
            
            metrics = {"TP": len(tp), "FP": len(fp), "FN": len(fn)}
            if len(true_lbls) == 0 and len(predicted_ids) == 0:
                metrics["TN"] = 1
            
            detailed_results.append({
                "sentence_id": idx,
                "source_file": f"ctibench_sentence_{idx}",
                "text": text,
                "true_labels": true_lbls,
                "predicted_labels_raw": [r.get("technique_id") if isinstance(r, dict) else r.technique_id for r in extracted_payload],
                "predicted_labels_normalized": predicted_ids,
                "metrics": metrics,
                "traceability_summary": traceability_summary,
                "timing_breakdown": tokens,
                "extracted_ttps": extracted_payload
            })
            
        except KeyboardInterrupt:
            print("\n[!] INTERRUPCIÓN MANUAL (Ctrl+C). Cancelando de forma segura y guardando los reportes completados...")
            llm_crashed = True
            break
        except Exception as e:
            error_str = str(e)
            tqdm.write(f" [!] ERROR crítico en id {idx}: {error_str}")
            
            api_errors = ["429", "quota", "resourceexhausted", "503", "500", "timeout", "not_found", "api", "connection", "unavailable"]
            if any(err in error_str.lower() for err in api_errors):
                print("\n[!] CORTE DE LLM/API DETECTADO. Deteniendo ejecución para salvaguardar el progreso.")
                llm_crashed = True
                break
                
            predicted_ids = []
            detailed_results.append({"sentence_id": idx, "source_file": f"ctibench_doc_{idx}", "error": error_str})
            
        predicted_labels.append(predicted_ids)
        
        # NUEVO: AUTO-GUARDADO (CHECKPOINT)
        current_session_minutes = (time.time() - start_time) / 60.0
        partial_output = {
            "status": f"INCOMPLETE - Processed {idx + 1}/{len(df)}",
            "model_used": model_used,
            "total_execution_minutes": round(previous_execution_minutes + current_session_minutes, 2),
            "hierarchy_analysis": hierarchy_stats,
            "detailed_executions": detailed_results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(partial_output, f, indent=4)
            
        time.sleep(5.0)  # Breve pausa para no saturar APIs
        
    if len(predicted_labels) < len(df):
        print(f"\n[!] Evaluadas solo {len(predicted_labels)} de {len(df)} sentencias debido a interrupción.")
        df = df.iloc[:len(predicted_labels)].copy()
        
    if len(df) == 0:
        print("\n[!] No se evaluó ninguna sentencia. Saliendo.")
        sys.exit(429 if 'llm_crashed' in locals() else 0)
        
    df['predicted_labels'] = predicted_labels
    evaluator = Evaluator(df)
    results = evaluator.evaluate()
    
    current_session_minutes = (time.time() - start_time) / 60.0
    total_execution_minutes = round(previous_execution_minutes + current_session_minutes, 2)
    
    # Estructura JSON estandarizada
    final_output = {
        "model_used": model_used,
        "total_execution_minutes": total_execution_minutes,
        "global_metrics": results,
        "hierarchy_analysis": hierarchy_stats,
        "detailed_executions": detailed_results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print("\n" + "="*80)
    print(f"🎯 RESULTADOS MICRO PARA CTIBENCH ({pipeline_type.upper()})")
    print("="*80)
    print(f"F0.5-Score: {results['micro']['f0.5']}")
    print(f"F1-Score:   {results['micro']['f1']}")
    print(f"Precision:  {results['micro']['precision']}")
    print(f"Recall:     {results['micro']['recall']}")
    print(f"\n💾 Resumen completo guardado en: {output_path}")
    if 'llm_crashed' in locals():
        sys.exit(429)

if __name__ == "__main__":
    run_evaluation()
