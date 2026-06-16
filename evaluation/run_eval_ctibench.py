import os
import sys
import json
import time
import argparse
import contextlib
from datetime import datetime
from tqdm import tqdm
from langchain_core.documents import Document

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.data_loaders import CtibenchDataLoader
from evaluation.metrics_calculator import Evaluator
from src.langchain_pipeline.phase4_inference import TTPAnalyzer
from src.langgraph_agents.main_langgraph import process_full_report
from dotenv import load_dotenv

load_dotenv(override=True)

def setup_langchain_components():
    from src.langgraph_agents.tools import get_retriever
    from src.core.llm_factory import get_llm
    
    retriever = get_retriever()
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
    actual_limit = min(limit, len(df))
    print(f"[*] Subsampling {actual_limit} random records for testing...")
    df = df.sample(n=actual_limit, random_state=42).reset_index(drop=True)
        
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
                t_start = time.time()
                doc = Document(page_content=text, metadata={"chunk_index": idx, "page_number": 1})
                candidates = retriever.get_filtered_mitre_candidates([doc], threshold=0.2)
                p3_time = time.time() - t_start
                if candidates:
                    t_inf = time.time()
                    detections, artificial_delay, tokens = analyzer.analyze_candidates(candidates)
                    p4_time = (time.time() - t_inf) - artificial_delay
                else:
                    detections, artificial_delay, tokens = [], 0.0, {"input_tokens": 0, "output_tokens": 0, "api_crashed": False}
                    p4_time = 0.0
                
                extracted_payload = [d.model_dump() for d in detections if d.is_present]
                
                timing_info = {
                    "phase1_ingestion_seconds": 0.0,
                    "phase3_retrieval_seconds": round(p3_time, 2),
                    "phase4_inference_seconds": round(p4_time, 2),
                    "input_tokens": tokens.get("input_tokens", 0),
                    "output_tokens": tokens.get("output_tokens", 0),
                    "api_crashed": tokens.get("api_crashed", False)
                }
                    
            elif pipeline_type == "langgraph":
                t_start = time.time()
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
                        
                        p4_time = timing_ph3.get("consolidator_seconds", 0) + timing_ph3.get("validator_node_seconds", 0) + timing_ph3.get("extraction_oracle_node_seconds", 0) + timing_ph3.get("triage_node_seconds", 0)
                        timing_info = {
                            "phase1_ingestion_seconds": 0.0,
                            "phase3_retrieval_seconds": 0.0,
                            "phase4_inference_seconds": round(p4_time, 2),
                            "langgraph_internal_breakdown": timing_ph3,
                            "input_tokens": timing_ph3.get("input_tokens", 0),
                            "output_tokens": timing_ph3.get("output_tokens", 0),
                            "api_crashed": timing_ph3.get("api_crashed", False)
                        }
                        
                if pipeline_type == "langgraph" and timing_ph3.get("api_crashed"):
                    print("\n[!] CORTE DE LLM/API DETECTADO DURANTE ESTA SENTENCIA (LangGraph). Deteniendo ejecución.")
                    llm_crashed = True
                    break
                        
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
                "source_file": f"ctibench_doc_{idx}",
                "true_labels": true_lbls,
                "predicted_labels_raw": [r.get("technique_id") if isinstance(r, dict) else getattr(r, 'technique_id', str(r)) for r in extracted_payload],
                "predicted_labels_normalized": predicted_ids,
                "metrics": metrics,
                "traceability_summary": traceability_summary,
                "timing_breakdown": timing_info,
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
            detailed_results.append({"source_file": f"ctibench_doc_{idx}", "error": error_str})
            
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
