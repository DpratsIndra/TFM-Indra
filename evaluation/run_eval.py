import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm
import contextlib
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.data_loaders import TramDataLoader
from evaluation.metrics_calculator import Evaluator
from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from src.langchain_pipeline.phase4_inference import TTPAnalyzer
from src.langgraph_agents.graph_builder import process_full_report
from dotenv import load_dotenv

load_dotenv()

def get_langchain_engine():
    """Sets up the Qdrant connection and LLM for LangChain single-sentence inference."""
    qdrant_url = f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
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

def get_academic_sample(df: pd.DataFrame, total_size: int = 1000) -> pd.DataFrame:
    """
    Balancear el dataset TRAM artificialmente. Como el 90% de las oraciones en 
    reportes CTI no contienen TTPs, si evaluamos todo en crudo el accuracy sería engañoso.
    Forzamos un split ~50/50 entre oraciones positivas y negativas para obtener 
    métricas de F1-Score realistas y comparar con papers académicos.
    """
    df_pos = df[df['true_labels'].map(len) > 0]
    df_neg = df[df['true_labels'].map(len) == 0]
    
    half = total_size // 2
    
    # Sample positives and negatives safely
    sample_pos = df_pos.sample(n=min(half, len(df_pos)), random_state=42)
    sample_neg = df_neg.sample(n=min(total_size - len(sample_pos), len(df_neg)), random_state=42)
    
    # Combine and shuffle
    df_sampled = pd.concat([sample_pos, sample_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_sampled

def run_tram_evaluation(file_path: str, sample_size: int = None, pipeline_type: str = "langchain", resume_from: str = None):
    print(f"\n[*] Loading TRAM dataset from: {file_path}")
    loader = TramDataLoader()
    df = loader.load(file_path)
    
    if df.empty:
        print("[!] Error: DataFrame is empty. Check the JSON structure.")
        return
        
    # Academic Sampling Logic
    if sample_size and sample_size < 100:
        print(f"[*] Subsampling {sample_size} random records for quick testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        # Default academic evaluation size
        target_size = sample_size if sample_size else 200
        print(f"[*] Generating Academic Stratified Sample (N={target_size})...")
        df = get_academic_sample(df, total_size=target_size)
        
    total_records = len(df)
    print(f"[*] Dataset loaded: {total_records} sentences. Pipeline: {pipeline_type.upper()}\n")
    
    predicted_labels = []
    detailed_results = []
    hierarchy_stats = {"total_exact_matches": 0, "total_more_detailed": 0, "total_more_general": 0}
    
    if pipeline_type == "langchain":
        retriever, analyzer = get_langchain_engine()
        
    use_vlm = os.getenv("USE_VLM_EXTRACTION", "False").lower() in ("true", "1", "yes")
    use_rep = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag_vlm = "vlm_on" if use_vlm else "vlm_off"
    tag_rep = "rep_on" if use_rep else "rep_off"
    output_filename = f"tram_eval_{pipeline_type}_{tag_vlm}_{tag_rep}_{timestamp}.json"
    if resume_from and os.path.exists(resume_from):
        print(f"[*] Resuming from checkpoint: {resume_from}")
        output_path = resume_from
        with open(resume_from, "r", encoding="utf-8") as f:
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
    
    print("="*60)
    print(f"🚀 INITIATING INFERENCE LOOP ({pipeline_type.upper()})")
    print("="*60)
    
    start_time = time.time()
    
    start_idx_val = len(predicted_labels)
    df_remaining = df.iloc[start_idx_val:]
    
    for idx, row in tqdm(df_remaining.iterrows(), total=total_records, desc="Processing Sentences", initial=start_idx_val):
            
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
                predicted_ids = [d.technique_id for d in detections if d.is_present]
                    
            elif pipeline_type == "langgraph":
                sanitized_chunks = [{"text": text, "metadata": {"chunk_index": idx}}]
                global_context = "This is an isolated sentence from a CTI report."
                
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        result = process_full_report(
                            source_file=f"tram_sentence_{idx}",
                            global_context=global_context,
                            sanitized_chunks=sanitized_chunks
                        )
                        extracted = result.get("extracted_ttps", [])
                        predicted_ids = [ttp.get("technique_id") for ttp in extracted]
                        timing_ph3 = result.get("timing_breakdown_phase3", {})
                detections = [] # Mock for structural consistency if needed
                tokens = {
                    "input_tokens": timing_ph3.get("input_tokens", 0),
                    "output_tokens": timing_ph3.get("output_tokens", 0)
                }

            # Serializar las detecciones de LangChain (Pydantic objects a dict) si es necesario
            extracted_payload = []
            if pipeline_type == "langchain" and 'candidates' in locals() and candidates:
                if 'detections' in locals():
                    extracted_payload = [d.model_dump() for d in detections if d.is_present]
            elif pipeline_type == "langgraph" and 'extracted' in locals():
                extracted_payload = extracted
                
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
                "source_file": f"tram_sentence_{idx}",
                "text": text,
                "true_labels": true_lbls,
                "predicted_labels_raw": [r.get("technique_id") if isinstance(r, dict) else r.technique_id for r in extracted_payload],
                "predicted_labels_normalized": predicted_ids,
                "metrics": metrics,
                "traceability_summary": traceability_summary,
                "timing_breakdown": tokens,
                "extracted_ttps": extracted_payload
            })
            
            # Live Terminal Logging
            match_status = "✅ MATCH" if set(true_lbls) == set(predicted_ids) else "⚠️ DIFF"
            print(f"\n[Sentence {idx+1}/{total_records}] {match_status}")
            print(f"   True: {true_lbls}")
            print(f"   Pred: {predicted_ids}")
            
        except KeyboardInterrupt:
            print("\n[!] INTERRUPCIÓN MANUAL (Ctrl+C). Cancelando de forma segura y guardando los reportes completados...")
            llm_crashed = True
            break
        except Exception as e:
            error_str = str(e)
            print(f"\n[!] Error crítico procesando la sentencia {idx}: {error_str}")
            
            api_errors = ["429", "quota", "resourceexhausted", "503", "500", "timeout", "not_found", "api", "connection", "unavailable"]
            if any(err in error_str.lower() for err in api_errors):
                print("\n[!] CORTE DE LLM/API DETECTADO. Deteniendo ejecución para salvaguardar el progreso.")
                llm_crashed = True
                break
                
            predicted_ids = []
            detailed_results.append({"sentence_id": idx, "source_file": f"tram_sentence_{idx}", "error": error_str})

        predicted_labels.append(predicted_ids)
        
        time_to_sleep = 5.0
        
        # NUEVO: AUTO-GUARDADO (CHECKPOINT)
        current_session_minutes = (time.time() - start_time) / 60.0
        partial_output = {
            "status": f"INCOMPLETE - Processed {idx + 1}/{total_records}",
            "total_execution_minutes": round(previous_execution_minutes + current_session_minutes, 2),
            "hierarchy_analysis": hierarchy_stats,
            "detailed_executions": detailed_results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(partial_output, f, indent=4)
            
        time.sleep(time_to_sleep)
        
    if len(predicted_labels) < len(df):
        print(f"\n[!] Evaluadas solo {len(predicted_labels)} de {len(df)} sentencias debido a interrupción.")
        df = df.iloc[:len(predicted_labels)].copy()
        
    df['predicted_labels'] = predicted_labels
    
    if len(df) == 0:
        print("\n[!] No se evaluó ninguna sentencia. Saliendo.")
        sys.exit(429 if 'llm_crashed' in locals() else 0)
        
    print("\n[*] Calculating Metrics...")
    evaluator = Evaluator(df)
    results = evaluator.evaluate()
    
    current_session_minutes = (time.time() - start_time) / 60.0
    total_execution_minutes = round(previous_execution_minutes + current_session_minutes, 2)
    
    # Estructura JSON estandarizada para TRAM (coincide con CTIHAL)
    final_output = {
        "total_execution_minutes": total_execution_minutes,
        "global_metrics": results,
        "hierarchy_analysis": hierarchy_stats,
        "detailed_executions": detailed_results
    }
    
    # Save Main Results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print("\n" + "="*50)
    print(f"🎯 EVALUATION RESULTS (MICRO) - {pipeline_type.upper()}")
    print("="*50)
    print(f"Precision: {results['micro']['precision']}")
    print(f"Recall:    {results['micro']['recall']}")
    print(f"F1-Score:  {results['micro']['f1']}")
    print(f"F0.5-Score:{results['micro']['f0.5']}")
    print(f"\n[+] Full report saved to: {output_path}")
    if 'llm_crashed' in locals():
        sys.exit(429)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single-config evaluation on TRAM dataset")
    parser.add_argument("--dataset_path", type=str, default="data/eval_datasets/tram2/multi_label.json", help="Path to TRAM dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of sentences to evaluate (for quick testing)")
    parser.add_argument("--pipeline", type=str, choices=["langchain", "langgraph"], default="langchain", help="Which architecture to evaluate")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a previous JSON to resume execution")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"[!] Error: Dataset not found at {args.dataset_path}")
        sys.exit(1)
        
    run_tram_evaluation(args.dataset_path, sample_size=args.limit, pipeline_type=args.pipeline, resume_from=args.resume_from)
