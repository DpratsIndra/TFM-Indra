import os
import sys
import json
import time
import argparse
import contextlib
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.data_loaders import CtiHalDataLoader
from evaluation.metrics_calculator import Evaluator
from src.langchain_pipeline.main_pipeline import run_cti_extraction
from src.langgraph_agents.main_langgraph import run_langgraph_extraction
from dotenv import load_dotenv

# Cargar variables de entorno (USE_VLM_EXTRACTION, USE_PROMPT_REPETITION, etc.)
load_dotenv(override=True)

def run_evaluation():
    """
    Objetivo: Ejecutar una evaluación End-to-End (E2E) sobre el dataset CTI-HAL.
    A diferencia de TRAM (que evalúa oraciones sueltas), este script procesa PDFs enteros,
    lanzando el pipeline completo (ingesta, vector store, inferencia) y comparando
    las técnicas extraídas contra los ficheros markdown (ground truth) reales.
    """
    parser = argparse.ArgumentParser(description="Run single-config E2E evaluation on CTI-HAL dataset")
    parser.add_argument("--dataset_path", type=str, default="data/eval_datasets/ctihal", help="Path to CTI-HAL dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to evaluate (for quick testing)")
    parser.add_argument("--pipeline", type=str, choices=["langchain", "langgraph"], default="langchain", help="Which architecture to evaluate")
    args = parser.parse_args()

    # Leer configuración desde las variables de entorno
    vlm_mode = os.getenv("USE_VLM_EXTRACTION", "False")
    rep_mode = os.getenv("USE_PROMPT_REPETITION", "False")
    pipeline = args.pipeline

    print("\n" + "="*80)
    print(f"🚀 INICIANDO EVALUACIÓN CTI-HAL")
    print(f"   Pipeline:           {pipeline.upper()}")
    print(f"   VLM Extraction:     {vlm_mode}")
    print(f"   Prompt Repetition:  {rep_mode}")
    print("="*80)

    print(f"\n[*] Loading CTI-HAL dataset from: {args.dataset_path}")
    loader = CtiHalDataLoader(base_path=args.dataset_path)
    df = loader.load()
    
    if df.empty:
        print("[!] Error: No PDFs/MDs matched. Check folder structure (ensure 'reports' and 'annotations/.../annotator L' exist).")
        sys.exit(1)
        
    if args.limit:
        print(f"[*] Subsampling {args.limit} random reports for testing...")
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        
    print(f"[*] Successfully paired {len(df)} PDF raw reports with Ground Truth.")
    
    predicted_labels = []
    detailed_results = []
    hierarchy_stats = {"total_exact_matches": 0, "total_more_detailed": 0, "total_more_general": 0}
    
    # Bucle de inferencia E2E
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval: {pipeline}"):
        pdf_path = row['source_file']
        true_lbls = row['true_labels']
        predicted_ids = []
        pdf_name = os.path.basename(pdf_path)
        
        try:
            # Ejecución limpia: Dejamos que los logs de Fase pasen (INFO/ERROR)
            if pipeline == "langchain":
                results_list, timing = run_cti_extraction(pdf_path)
            else:
                results_list, timing = run_langgraph_extraction(pdf_path)
                        
            # Normalización y Trazabilidad Jerárquica
            normalized_preds = []
            traceability_summary = {"exact": 0, "more_detailed": 0, "more_general": 0}
            
            for result_ttp in results_list:
                pred_id = result_ttp.get("technique_id")
                if not pred_id:
                    continue
                    
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
            
            detailed_results.append({
                "source_file": pdf_path,
                "true_labels": true_lbls,
                "predicted_labels_raw": [r.get("technique_id") for r in results_list if "technique_id" in r],
                "predicted_labels_normalized": predicted_ids,
                "metrics": {"TP": len(tp), "FP": len(fp), "FN": len(fn)},
                "traceability_summary": traceability_summary,
                "timing_breakdown": timing,
                "extracted_ttps": results_list
            })
            
            tqdm.write(f" 📄 {pdf_name} -> TP: {len(tp)} | FP: {len(fp)} | FN: {len(fn)}")
            
        except Exception as e:
            tqdm.write(f" [!] ERROR crítico en {pdf_name}: {e}")
            sys.exit(1)
        
        predicted_labels.append(predicted_ids)
        
        # Cuidar el Rate Limit del Free Tier entre documentos pesados
        time.sleep(10)
        
    # Evaluar y exportar
    df['predicted_labels'] = predicted_labels
    evaluator = Evaluator(df)
    results = evaluator.evaluate()
    
    # Calcular tiempo total de esta config
    total_seconds = sum(
        sum(v for k, v in d.get("timing_breakdown", {}).items() if isinstance(v, (int, float))) 
        for d in detailed_results if "timing_breakdown" in d
    )
    
    # Estructura JSON estandarizada
    final_output = {
        "total_execution_minutes": round(total_seconds / 60.0, 2),
        "global_metrics": results,
        "hierarchy_analysis": hierarchy_stats,
        "detailed_executions": detailed_results
    }
    
    # Nombres de archivo dinámicos basados en la config actual
    tag_vlm = "vlm_on" if str(vlm_mode).lower() in ["true", "1", "yes"] else "vlm_off"
    tag_rep = "rep_on" if str(rep_mode).lower() in ["true", "1", "yes"] else "rep_off"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    output_filename = f"ctihal_eval_{pipeline}_{tag_vlm}_{tag_rep}_{timestamp}.json"
    output_path = os.path.join("data", "output", "evaluations", output_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print("\n" + "="*80)
    print(f"🎯 RESULTADOS MICRO PARA {pipeline.upper()}")
    print("="*80)
    print(f"F0.5-Score: {results['micro']['f0.5']}")
    print(f"F1-Score:   {results['micro']['f1']}")
    print(f"Precision:  {results['micro']['precision']}")
    print(f"Recall:     {results['micro']['recall']}")
    print(f"\n💾 Resumen completo guardado en: {output_path}")

if __name__ == "__main__":
    run_evaluation()
