import os
import sys
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.data_loaders import AptReportDataLoader
from evaluation.metrics_calculator import Evaluator
from src.langchain_pipeline.main_pipeline import run_cti_extraction
from src.langgraph_agents.main_langgraph import run_langgraph_extraction
from dotenv import load_dotenv

# Cargar variables de entorno (USE_VLM_EXTRACTION, USE_PROMPT_REPETITION, etc.)
load_dotenv()

def run_evaluation():
    """
    Ejecutar una evaluación End-to-End sobre el dataset APT_REPORT.
    Se utiliza el nombre de la carpeta (ej. APT28) para obtener las técnicas de MITRE como Ground Truth.
    """
    parser = argparse.ArgumentParser(description="Run single-config E2E evaluation on APT_REPORT dataset")
    parser.add_argument("--dataset_path", type=str, default="data/eval_datasets/APT_REPORT", help="Path to APT_REPORT dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to evaluate (for quick testing)")
    parser.add_argument("--pipeline", type=str, choices=["langchain", "langgraph"], default="langgraph", help="Which architecture to evaluate")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a previous JSON to resume execution")
    args = parser.parse_args()

    # Leer configuración desde las variables de entorno
    vlm_mode = os.getenv("USE_VLM_EXTRACTION", "False")
    rep_mode = os.getenv("USE_PROMPT_REPETITION", "False")
    pipeline = args.pipeline

    print("\n" + "="*80)
    print("🚀 INICIANDO EVALUACIÓN APT_REPORT (CROSS-REFERENCE WITH MITRE GROUPS)")
    print(f"   Pipeline:           {pipeline.upper()}")
    print(f"   VLM Extraction:     {vlm_mode}")
    print(f"   Prompt Repetition:  {rep_mode}")
    
    profile = os.getenv("EXECUTION_PROFILE", "LOCAL").upper()
    if profile == "REMOTE":
        use_gemma = os.getenv("USE_GEMMA4", "False").lower() in ("true", "1", "yes")
        model_used = os.getenv("VLLM_MODEL_NAME_GEMMA", "gemma4") if use_gemma else os.getenv("VLLM_MODEL_NAME", "gpt-oss-20b")
    else:
        model_used = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    
    print(f"   Model Used:         {model_used} ({profile})")
    print("="*80)

    print(f"\n[*] Loading APT_REPORT dataset from: {args.dataset_path}")
    loader = AptReportDataLoader(base_path=args.dataset_path)
    df = loader.load()
    
    if df.empty:
        print("[!] Error: No PDFs/MDs matched. Ensure mitre_group_mapping.json exists and dataset has PDFs.")
        sys.exit(1)
        
    limit = args.limit if args.limit else 10
    print(f"[*] Subsampling {limit} random reports for testing...")
    df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        
    print(f"[*] Successfully paired {len(df)} PDF raw reports with Ground Truth.")
    
    predicted_labels = []
    detailed_results = []
    hierarchy_stats = {"total_exact_matches": 0, "total_more_detailed": 0, "total_more_general": 0}
    
    # Definir ruta de salida antes del bucle para auto-guardado
    tag_vlm = "vlm_on" if str(vlm_mode).lower() in ["true", "1", "yes"] else "vlm_off"
    tag_rep = "rep_on" if str(rep_mode).lower() in ["true", "1", "yes"] else "rep_off"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    output_filename = f"apt_eval_{pipeline}_{model_used}_{tag_vlm}_{tag_rep}_{timestamp}.json"
    
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
            print(f"[*] Resumed {start_idx} processed reports. Continuing from index {start_idx}...")
    else:
        output_path = os.path.join("data", "output", "evaluations", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_idx_val = len(predicted_labels)
    
    df_remaining = df.iloc[start_idx_val:]
    # Bucle de inferencia E2E
    for idx, row in tqdm(df_remaining.iterrows(), total=len(df), desc=f"Eval: {pipeline}", initial=start_idx_val):
            
        pdf_path = row['source_file']
        true_lbls = row['true_labels']
        group_name = row.get('group_name', 'Unknown')
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
            
            metrics = {"TP": len(tp), "FP": len(fp), "FN": len(fn)}
            if len(true_lbls) == 0 and len(predicted_ids) == 0:
                metrics["TN"] = 1
            
            if timing.get("api_crashed"):
                print("\n[!] CORTE DE LLM/API DETECTADO DURANTE ESTE REPORTE. Guardando extracción parcial y deteniendo ejecución.")
                print("    El documento se reanudará de forma automática gracias al Caché de Chunks en la próxima ejecución.")
                llm_crashed = True
                break
                
            detailed_results.append({
                "source_file": pdf_path,
                "group_name": group_name,
                "true_labels": true_lbls,
                "predicted_labels_raw": [r.get("technique_id") for r in results_list if "technique_id" in r],
                "predicted_labels_normalized": predicted_ids,
                "metrics": metrics,
                "traceability_summary": traceability_summary,
                "timing_breakdown": timing,
                "extracted_ttps": results_list
            })
            
            tqdm.write(f" 📄 [{group_name}] {pdf_name} -> TP: {len(tp)} | FP: {len(fp)} | FN: {len(fn)}")
        except KeyboardInterrupt:
            print("\n[!] INTERRUPCIÓN MANUAL (Ctrl+C). Cancelando de forma segura y guardando los reportes completados...")
            llm_crashed = True
            break
        except Exception as e:
            error_str = str(e)
            tqdm.write(f" [!] ERROR crítico en {pdf_name}: {error_str}")
            
            api_errors = ["429", "quota", "resourceexhausted", "503", "500", "timeout", "not_found", "api", "connection", "unavailable"]
            if any(err in error_str.lower() for err in api_errors):
                print("\n[!] CORTE DE LLM/API DETECTADO. Deteniendo ejecución para salvaguardar el progreso.")
                llm_crashed = True
                break
                
            predicted_ids = []
            detailed_results.append({"source_file": pdf_path, "error": error_str})
        
        predicted_labels.append(predicted_ids)
        
        # Calcular tiempo total acumulado hasta ahora
        total_seconds_partial = sum(
            sum(v for k, v in d.get("timing_breakdown", {}).items() if isinstance(v, (int, float))) 
            for d in detailed_results if "timing_breakdown" in d
        )
        
        # NUEVO: AUTO-GUARDADO (CHECKPOINT)
        partial_output = {
            "status": f"INCOMPLETE - Processed {idx + 1}/{len(df)}",
            "model_used": model_used,
            "total_execution_minutes": round(total_seconds_partial / 60.0, 2),
            "hierarchy_analysis": hierarchy_stats,
            "detailed_executions": detailed_results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(partial_output, f, indent=4)
        
        # Cuidar el Rate Limit del Free Tier entre documentos pesados
        time.sleep(2)
        
        # NUEVO: Forzar limpieza de RAM para evitar OOM Killer con Docling
        import gc
        gc.collect()
        
    if len(predicted_labels) < len(df):
        print(f"\n[!] Evaluados solo {len(predicted_labels)} de {len(df)} reportes debido a interrupción.")
        df = df.iloc[:len(predicted_labels)].copy()
        
    if len(df) == 0:
        print("\n[!] No se evaluó ningún reporte por completo. Creando JSON base para permitir reanudación.")
        results = {"micro": {"precision": 0, "recall": 0, "f1": 0, "f0.5": 0}}
    else:
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
        "model_used": model_used,
        "total_execution_minutes": round(total_seconds / 60.0, 2),
        "global_metrics": results,
        "hierarchy_analysis": hierarchy_stats,
        "detailed_executions": detailed_results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print("\n" + "="*80)
    print(f"🎯 RESULTADOS MICRO PARA APT_REPORT ({pipeline.upper()})")
    print("="*80)
    print(f"F0.5-Score: {results['micro']['f0.5']}")
    print(f"F1-Score:   {results['micro']['f1']}")
    print(f"Precision:  {results['micro']['precision']}")
    print(f"Recall:     {results['micro']['recall']}")
    print(f"\n[+] Full report saved to: {output_path}")
    if 'llm_crashed' in locals():
        sys.exit(429)

if __name__ == "__main__":
    run_evaluation()
