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
    args = parser.parse_args()

    pipeline_type = args.pipeline

    print("\n" + "="*80)
    print("🚀 INICIANDO EVALUACIÓN CTIBENCH (TEXT-LEVEL)")
    print(f"   Pipeline:           {pipeline_type.upper()}")
    print("="*80)

    print(f"\n[*] Loading CTIBench dataset from: {args.dataset_path}")
    loader = CtibenchDataLoader(base_path=args.dataset_path)
    df = loader.load()
    
    if df.empty:
        print("[!] Error: No data found. Ensure cti-ate.tsv exists.")
        sys.exit(1)
        
    if args.limit:
        print(f"[*] Subsampling {args.limit} random records for testing...")
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        
    print(f"[*] Total sentences for evaluation: {len(df)}")
    
    if pipeline_type == "langchain":
        retriever, analyzer = setup_langchain_components()
        
    predicted_labels = []
    detailed_results = []
    hierarchy_stats = {"total_exact_matches": 0, "total_more_detailed": 0, "total_more_general": 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval: {pipeline_type}"):
        text = row['text']
        true_lbls = row['true_labels']
        predicted_ids = []
        
        try:
            if pipeline_type == "langchain":
                doc = Document(page_content=text, metadata={"chunk_index": idx, "page_number": 1})
                candidates = retriever.get_filtered_mitre_candidates([doc], threshold=0.2)
                if candidates:
                    detections, _ = analyzer.analyze_candidates(candidates)
                    predicted_ids = [d.technique_id for d in detections if d.is_present]
                    extracted_payload = [d.model_dump() for d in detections if d.is_present]
                else:
                    extracted_payload = []
                    
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
            
            detailed_results.append({
                "sentence_id": idx,
                "text": text,
                "true_labels": true_lbls,
                "predicted_labels_raw": [r.get("technique_id") for r in extracted_payload if "technique_id" in r],
                "predicted_labels_normalized": predicted_ids,
                "metrics": {"TP": len(tp), "FP": len(fp), "FN": len(fn)},
                "traceability_summary": traceability_summary,
                "extracted_ttps": extracted_payload
            })
            
        except Exception as e:
            tqdm.write(f" [!] ERROR crítico en id {idx}: {e}")
            predicted_ids = []
            detailed_results.append({"sentence_id": idx, "error": str(e)})
            
        predicted_labels.append(predicted_ids)
        time.sleep(0.5)  # Breve pausa para no saturar APIs
        
    df['predicted_labels'] = predicted_labels
    evaluator = Evaluator(df)
    results = evaluator.evaluate()
    
    final_output = {
        "global_metrics": results,
        "hierarchy_analysis": hierarchy_stats,
        "detailed_executions": detailed_results
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"ctibench_eval_{pipeline_type}_{timestamp}.json"
    output_path = os.path.join("data", "output", "evaluations", output_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

if __name__ == "__main__":
    run_evaluation()
