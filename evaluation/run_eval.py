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
from langchain_google_genai import ChatGoogleGenerativeAI
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
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"), temperature=0.0)
    analyzer = TTPAnalyzer(llm=llm)
    
    return retriever, analyzer

def get_academic_sample(df: pd.DataFrame, total_size: int = 1000) -> pd.DataFrame:
    """
    Creates a balanced sample to avoid evaluating thousands of empty sentences.
    Attempts a 50/50 split between sentences with TTPs and sentences without.
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

def run_tram_evaluation(file_path: str, sample_size: int = None, pipeline_type: str = "langchain"):
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
        target_size = sample_size if sample_size else 1000
        print(f"[*] Generating Academic Stratified Sample (N={target_size})...")
        df = get_academic_sample(df, total_size=target_size)
        
    total_records = len(df)
    print(f"[*] Dataset loaded: {total_records} sentences. Pipeline: {pipeline_type.upper()}\n")
    
    predicted_labels = []
    detailed_results = []
    
    if pipeline_type == "langchain":
        retriever, analyzer = get_langchain_engine()
    
    print("="*60)
    print(f"🚀 INITIATING INFERENCE LOOP ({pipeline_type.upper()})")
    print("="*60)
    
    for idx, row in tqdm(df.iterrows(), total=total_records, desc="Processing Sentences"):
        text = row['text']
        true_lbls = row['true_labels']
        predicted_ids = []
        
        if pipeline_type == "langchain":
            doc = Document(page_content=text, metadata={"chunk_index": idx, "page_number": 1})
            candidates = retriever.get_filtered_mitre_candidates([doc], threshold=0.8)
            if candidates:
                detections, artificial_delay = analyzer.analyze_candidates(candidates)
                predicted_ids = [d.technique_id for d in detections if d.is_present]
                
        elif pipeline_type == "langgraph":
            sanitized_chunks = [{"text": text, "metadata": {"chunk_index": idx}}]
            global_context = "This is an isolated sentence from a CTI report."
            
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    try:
                        result = process_full_report(
                            source_file=f"tram_sentence_{idx}",
                            global_context=global_context,
                            sanitized_chunks=sanitized_chunks
                        )
                        extracted = result.get("extracted_ttps", [])
                        predicted_ids = [ttp.get("technique_id") for ttp in extracted]
                    except Exception as e:
                        predicted_ids = []

        predicted_ids = list(set(predicted_ids))
        predicted_labels.append(predicted_ids)
        
        tp = list(set(true_lbls) & set(predicted_ids))
        fp = list(set(predicted_ids) - set(true_lbls))
        fn = list(set(true_lbls) - set(predicted_ids))
        
        # Serializar las detecciones de LangChain (Pydantic objects a dict) si es necesario
        extracted_payload = []
        if pipeline_type == "langchain" and 'candidates' in locals() and candidates:
            if 'detections' in locals():
                extracted_payload = [d.model_dump() for d in detections if d.is_present]
        elif pipeline_type == "langgraph" and 'extracted' in locals():
            extracted_payload = extracted

        detailed_results.append({
            "sentence_id": idx,
            "text": text,
            "true_labels": true_lbls,
            "predicted_labels": predicted_ids,
            "false_positives": fp,
            "false_negatives": fn,
            "extracted_ttps": extracted_payload
        })
        
        # Live Terminal Logging
        match_status = "✅ MATCH" if set(true_lbls) == set(predicted_ids) else "⚠️ DIFF"
        print(f"\n[Sentence {idx+1}/{total_records}] {match_status}")
        print(f"   True: {true_lbls}")
        print(f"   Pred: {predicted_ids}")
        
        time_to_sleep = 4.5 if pipeline_type == "langchain" else 2.0
        time.sleep(time_to_sleep)
        
    df['predicted_labels'] = predicted_labels
    
    print("\n[*] Calculating Metrics...")
    evaluator = Evaluator(df)
    results = evaluator.evaluate()
    results["detailed_report"] = detailed_results
    
    # ---------------------------------------------------------
    # Extract DIFFs for Qualitative Error Analysis (TFM)
    # ---------------------------------------------------------
    diffs = []
    for idx, row in df.iterrows():
        true_set = set(row['true_labels'])
        pred_set = set(row['predicted_labels'])
        
        if true_set != pred_set:
            diffs.append({
                "sentence_id": idx,
                "text": row['text'],
                "human_ground_truth": list(true_set),
                "llm_prediction": list(pred_set),
                "false_negatives_missed_by_llm": list(true_set - pred_set),
                "false_positives_hallucinated_by_llm": list(pred_set - true_set)
            })
    
    use_vlm = os.getenv("USE_VLM_EXTRACTION", "False").lower() in ("true", "1", "yes")
    use_rep = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag_vlm = "vlm_on" if use_vlm else "vlm_off"
    tag_rep = "rep_on" if use_rep else "rep_off"
    
    # Save Main Results
    output_filename = f"tram_eval_{pipeline_type}_{tag_vlm}_{tag_rep}_{timestamp}.json"
    output_path = os.path.join("data", "output", "evaluations", output_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Save DIFFs
    diff_filename = f"diff_analysis_{pipeline_type}_{tag_vlm}_{tag_rep}_{timestamp}.json"
    diff_path = os.path.join("data", "output", "evaluations", diff_filename)
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=4)
        
    print("\n" + "="*50)
    print(f"🎯 EVALUATION RESULTS (MICRO) - {pipeline_type.upper()}")
    print("="*50)
    print(f"Precision: {results['micro']['precision']}")
    print(f"Recall:    {results['micro']['recall']}")
    print(f"F1-Score:  {results['micro']['f1']}")
    print(f"F0.5-Score:{results['micro']['f0.5']}")
    print(f"\n[+] Full report saved to: {output_path}")
    print(f"[+] Saved {len(diffs)} conflict cases for manual review in: {diff_path}")

if __name__ == "__main__":
    dataset_path = "data/eval_datasets/tram2/multi_label.json"
    
    if not os.path.exists(dataset_path):
        print(f"[!] Error: Dataset not found at {dataset_path}")
        sys.exit(1)
        
    # Elige aquí qué pipeline evaluar: "langchain" o "langgraph"
    # Cambia el sample_size para pruebas rápidas
    run_tram_evaluation(dataset_path, sample_size=None, pipeline_type="langchain")
