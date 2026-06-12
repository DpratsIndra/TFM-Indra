import os
import sys
import argparse
import logging
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath("."))
from evaluation.data_loaders import TramDataLoader
from src.langgraph_agents.tools import get_retriever

logging.basicConfig(level=logging.INFO, format="%(message)s")

def evaluate_embeddings(use_reranker=False, qdrant_k=100, rerank_k=25):
    print("="*60)
    print(f"TRAM EVALUATION")
    print(f"Mode: {'Qdrant + Reranker' if use_reranker else 'Qdrant Only'}")
    print("="*60)

    # Load TRAM dataset
    loader = TramDataLoader()
    tram_file = "data/eval_datasets/TRAM/multi_label.json"
    
    if not os.path.exists(tram_file):
        print(f"[ERROR] TRAM dataset not found at {tram_file}")
        return

    df = loader.load(tram_file)
    print(f"[*] Loaded {len(df)} labeled sentences from TRAM.")

    retriever = get_retriever()
    if not retriever:
        print("[ERROR] Could not connect to Qdrant.")
        return

    if use_reranker and not getattr(retriever, 'reranker', None):
        print("[WARNING] Reranker is disabled or not loaded. Falling back to Qdrant only.")
        use_reranker = False

    # Metrics
    if use_reranker:
        top_k_thresholds = [1, 5, 10, 25] # No podemos evaluar > 25 si solo rerankeamos 25
    else:
        top_k_thresholds = [1, 5, 10, 25, 50, 100]
        
    hits_at_k = {k: 0 for k in top_k_thresholds}
    labels_found_at_k = {k: 0 for k in top_k_thresholds}
    total_samples = 0
    total_true_labels = 0

    print(f"[*] Running queries... (This may take a while{' on CPU' if use_reranker else ''})")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        true_labels = row['true_labels']
        
        if not true_labels:
            continue
            
        total_samples += 1
        total_true_labels += len(true_labels)

        # 1. QDRANT RETRIEVAL
        candidates_with_score = retriever.vector_store.similarity_search_with_score(text, k=qdrant_k)
        
        if use_reranker:
            # Tomamos solo los top 25 de Qdrant para pasarlos por el Reranker (igual que en producción)
            candidates_to_rerank = candidates_with_score[:rerank_k]
            candidates = [doc for doc, _ in candidates_to_rerank]
            qdrant_scores = [q_score for _, q_score in candidates_to_rerank]
            
            pairs = [[text, doc.page_content] for doc in candidates]
            if pairs:
                scores = retriever.reranker.predict(pairs, batch_size=32, activation_fn=torch.nn.Sigmoid())
                
                combined = []
                for doc, r_score, q_score in zip(candidates, scores, qdrant_scores):
                    # Usamos la misma fórmula 70/30 que tienes en producción
                    final_score = (0.7 * float(r_score)) + (0.3 * float(q_score))
                    tech_id = doc.metadata.get("technique_id", "Unknown").upper()
                    combined.append((tech_id, final_score))
                
                combined.sort(key=lambda x: x[1], reverse=True)
                retrieved_ids = [tech_id for tech_id, _ in combined]
            else:
                retrieved_ids = []
        else:
            retrieved_ids = [doc.metadata.get("technique_id", "Unknown").upper() for doc, _ in candidates_with_score]

        # Evaluate Recall@K
        for k in top_k_thresholds:
            top_k_retrieved = set(retrieved_ids[:k])
            found_labels = [label for label in true_labels if label in top_k_retrieved]
            labels_found_at_k[k] += len(found_labels)
            if len(found_labels) > 0:
                hits_at_k[k] += 1

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {'QDRANT + RERANKER (Top 25)' if use_reranker else 'QDRANT ONLY (Top 100)'}")
    print("="*60)
    print(f"Total Sentences Evaluated: {total_samples}")
    print(f"Total True Labels to Find: {total_true_labels}\n")

    for k in top_k_thresholds:
        hit_rate = (hits_at_k[k] / total_samples) * 100
        recall = (labels_found_at_k[k] / total_true_labels) * 100
        print(f"Top-{k:<3} | Hit Rate (Sentence): {hit_rate:>5.2f}% | Recall (Labels): {recall:>5.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Embeddings on TRAM")
    parser.add_argument("--use-reranker", action="store_true", help="Enable Cross-Encoder Reranking")
    args = parser.parse_args()
    
    evaluate_embeddings(use_reranker=args.use_reranker)
