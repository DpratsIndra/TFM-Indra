import logging
from typing import List, Dict, Any

import torch
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CandidateRetriever:
    """
    Handles Phase 3 of the LangChain pipeline: Hybrid Retrieval and Cross-Encoder Reranking.
    Acts as a high-precision funnel using the 'Retrieve-then-Rerank' pattern to reduce
    the load and hallucinations in the subsequent LLM inference phase.
    """

    def __init__(self, vector_store: QdrantVectorStore, device: str = None) -> None:
        """
        Initializes the retriever with a vector database and a Cross-Encoder model.
        
        Args:
            vector_store (QdrantVectorStore): The initialized Qdrant vector database from Phase 2.
            device (str, optional): The device to run the Cross-Encoder on ('cuda', 'mps', 'cpu').
                                    Auto-detects if None is provided.
        """
        self.vector_store = vector_store
        
        if device is None:
            # Auto-detect the optimal device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Cross-Encoder model (BAAI/bge-reranker-base) on {self.device}...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base', device=self.device)

    def _get_initial_candidates(self, chunk: Document, top_k: int = 35) -> List[Document]:
        """
        Retrieves the initial set of candidate MITRE techniques from Qdrant.
        
        Args:
            chunk (Document): A chunk of text from the parsed CTI report.
            top_k (int): The number of candidates to retrieve.
            
        Returns:
            List[Document]: The top K MITRE techniques retrieved via semantic search.
        """
        # Execute the vector similarity search
        candidates = self.vector_store.similarity_search(chunk.page_content, k=top_k)
        return candidates

    def get_filtered_mitre_candidates(self, report_chunks: List[Document], threshold: float = 0.50) -> List[Dict[str, Any]]:
        logger.info(f"Starting retrieval and reranking for {len(report_chunks)} chunks...")
        
        all_initial_candidates = []
        all_pairs = []
        
        for chunk in report_chunks:
            chunk_text = chunk.page_content
            candidates = self._get_initial_candidates(chunk, top_k=15)
            all_initial_candidates.append((chunk, candidates))
            for doc in candidates:
                all_pairs.append([chunk_text, doc.page_content])
                
        logger.info(f"Reranking {len(all_pairs)} pairs in batch mode...")
        scores = self.reranker.predict(all_pairs, batch_size=32, activation_fn=torch.nn.Sigmoid()) if all_pairs else []
        
        chunk_results = []
        score_idx = 0
        for chunk, candidates in all_initial_candidates:
            valid_candidates = []
            for doc in candidates:
                score = float(scores[score_idx])
                score_idx += 1
                if score >= threshold:
                    valid_candidates.append({
                        "technique_id": doc.metadata.get("technique_id", "Unknown"),
                        "name": doc.metadata.get("name", "Unknown"),
                        "tactics": [t.strip() for t in doc.metadata.get("tactics", "").split(",") if t.strip()],
                        "description": doc.metadata.get("full_description", ""),
                        "score": round(score, 4)
                    })
            
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x["score"], reverse=True)
                for i in range(0, len(valid_candidates), 10):
                    chunk_results.append({
                        "chunk": chunk,
                        "candidates": valid_candidates[i:i+10]
                    })
                
        logger.info(f"Retrieval complete. {len(chunk_results)} chunk inference tasks prepared.")
        return chunk_results
