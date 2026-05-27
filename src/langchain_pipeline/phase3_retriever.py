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

    def _get_initial_candidates(self, chunk: Document, top_k: int = 15) -> List[Document]:
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

    def get_filtered_mitre_candidates(self, report_chunks: List[Document], threshold: float = 0.45) -> Dict[str, Dict[str, Any]]:
        """
        Main orchestrator for the retrieval phase. Retrieves candidates for all chunks,
        reranks them in a single optimized batch, and aggregates the results.
        
        Args:
            report_chunks (List[Document]): Chunked documents from Phase 1.
            threshold (float): Minimum score for the Cross-Encoder.
            
        Returns:
            Dict[str, Dict[str, Any]]: Aggregated mapping of MITRE techniques to supporting chunks.
        """
        grouped_results: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Starting retrieval and reranking for {len(report_chunks)} chunks...")
        
        all_initial_candidates = []
        all_pairs = []
        
        # Step 1: Fast Vector Search (High Recall) for all chunks
        for chunk in report_chunks:
            chunk_text = chunk.page_content
            candidates = self._get_initial_candidates(chunk, top_k=15)
            all_initial_candidates.append((chunk, candidates))
            
            for doc in candidates:
                all_pairs.append([chunk_text, doc.page_content])
                
        # Step 2: Accurate Reranking in Batch (High Precision)
        logger.info(f"Reranking {len(all_pairs)} pairs in batch mode...")
        scores = self.reranker.predict(all_pairs) if all_pairs else []
        
        # Re-associate scores with chunks and apply filtering
        score_idx = 0
        for chunk, candidates in all_initial_candidates:
            chunk_text = chunk.page_content
            
            # Extract page number based on loader metadata (PyMuPDF uses 'page', Unstructured uses 'page_number')
            page_val = chunk.metadata.get("page")
            if page_val is not None:
                page_num = int(page_val) + 1  # PyMuPDF is 0-indexed
            else:
                page_num = chunk.metadata.get("page_number", "Unknown")
                
            chunk_idx = chunk.metadata.get("chunk_index", "Unknown")
            
            for doc in candidates:
                score = float(scores[score_idx])
                score_idx += 1
                
                if score >= threshold:
                    tech_id = doc.metadata.get("technique_id", "Unknown")
                    tech_name = doc.metadata.get("name", "Unknown")
                    tactics_str = doc.metadata.get("tactics", "")
                    
                    if tech_id not in grouped_results:
                        grouped_results[tech_id] = {
                            "name": tech_name,
                            "tactics": [t.strip() for t in tactics_str.split(",") if t.strip()],
                            "score": score,
                            "supporting_chunks": []
                        }
                    else:
                        if score > grouped_results[tech_id]["score"]:
                            grouped_results[tech_id]["score"] = score
                            
                    is_unique = all(c["text"] != chunk_text for c in grouped_results[tech_id]["supporting_chunks"])
                    if is_unique:
                        grouped_results[tech_id]["supporting_chunks"].append({
                            "text": chunk_text,
                            "location": f"Page {page_num}, Chunk {chunk_idx}",
                            "score": round(score, 4)
                        })
                        
        for tech_id in grouped_results:
            chunks = grouped_results[tech_id]["supporting_chunks"]
            chunks.sort(key=lambda x: x["score"], reverse=True)
            grouped_results[tech_id]["supporting_chunks"] = chunks
            
        logger.info(f"Retrieval complete. Found {len(grouped_results)} unique techniques across all chunks.")
        return grouped_results
