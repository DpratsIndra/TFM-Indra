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

    def _rerank_and_filter(self, chunk_text: str, candidates: List[Document], threshold: float = 0.45) -> List[Document]:
        """
        Reranks the initial candidates using the Cross-Encoder and filters out
        those that score below the threshold.
        
        Args:
            chunk_text (str): The text from the CTI report chunk.
            candidates (List[Document]): The initial candidates from Qdrant.
            threshold (float): The minimum score to accept a candidate.
            
        Returns:
            List[Document]: The highly relevant candidates that passed the threshold.
        """
        if not candidates:
            return []
            
        # Build pairs of (Query Chunk, Candidate MITRE Description)
        pairs = [[chunk_text, doc.page_content] for doc in candidates]
        
        # Get relevance scores from the Cross-Encoder
        scores = self.reranker.predict(pairs)
        
        filtered_candidates = []
        for score, doc in zip(scores, candidates):
            if score >= threshold:
                # Store the score in metadata for reference or downstream logic
                doc.metadata['rerank_score'] = float(score)
                filtered_candidates.append(doc)
                
        # Optional: Sort the filtered candidates by score descending
        filtered_candidates.sort(key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
        return filtered_candidates

    def get_filtered_mitre_candidates(self, report_chunks: List[Document], threshold: float = 0.45) -> Dict[str, Dict[str, Any]]:
        """
        Main orchestrator for the retrieval phase. Iterates over all CTI chunks,
        fetches candidates, reranks them, and aggregates the results by technique_id.
        
        Args:
            report_chunks (List[Document]): The chunked documents from Phase 1.
            threshold (float): The threshold for the Cross-Encoder reranker.
            
        Returns:
            Dict[str, Dict[str, Any]]: An aggregated mapping of MITRE techniques to their supporting chunks.
            Example: {"T1059": {"name": "Command and Scripting", "supporting_chunks": ["chunk 1", "chunk 2"]}}
        """
        grouped_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Starting retrieval and reranking for {len(report_chunks)} chunks...")
        
        for chunk in report_chunks:
            chunk_text = chunk.page_content
            
            # Step 1: Fast Vector Search (High Recall)
            initial_candidates = self._get_initial_candidates(chunk, top_k=15)
            
            # Step 2: Accurate Reranking (High Precision)
            final_candidates = self._rerank_and_filter(chunk_text, initial_candidates, threshold=threshold)
            
            # Step 3: Logical Grouping
            for doc in final_candidates:
                tech_id = doc.metadata.get("technique_id", "Unknown")
                tech_name = doc.metadata.get("name", "Unknown")
                
                # Initialize the technique entry if it's the first time we see it
                if tech_id not in grouped_results:
                    grouped_results[tech_id] = {
                        "name": tech_name,
                        "supporting_chunks": []
                    }
                    
                # Append the chunk to the supporting chunks only if it's unique
                # This prevents redundant context from overloading the LLM in Phase 4
                if chunk_text not in grouped_results[tech_id]["supporting_chunks"]:
                    grouped_results[tech_id]["supporting_chunks"].append(chunk_text)
                    
        logger.info(f"Retrieval complete. Found {len(grouped_results)} unique techniques across all chunks.")
        return grouped_results
