import os
import requests
import logging
import math
from typing import List, Dict, Any

import torch
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CandidateRetriever:
    """
    Handles Phase 3 of the LangChain pipeline: Hybrid Retrieval and Cross-Encoder Reranking.
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

        use_remote = os.getenv("EXECUTION_PROFILE", "LOCAL").upper() == "REMOTE"
        use_local_reranker = os.getenv("USE_LOCAL_RERANKER", "False").lower() in ("true", "1", "yes")
        
        if use_remote and not use_local_reranker:
            logger.info("Using remote reranker. Local CrossEncoder will not be loaded.")
            self.reranker = None
        else:
            logger.info(
                f"Loading Cross-Encoder model (BAAI/bge-reranker-base) on {self.device}..."
            )
            self.reranker = CrossEncoder("BAAI/bge-reranker-base", device=self.device)

    def _get_initial_candidates(
        self, chunk: Document, top_k: int = 25
    ) -> List[tuple]:
        """
        Retrieves the initial set of candidate MITRE techniques from Qdrant.

        Args:
            chunk (Document): A chunk of text from the parsed CTI report.
            top_k (int): The number of candidates to retrieve.

        Returns:
            List[tuple]: List of tuples containing (Document, score).
        """
        # Execute the vector similarity search
        candidates_with_score = self.vector_store.similarity_search_with_score(chunk.page_content, k=top_k)
        return candidates_with_score

    def get_filtered_mitre_candidates(
        self, report_chunks: List[Document], threshold: float = None
    ) -> List[Dict[str, Any]]:
        # Si no se provee por argumento, se lee de .env (por defecto 0.20 para ser permisivos)
        if threshold is None:
            threshold = float(os.getenv("RERANKER_THRESHOLD", "0.20"))

        logger.info(
            f"Starting retrieval and reranking for {len(report_chunks)} chunks (Threshold: {threshold})..."
        )

        all_initial_candidates = []

        for chunk in report_chunks:
            candidates = self._get_initial_candidates(chunk, top_k=25)
            all_initial_candidates.append((chunk, candidates))

        use_remote = os.getenv("EXECUTION_PROFILE", "LOCAL").upper() == "REMOTE"
        use_local_reranker = os.getenv("USE_LOCAL_RERANKER", "False").lower() in ("true", "1", "yes")
        reranker_url = os.getenv("RERANKER_URL")
        reranker_model = os.getenv(
            "RERANKER_MODEL_NAME", "jina-reranker-v2-base-multilingual"
        )

        if use_remote and not use_local_reranker and not reranker_url:
            raise ValueError(
                "RERANKER_URL must be defined in .env for REMOTE execution profile"
            )

        chunk_results = []
        for chunk, candidates_with_score in all_initial_candidates:
            valid_candidates = []
            candidates = [doc for doc, _ in candidates_with_score]
            qdrant_scores = [q_score for _, q_score in candidates_with_score]
            
            if use_remote and not use_local_reranker:
                # Le mandamos los candidatos en texto crudo a TEI/Jina para que haga un
                # cross-encoding real contra la query y nos devuelva un score ajustado.
                docs_text = [doc.page_content for doc in candidates]
                payload = {
                    "model": reranker_model,
                    "query": chunk.page_content,
                    "documents": docs_text,
                }
                try:
                    response = requests.post(
                        reranker_url,
                        json=payload,
                        headers={"Authorization": "Bearer EMPTY"},
                        timeout=60,
                    )
                    response.raise_for_status()
                    if response.status_code == 200:
                        res_data = response.json()
                        # Jina/TEI suele devolver results: [{"index": X, "relevance_score": Y}]
                        candidates_with_scores = []
                        for item in res_data.get("results", []):
                            idx = item["index"]
                            raw_score = float(item["relevance_score"])
                            # El endpoint TEI/Jina nos devuelve los logits en bruto (pueden ser negativos o > 1).
                            # Overflow-safe Sigmoid para forzarlos a rango [0, 1] y mantener compatibilidad.
                            if raw_score < -709:
                                r_score = 0.0
                            else:
                                r_score = 1 / (1 + math.exp(-raw_score))
                                
                            doc = candidates[idx]
                            q_score = qdrant_scores[idx]
                            final_score = (0.7 * float(r_score)) + (0.3 * float(q_score))
                            
                            candidates_with_scores.append(
                                {
                                    "technique_id": doc.metadata.get("technique_id", "Unknown"),
                                    "name": doc.metadata.get("name", "Unknown"),
                                    "tactics": [t.strip() for t in doc.metadata.get("tactics", "").split(",") if t.strip()],
                                    "description": doc.metadata.get("full_description", ""),
                                    "score": round(final_score, 4),
                                }
                            )

                        candidates_with_scores.sort(key=lambda x: x["score"], reverse=True)
                        valid_candidates = [c for c in candidates_with_scores if c["score"] >= threshold]

                        if len(valid_candidates) == 0 and len(candidates_with_scores) > 0:
                            logger.debug("Adaptive reranking activated. Taking top 3 ignoring threshold.")
                            valid_candidates = candidates_with_scores[:3]
                except Exception as e:
                    logger.error(f"Error calling remote reranker: {e}")
            else:
                # Fallback Local original
                pairs = [[chunk.page_content, doc.page_content] for doc in candidates]
                if pairs:
                    scores = self.reranker.predict(
                        pairs, batch_size=32, activation_fn=torch.nn.Sigmoid()
                    )
                    candidates_with_scores = []
                    for doc, r_score, q_score in zip(candidates, scores, qdrant_scores):
                        final_score = (0.7 * float(r_score)) + (0.3 * float(q_score))
                        candidates_with_scores.append(
                            {
                                "technique_id": doc.metadata.get("technique_id", "Unknown"),
                                "name": doc.metadata.get("name", "Unknown"),
                                "tactics": [t.strip() for t in doc.metadata.get("tactics", "").split(",") if t.strip()],
                                "description": doc.metadata.get("full_description", ""),
                                "score": round(final_score, 4),
                            }
                        )

                    candidates_with_scores.sort(key=lambda x: x["score"], reverse=True)
                    valid_candidates = [c for c in candidates_with_scores if c["score"] >= threshold]

                    if len(valid_candidates) == 0 and len(candidates_with_scores) > 0:
                        logger.debug("Adaptive reranking activated. Taking top 3 ignoring threshold.")
                        valid_candidates = candidates_with_scores[:3]

                if valid_candidates:
                    logger.debug(
                        f"Chunk {chunk.page_content[:30]}... Max valid score: {max(c['score'] for c in valid_candidates)}"
                    )
                else:
                    logger.debug(
                        f"Chunk {chunk.page_content[:30]}... 0 candidates passed threshold {threshold}. Max score was: {max(scores) if len(scores) > 0 else 'N/A'}"
                    )

            if valid_candidates:
                valid_candidates.sort(key=lambda x: x["score"], reverse=True)
                chunk_results.append({"chunk": chunk, "candidates": valid_candidates})

        logger.info(
            f"Retrieval complete. {len(chunk_results)} chunk inference tasks prepared."
        )
        return chunk_results
