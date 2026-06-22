import os
import json
import math
import torch
import requests
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from src.langchain_pipeline.phase3_retriever import CandidateRetriever
from src.core.embedding_factory import get_embeddings


# ==============================================================================
# GLOBALS & CACHING
# ==============================================================================


@lru_cache(maxsize=1)
def get_retriever() -> CandidateRetriever:
    """
    Initializes and caches the Qdrant VectorStore and the CrossEncoder Retriever.
    This prevents reloading models and creating new DB connections on every tool call.
    """
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        collection_name = "mitre_attack"

        # Connect to Qdrant
        client = QdrantClient(url=qdrant_url)
        # Validate connection
        client.get_collections()

        # Initialize Embeddings
        # Initialize Embeddings
        embeddings = get_embeddings()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Create LangChain VectorStore
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        return CandidateRetriever(vector_store=vector_store)

    except Exception as e:
        print(f"[ERROR] Tool Initialization Error (Qdrant): {e}")
        return None


@lru_cache(maxsize=1)
def load_mitre_json() -> dict:
    """
    Loads the raw MITRE STIX JSON file and builds a dictionary mapping
    Technique IDs (e.g., 'T1059.001') to their official descriptions,
    enriched with known tools/malware that use the technique.
    """
    mitre_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../data/mitre_data/enterprise-attack.json"
        )
    )

    if not os.path.exists(mitre_file):
        print(
            f"[ERROR] Tool Initialization Error: MITRE JSON not found at {mitre_file}"
        )
        return {}

    try:
        with open(mitre_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", []) if isinstance(data, dict) else data
        if not isinstance(objects, list):
            return {}

        entity_names = {}
        for obj in objects:
            if obj.get("type") in ["malware", "tool", "intrusion-set"]:
                entity_names[obj.get("id")] = obj.get("name")

        technique_tools = {}
        for obj in objects:
            if (
                obj.get("type") == "relationship"
                and obj.get("relationship_type") == "uses"
            ):
                source_id = obj.get("source_ref")
                target_id = obj.get("target_ref")

                if (
                    target_id
                    and target_id.startswith("attack-pattern--")
                    and source_id in entity_names
                ):
                    if target_id not in technique_tools:
                        technique_tools[target_id] = []
                    tool_name = entity_names[source_id]
                    if tool_name not in technique_tools[target_id]:
                        technique_tools[target_id].append(tool_name)

        lookup_dict = {}
        for obj in objects:
            if obj.get("type") != "attack-pattern":
                continue

            external_refs = obj.get("external_references", [])
            tech_id = None
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    tech_id = ref.get("external_id")
                    break

            if tech_id:
                base_desc = obj.get("description", "No description available.")
                stix_id = obj.get("id")
                tools_used = technique_tools.get(stix_id, [])

                if tools_used:
                    enriched_desc = f"{base_desc}\n\nKNOWN ASSOCIATED TOOLS/MALWARE: {', '.join(tools_used)}"
                else:
                    enriched_desc = base_desc

                lookup_dict[tech_id.upper()] = {
                    "description": enriched_desc,
                    "name": obj.get("name", "Unknown"),
                    "tactics": [
                        kc.get("phase_name") for kc in obj.get("kill_chain_phases", [])
                    ],
                }

        return lookup_dict
    except Exception as e:
        print(f"[ERROR] Tool Initialization Error (JSON): {e}")
        return {}


# ==============================================================================
# LANGCHAIN TOOLS
# ==============================================================================


def get_mitre_candidates(query: str, top_k: int = 25) -> tuple:
    """Semantic search for node use. Returns a tuple: (list_of_formatted_strings, metadata_map)"""
    retriever = get_retriever()
    if not retriever:
        return [], {}

    try:
        # 1. Hybrid Search
        candidates_with_score = retriever.vector_store.similarity_search_with_score(query, k=top_k)
        if not candidates_with_score:
            return [], {}
            
        candidates = [doc for doc, _ in candidates_with_score]
        qdrant_scores = [q_score for _, q_score in candidates_with_score]

        # 2. Rerank
        # Reevaluamos los candidatos devueltos por Qdrant para tener scores más finos.
        # En producción usamos Jina (remoto), y en local tiramos del CrossEncoder por defecto.
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

        scores = []
        if use_remote and not use_local_reranker and candidates:
            docs_text = [doc.page_content for doc in candidates]
            payload = {"model": reranker_model, "query": query, "documents": docs_text}
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
                    scores = [0.0] * len(candidates)
                    for item in res_data.get("results", []):
                        idx = item["index"]
                        raw_score = float(item["relevance_score"])
                        # Jina returns raw logits which are not useful for the 0.50 cutoff.
                        # Comprimimos a 0-1 con sigmoide igual que hace el CrossEncoder en local.
                        scores[idx] = 1 / (1 + math.exp(-raw_score))
            except Exception as e:
                print(f"Error calling remote reranker: {e}")
                scores = [0.0] * len(candidates)
        else:
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = (
                retriever.reranker.predict(
                    pairs, batch_size=32, activation_fn=torch.nn.Sigmoid()
                )
                if pairs
                else []
            )

        # 3. Filter (Threshold)
        # Bajamos el threshold por defecto a 0.20
        threshold = float(os.getenv("RERANKER_THRESHOLD", "0.20"))
        if len(scores) > 0:
            print(
                f"[DEBUG] get_mitre_candidates - Max Reranker Score before filtering: {max(scores):.4f}"
            )

        results_with_scores = []
        for doc, r_score, q_score in zip(candidates, scores, qdrant_scores):
            tech_id = str(doc.metadata.get("technique_id", "Unknown")).strip().upper()
            name = doc.metadata.get("name", "Unknown")
            tactics_str = doc.metadata.get("tactics", "")
            tactics = [t.strip() for t in tactics_str.split(",") if t.strip()]
            desc = doc.metadata.get("full_description", "No description available.")[:500]
            final_score = (0.7 * float(r_score)) + (0.3 * float(q_score))

            results_with_scores.append((final_score, tech_id, name, tactics_str, tactics, desc))

        # Ordenamos de mayor a menor
        results_with_scores.sort(key=lambda x: x[0], reverse=True)

        # Filtramos los que superan el threshold
        top_results = [r for r in results_with_scores if r[0] >= threshold]

        if len(top_results) == 0 and len(results_with_scores) > 0:
            print("Adaptive reranking activated. Taking top 3 ignoring threshold.")
            top_results = results_with_scores[:3]

        metadata_map = {}
        formatted_list = []
        for score, tech_id, name, tactics_str, tactics, desc in top_results:
            formatted_list.append(
                f"- {tech_id}: {name} (Tactics: {tactics_str})\n  Description: {desc}..."
            )
            metadata_map[tech_id] = {"name": name, "tactics": tactics, "score": score}

        return formatted_list, metadata_map

    except Exception as e:
        print(f"Error during search: {str(e)}")
        return [], {}
