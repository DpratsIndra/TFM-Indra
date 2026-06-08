import os
import json
from functools import lru_cache
from langchain_core.tools import tool

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from src.langchain_pipeline.phase3_retriever import CandidateRetriever

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
        from src.core.embedding_factory import get_embeddings
        embeddings = get_embeddings()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Create LangChain VectorStore
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
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
    mitre_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../data/mitre_data/enterprise-attack.json'
    ))
    
    if not os.path.exists(mitre_file):
        print(f"[ERROR] Tool Initialization Error: MITRE JSON not found at {mitre_file}")
        return {}
        
    try:
        with open(mitre_file, 'r', encoding='utf-8') as f:
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
            if obj.get("type") == "relationship" and obj.get("relationship_type") == "uses":
                source_id = obj.get("source_ref")
                target_id = obj.get("target_ref")
                
                if target_id and target_id.startswith("attack-pattern--") and source_id in entity_names:
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
                    
                lookup_dict[tech_id.upper()] = enriched_desc
                
        return lookup_dict
    except Exception as e:
        print(f"[ERROR] Tool Initialization Error (JSON): {e}")
        return {}

# ==============================================================================
# LANGCHAIN TOOLS
# ==============================================================================

def get_mitre_candidates(query: str) -> list:
    """Semantic search without the @tool decorator for direct node use. Returns a list of strings."""
    retriever = get_retriever()
    if not retriever:
        return []
        
    try:
        import os
        import requests
        # 1. Hybrid Search
        candidates = retriever.vector_store.similarity_search(query, k=15)
        if not candidates:
            return []
            
        # 2. Rerank
        # Reevaluamos los candidatos devueltos por Qdrant para tener scores más finos.
        # En producción usamos Jina (remoto), y en local tiramos del CrossEncoder por defecto.
        use_remote = os.getenv("EXECUTION_PROFILE", "LOCAL").upper() == "REMOTE"
        reranker_url = os.getenv("RERANKER_URL", "http://10.0.152.198:8005/v1/rerank")
        reranker_model = os.getenv("RERANKER_MODEL_NAME", "jina-reranker-v2-base-multilingual")
        
        scores = []
        if use_remote and candidates:
            docs_text = [doc.page_content for doc in candidates]
            payload = {
                "model": reranker_model,
                "query": query,
                "documents": docs_text
            }
            try:
                response = requests.post(reranker_url, json=payload, headers={"Authorization": "Bearer EMPTY"})
                if response.status_code == 200:
                    res_data = response.json()
                    scores = [0.0] * len(candidates)
                    import math
                    for item in res_data.get("results", []):
                        idx = item["index"]
                        raw_score = float(item["relevance_score"])
                        # Jina devuelve logits puros que no nos sirven para el corte de 0.50.
                        # Comprimimos a 0-1 con sigmoide igual que hace el CrossEncoder en local.
                        scores[idx] = 1 / (1 + math.exp(-raw_score))
            except Exception as e:
                print(f"Error calling remote reranker: {e}")
                scores = [0.0] * len(candidates)
        else:
            import torch
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = retriever.reranker.predict(pairs, batch_size=32, activation_fn=torch.nn.Sigmoid()) if pairs else []
        
        # 3. Filter (Threshold)
        threshold = float(os.getenv("RERANKER_THRESHOLD", "0.50"))
        if scores:
            print(f"[DEBUG] get_mitre_candidates - Max Reranker Score before filtering: {max(scores):.4f}")
            
        results = []
        for doc, score in zip(candidates, scores):
            if float(score) >= threshold:
                tech_id = doc.metadata.get("technique_id", "Unknown")
                name = doc.metadata.get("name", "Unknown")
                tactics = doc.metadata.get("tactics", "")
                desc = doc.metadata.get("full_description", "No description available.")[:500]
                results.append((float(score), f"- {tech_id}: {name} (Tactics: {tactics})\n  Description: {desc}..."))
                
        results.sort(key=lambda x: x[0], reverse=True)
        return [res[1] for res in results]
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []

@tool("MITRE_Oracle")
def mitre_oracle(query: str) -> str:
    """Use this tool to search the MITRE ATT&CK database using semantic search."""
    retriever = get_retriever()
    if not retriever:
        return "Error: Could not connect to Qdrant."
        
    try:
        import torch
        
        # 1. Hybrid Search
        # OPTIMIZACIÓN: Bajar de 30 a 15
        candidates = retriever.vector_store.similarity_search(query, k=15)
        if not candidates:
            return "No matching MITRE techniques found."
            
        # 2. Rerank
        import os
        import requests
        use_remote = os.getenv("EXECUTION_PROFILE", "LOCAL").upper() == "REMOTE"
        reranker_url = os.getenv("RERANKER_URL", "http://10.0.152.198:8005/v1/rerank")
        reranker_model = os.getenv("RERANKER_MODEL_NAME", "jina-reranker-v2-base-multilingual")
        
        scores = []
        if use_remote and candidates:
            docs_text = [doc.page_content for doc in candidates]
            payload = {
                "model": reranker_model,
                "query": query,
                "documents": docs_text
            }
            try:
                response = requests.post(reranker_url, json=payload, headers={"Authorization": "Bearer EMPTY"})
                if response.status_code == 200:
                    res_data = response.json()
                    scores = [0.0] * len(candidates)
                    import math
                    for item in res_data.get("results", []):
                        idx = item["index"]
                        raw_score = float(item["relevance_score"])
                        # Jina devuelve logits puros que no nos sirven para el corte.
                        # Comprimimos a 0-1 con sigmoide igual que hace el CrossEncoder en local.
                        scores[idx] = 1 / (1 + math.exp(-raw_score))
            except Exception as e:
                print(f"Error calling remote reranker: {e}")
                scores = [0.0] * len(candidates)
        else:
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = retriever.reranker.predict(pairs, batch_size=32, activation_fn=torch.nn.Sigmoid()) if pairs else []
        
        # 3. Filter (Threshold)
        threshold = float(os.getenv("RERANKER_THRESHOLD", "0.50"))
        if scores:
            print(f"[DEBUG] mitre_oracle - Max Reranker Score before filtering: {max(scores):.4f}")
            
        results = []
        for doc, score in zip(candidates, scores):
            if float(score) >= threshold:
                tech_id = doc.metadata.get("technique_id", "Unknown")
                name = doc.metadata.get("name", "Unknown")
                tactics = doc.metadata.get("tactics", "")
                desc = doc.metadata.get("full_description", "No description available.")[:500]
                results.append((float(score), f"- {tech_id}: {name} (Tactics: {tactics})\n  Description: {desc}..."))
                
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = [res[1] for res in results[:6]]
        
        if not top_results:
            return "No highly confident MITRE techniques found after reranking."
            
        return "CANDIDATE TECHNIQUES:\n" + "\n\n".join(top_results)
        
    except Exception as e:
        return f"Error during search: {str(e)}"


@tool("MITRE_ID_Lookup")
def mitre_id_lookup(technique_id: str) -> str:
    """
    Looks up the official MITRE ATT&CK description for a specific Technique ID (e.g., 'T1059.001').
    """
    lookup_dict = load_mitre_json()
    if not lookup_dict:
        return "Error: Local MITRE knowledge base could not be loaded."
        
    tech_id_upper = technique_id.strip().upper()
    
    if tech_id_upper in lookup_dict:
        # Return the official description
        return lookup_dict[tech_id_upper]
    else:
        return "Technique ID not found. The LLM must not use this ID."
