import json
import logging
from typing import List, Dict, Any, Optional

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

class MitreIndexer:
    """
    Handles Phase 2 of the LangChain pipeline: MITRE ATT&CK Indexing.
    Loads MITRE Enterprise JSON data, semantically enriches the descriptions,
    and indexes them into a Qdrant vector database using HuggingFace embeddings.
    """

    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "mitre_attack") -> None:
        """
        Initializes the indexer with Qdrant connection details.
        
        Args:
            qdrant_url (str): The URL of the Qdrant instance.
            collection_name (str): The name of the Qdrant collection to store vectors.
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Automatically select the best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        # BAAI/bge-m3 is a state-of-the-art multilingual model great for retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def load_mitre_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads and parses a MITRE ATT&CK Enterprise JSON file (typically STIX 2.1 format).
        Extracts technique_id, name, description, tactics, and platforms.
        
        Args:
            file_path (str): Path to the MITRE JSON file.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a technique.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MITRE data from {file_path}: {e}")
            raise
            
        techniques = []
        
        # MITRE ATT&CK data is typically wrapped in a STIX bundle with an "objects" array.
        objects = data.get("objects", []) if isinstance(data, dict) else data
        
        if not isinstance(objects, list):
            logger.warning("Unexpected JSON structure. Expected a list or a STIX bundle with 'objects'.")
            return techniques
            
        for obj in objects:
            # We are only interested in attack techniques
            if obj.get("type") != "attack-pattern":
                continue
                
            # Extract technique ID (e.g., T1059) from STIX external_references
            external_refs = obj.get("external_references", [])
            technique_id = "Unknown"
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    technique_id = ref.get("external_id", "Unknown")
                    break
                    
            if not technique_id.startswith("T"):
                continue  # Skip non-standard techniques or sub-objects without proper IDs
                
            # Extract tactics from STIX kill_chain_phases
            tactics = [
                phase.get("phase_name", "") 
                for phase in obj.get("kill_chain_phases", []) 
                if phase.get("kill_chain_name") == "mitre-attack"
            ]
            
            technique_dict = {
                "technique_id": technique_id,
                "name": obj.get("name", "Unknown"),
                "description": obj.get("description", ""),
                "tactics": tactics,
                "platforms": obj.get("x_mitre_platforms", []),
                # Storing potential examples if present in this object or custom schema
                "procedure_examples": obj.get("procedure_examples", [])
            }
            techniques.append(technique_dict)
            
        logger.info(f"Loaded {len(techniques)} techniques from {file_path}.")
        return techniques

    def _enrich_description(self, technique: Dict[str, Any]) -> str:
        """
        Enriches the technique's description with tactical keywords and procedure examples.
        This concatenation improves the semantic vector density, increasing the Recall of RAG.
        
        Args:
            technique (Dict[str, Any]): The extracted technique dictionary.
            
        Returns:
            str: The enriched text ready for vectorization.
        """
        base_desc = technique.get("description", "").strip()
        
        keywords = []
        tactics = technique.get("tactics", [])
        if tactics:
            keywords.append(f"Tactics: {', '.join(tactics)}")
            
        platforms = technique.get("platforms", [])
        if platforms:
            keywords.append(f"Platforms: {', '.join(platforms)}")
            
        examples = technique.get("procedure_examples", [])
        if examples:
            if isinstance(examples, list):
                examples_text = " | ".join(str(e) for e in examples)
            else:
                examples_text = str(examples)
            keywords.append(f"Procedure Examples: {examples_text}")
            
        # Combine base description with semantic keywords
        if keywords:
            enriched_text = f"{base_desc}\n\nTechnical Keywords & Examples: {'; '.join(keywords)}"
        else:
            enriched_text = base_desc
            
        return enriched_text

    def build_vector_store(self, file_path: str) -> Optional[QdrantVectorStore]:
        """
        End-to-end method to load the JSON, enrich descriptions, create Document objects,
        and index them into Qdrant.
        
        Args:
            file_path (str): Path to the MITRE JSON file.
            
        Returns:
            Optional[QdrantVectorStore]: The initialized Qdrant vector store, or None if failed.
        """
        techniques = self.load_mitre_data(file_path)
        if not techniques:
            logger.error("No techniques extracted. Aborting vectorization.")
            return None
            
        documents = []
        for tech in techniques:
            enriched_content = self._enrich_description(tech)
            
            # Metadata must be flat and use simple types for Qdrant compatibility
            metadata = {
                "technique_id": tech.get("technique_id"),
                "name": tech.get("name"),
                "tactics": ", ".join(tech.get("tactics", [])),
                "platforms": ", ".join(tech.get("platforms", []))
            }
            
            doc = Document(page_content=enriched_content, metadata=metadata)
            documents.append(doc)
            
        logger.info(f"Prepared {len(documents)} Document objects for vectorization.")
        
        try:
            # Initialize Qdrant Client explicitly to test connection
            client = QdrantClient(url=self.qdrant_url)
            # A lightweight query to verify the connection
            client.get_collections()
            
            # Use LangChain's QdrantVectorStore to create the collection and embed documents
            logger.info(f"Indexing documents into Qdrant using device: {self.device}...")
            vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=self.qdrant_url,
                collection_name=self.collection_name,
            )
            logger.info("Successfully built and indexed the MITRE ATT&CK vector store.")
            return vector_store
            
        except Exception as e:
            # Catching general Exception handles both ConnectionError and UnexpectedResponse
            logger.error(f"Qdrant Connection or Indexing Error: {e}. Ensure Qdrant is running.")
            return None

if __name__ == "__main__":
    import os
    import sys
    import urllib.request
    
    # Add project root to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Setup paths
    mitre_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/mitre_data'))
    mitre_file = os.path.join(mitre_dir, 'enterprise-attack.json')
    
    # Create directory if it doesn't exist
    os.makedirs(mitre_dir, exist_ok=True)
    
    # Download MITRE data if not present
    if not os.path.exists(mitre_file):
        url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
        print(f"Descargando datos oficiales de MITRE ATT&CK desde {url}...")
        urllib.request.urlretrieve(url, mitre_file)
        print("Descarga completada.")
        
    print("Iniciando Fase 2: Construcción de la Base Vectorial (Qdrant)...")
    indexer = MitreIndexer()
    store = indexer.build_vector_store(mitre_file)
    if store:
        print("¡Indexación finalizada con éxito! Ya puedes ejecutar main_pipeline.py")
    else:
        print("Hubo un error durante la indexación.")
