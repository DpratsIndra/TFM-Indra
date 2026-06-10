import json
import logging
import os
import sys
import urllib.request
from typing import List, Dict, Any, Optional

# Add the project root to sys.path so imports from src work when running the file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.embedding_factory import get_embeddings

import torch
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class MitreIndexer:
    """
    Handles Phase 2 of the LangChain pipeline: MITRE ATT&CK Indexing.
    Loads MITRE Enterprise JSON data, semantically enriches the descriptions,
    and indexes them into a Qdrant vector database using HuggingFace embeddings.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "mitre_attack",
    ) -> None:
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

        self.embeddings = get_embeddings()

    def load_mitre_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MITRE data from {file_path}: {e}")
            raise

        objects = data.get("objects", []) if isinstance(data, dict) else data
        if not isinstance(objects, list):
            logger.warning(
                "Unexpected JSON structure. Expected a list or a STIX bundle with 'objects'."
            )
            return []

        # 1. Extraer nombres de herramientas y malware
        entity_names = {}
        for obj in objects:
            if obj.get("type") in ["malware", "tool", "intrusion-set"]:
                entity_names[obj.get("id")] = obj.get("name")

        # 2. Resolver las relaciones "uses" (Herramienta -> usa -> Técnica)
        technique_examples = {}
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
                    if target_id not in technique_examples:
                        technique_examples[target_id] = []
                    tool_name = entity_names[source_id]
                    if tool_name not in technique_examples[target_id]:
                        technique_examples[target_id].append(tool_name)

        # 3. Construir las técnicas enriquecidas
        techniques = []
        for obj in objects:
            if obj.get("type") != "attack-pattern":
                continue

            if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                continue

            external_refs = obj.get("external_references", [])
            technique_id = "Unknown"
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    technique_id = ref.get("external_id", "Unknown")
                    break

            if not technique_id.startswith("T"):
                continue

            tactics = [
                phase.get("phase_name", "")
                for phase in obj.get("kill_chain_phases", [])
                if phase.get("kill_chain_name") == "mitre-attack"
            ]

            # Recuperar las herramientas asociadas a esta técnica
            stix_id = obj.get("id")
            examples = technique_examples.get(stix_id, [])

            technique_dict = {
                "technique_id": technique_id,
                "name": obj.get("name", "Unknown"),
                "description": obj.get("description", ""),
                "tactics": tactics,
                "platforms": obj.get("x_mitre_platforms", []),
                "procedure_examples": examples,
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
            enriched_text = (
                f"{base_desc}\n\nTechnical Keywords & Examples: {'; '.join(keywords)}"
            )
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

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

        documents = []
        for tech in techniques:
            enriched_content = self._enrich_description(tech)

            # Metadata must be flat and use simple types for Qdrant compatibility
            metadata = {
                "technique_id": tech.get("technique_id"),
                "name": tech.get("name"),
                "tactics": ", ".join(tech.get("tactics", [])),
                "platforms": ", ".join(tech.get("platforms", [])),
                # CRITICAL FIX: Pass the enriched content (which includes tools and malware names)
                # to the metadata, so the LLM Validator can read the same evidence Qdrant used.
                "full_description": enriched_content,
            }

            # Chunk the enriched content for fast Cross-Encoder evaluation
            chunks = splitter.split_text(enriched_content)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))

        logger.info(f"Prepared {len(documents)} Document objects for vectorization.")

        try:
            # Initialize Qdrant Client explicitly to test connection
            client = QdrantClient(url=self.qdrant_url)
            # A lightweight query to verify the connection
            client.get_collections()

            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

            # Use LangChain's QdrantVectorStore to create the collection and embed documents
            logger.info(
                f"Indexing documents into Qdrant using device: {self.device}..."
            )
            vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                url=self.qdrant_url,
                collection_name=self.collection_name,
                force_recreate=True,
            )
            logger.info("Successfully built and indexed the MITRE ATT&CK vector store.")
            return vector_store

        except Exception as e:
            # Catching general Exception handles both ConnectionError and UnexpectedResponse
            logger.error(
                f"Qdrant Connection or Indexing Error: {e}. Ensure Qdrant is running."
            )
            return None


def setup_mitre_index(
    qdrant_url: str = "http://localhost:6333", collection_name: str = "mitre_attack"
):
    """
    Helper function to automatically download the MITRE ATT&CK dataset
    and populate the Qdrant vector database if it hasn't been set up yet.
    """
    mitre_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/mitre_data")
    )
    mitre_file = os.path.join(mitre_dir, "enterprise-attack.json")

    os.makedirs(mitre_dir, exist_ok=True)

    if not os.path.exists(mitre_file):
        url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
        logger.info(f"Descargando datos oficiales de MITRE ATT&CK desde {url}...")
        print(f"Descargando datos oficiales de MITRE ATT&CK desde {url}...")
        urllib.request.urlretrieve(url, mitre_file)
        logger.info("Descarga completada.")
        print("Descarga completada.")

    logger.info("Iniciando Fase 2: Construcción de la Base Vectorial (Qdrant)...")
    print("Iniciando Fase 2: Construcción de la Base Vectorial (Qdrant)...")
    indexer = MitreIndexer(qdrant_url=qdrant_url, collection_name=collection_name)
    store = indexer.build_vector_store(mitre_file)
    if not store:
        raise Exception("Fallo al construir el índice de MITRE en Qdrant.")
    return store


if __name__ == "__main__":
    # Add project root to sys.path
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )

    try:
        setup_mitre_index()
        print("¡Indexación finalizada con éxito! Ya puedes ejecutar main_pipeline.py")
    except Exception as e:
        print(f"Hubo un error durante la indexación: {e}")
