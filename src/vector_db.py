import requests
from typing import List, Dict, Any
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class MitreIndexer:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "mitre_attack_techniques"):
        self.mitre_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = "BAAI/bge-m3"

    def fetch_data(self) -> Dict[str, Any]:
        print("Descargando base de datos de MITRE ATT&CK...")
        response = requests.get(self.mitre_url)
        response.raise_for_status()
        return response.json()

    def _extract_technique_id(self, obj: Dict[str, Any]) -> str:
        for ref in obj.get('external_references',[]):
            if ref.get('source_name') == 'mitre-attack':
                return ref.get('external_id', 'Desconocido')
        return 'Desconocido'

    def parse_documents(self, mitre_data: Dict[str, Any]) -> List[Document]:
        print("Procesando y limpiando técnicas...")
        documents =[]

        for obj in mitre_data.get('objects',[]):
            if obj.get('type') != 'attack-pattern':
                continue
            if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
                continue
            
            technique_id = self._extract_technique_id(obj)
            name = obj.get('name', 'Sin nombre')
            description = obj.get('description', 'N/A')
            detection_guide = obj.get('x_mitre_detection', 'N/A')

            tech_artifacts = [
                ref.get('source_name') 
                for ref in obj.get('external_references', []) 
                if ref.get('source_name') != 'mitre-attack'
            ]
            
            tactics =[phase.get('phase_name') for phase in obj.get('kill_chain_phases', [])]
            platforms = obj.get('x_mitre_platforms',[])

            content = (
                f"ID: {technique_id}\n"
                f"Nombre: {name}\n"
                f"Tácticas: {', '.join(tactics) if tactics else 'N/A'}\n"
                f"Tecnologías/Artefactos: {', '.join(tech_artifacts) if tech_artifacts else 'N/A'}\n"
                f"Plataformas: {', '.join(platforms) if platforms else 'N/A'}\n"
                f"Guía de Detección: {detection_guide}\n"
                f"Descripción: {description}"
            )

            metadata = {
                'id': technique_id,
                'name': name,
                'tactics': tactics,
                'platforms': platforms,
                'tech_artifacts': tech_artifacts,
                'detection_guide': detection_guide,
                'description': description,
                'is_subtechnique': obj.get('x_mitre_is_subtechnique', False)
            }

            documents.append(Document(page_content=content, metadata=metadata))

        print(f"Se han extraído {len(documents)} técnicas de MITRE ATT&CK.")
        return documents

    def index_to_qdrant(self, documents: List[Document]) -> None:
        print("Cargando modelo de embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("Conectando a Qdrant y poblando la base de datos vectorial...")
        QdrantVectorStore.from_documents(
            documents,
            embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name,
            force_recreate=True,
            prefer_grpc=True
        )
        print(f"Base de datos vectorial creada con {len(documents)} documentos.")

    def run(self) -> None:
        raw_data = self.fetch_data()
        docs = self.parse_documents(raw_data)
        self.index_to_qdrant(docs)


if __name__ == "__main__":
    indexer = MitreIndexer()
    indexer.run()