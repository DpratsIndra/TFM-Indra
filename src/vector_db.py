import requests
import json
import os
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def main():
    # Descargar datos de MITRE ATT&CK (formato STIX 2.1)
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    print("Descargando la base de datos de MITRE ATT&CK...")
    response = requests.get(url)
    mitre_data = response.json()

    # Extraer y limpiar las técnicas (Attack Patterns)
    documents = []
    print("Procesando y limpiando técnicas...")

    for obj in mitre_data.get('objects', []):
        if obj.get('type') == 'attack-pattern':
            if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
                continue  # Omitir técnicas revocadas o deprecadas
            
            # Metadatos básicos
            technique_id = next((ref.get('external_id') for ref in obj.get('external_references', []) if ref.get('source_name') == 'mitre-attack'), 'Desconocido')
            name = obj.get('name', 'Sin nombre')

            # Extracción de artefactos técnicos (APIs, herramientas, blogs)
            tech_artifacts = [ref.get('source_name') for ref in obj.get('external_references', []) if ref.get('source_name') != 'mitre-attack']
            tech_artifacts_str = ', '.join(tech_artifacts) if tech_artifacts else 'N/A'

            # Tácticas y Plataformas asociadas
            tactics = [phase.get('phase_name') for phase in obj.get('kill_chain_phases', [])]
            tactics_str = ', '.join(tactics) if tactics else 'N/A'

            platforms = obj.get('x_mitre_platforms', [])
            platforms_str = ', '.join(platforms) if platforms else 'N/A'

            # Guía de detección
            detection_guide = obj.get('x_mitre_detection', 'N/A')

            content = f'''
ID: {technique_id}
Nombre: {name}
Tácticas: {tactics_str}
Tecnologías/Artefactos: {tech_artifacts_str}
Plataformas: {platforms_str}
Guía de Detección: {detection_guide}
Descripción: {obj.get('description', 'N/A')}
            '''.strip()

            # Metadatos para filtrado lógico por Agente
            metadata = {
                'id': technique_id,
                'name': name,
                'tactics': tactics,
                'platforms': platforms,
                'tech_artifacts': tech_artifacts,
                'detection_guide': detection_guide,
                'description': obj.get('description', 'N/A'),
                'is_subtechnique': obj.get('x_mitre_is_subtechnique', False)
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

    print(f'Se han extraído {len(documents)} técnicas de MITRE ATT&CK.')

    # Crear los Embeddings y la base de datos vectorial
    print('Descargando modelo de embeddings...')
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print('Conectando a Qdrant y creando la base de datos vectorial...')

    QdrantVectorStore.from_documents(
        documents,
        embeddings,
        url="http://localhost:6333",
        collection_name="mitre_attack_techniques",
        force_recreate=True,
        prefer_grpc=True
    )

    print(f'Base de datos vectorial creada con {len(documents)} documentos.')

if __name__ == "__main__":
    main()