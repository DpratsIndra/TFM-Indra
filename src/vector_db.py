import requests
import json
import os
from langchain_elasticsearch import ElasticsearchStore
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
        if obj.get('type') == 'attack-pattern' and not obj.get('x_mitre_is_subtechnique'):
            # Buscar el ID oficial de MITRE
            technique_id = 'Desconocido'
            for ext_ref in obj.get('external_references', []):
                if ext_ref.get('source_name') == 'mitre-attack':
                    technique_id = ext_ref.get('external_id')
                    break

            name = obj.get('name', 'Sin nombre')
            description = obj.get('description', 'Sin descripción')

            # Crear un documento estructurado para LangChain
            doc = Document(
                page_content=f'Técnica: {name}\nDescripción: {description}',
                metadata={'id': technique_id, 'name': name}
            )
            documents.append(doc)

    print(f'Se han extraído {len(documents)} técnicas de MITRE ATT&CK.')

    # Crear los Embeddings y la base de datos vectorial
    print('Descargando modelo de embeddings...')
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print('Conectando a Elasticsearch y creando la base de datos vectorial...')

    vector_store = ElasticsearchStore(
        embedding=embeddings,
        es_url="http://localhost:9200",
        index_name="mitre_hybrid_index",
    )

    vector_store.add_documents(documents)

    print(f'Base de datos vectorial creada con {len(documents)} documentos.')

    # Prueba de consulta
    query = "El adversario intentó acceder al sistema probando miles de contraseñas de forma automatizada."
    print(f'\nPrueba de consulta: "{query}"')

    resultados = vector_store.similarity_search(query, k=3)

    for i, res in enumerate(resultados):
        print(f'\nResultado {i+1}:')
        print(f'ID MITRE: {res.metadata["id"]}')
        print(f'Técnica: {res.metadata["name"]}')
        print(f'Descripción: {res.page_content}')

if __name__ == "__main__":
    main()