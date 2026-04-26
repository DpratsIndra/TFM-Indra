from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

def inicialize_search():
    print('Cargando modelo de embeddings...')
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print('Conectando a Qdrant y preparando la base de datos vectorial...')
    cliente_qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=True)

    vector_store = QdrantVectorStore(
        embedding=embeddings,
        client=cliente_qdrant,
        collection_name="mitre_attack_techniques"
    )

    return vector_store

def main():
    vector_store = inicialize_search()

    query = "Para mantener la persistencia en el equipo de la víctima, el malware modificó el registro de Windows añadiendo una nueva clave en la ruta 'HKCU\\Software\\Windows\\CurrentVersion\\Run'. De esta forma, el programa malicioso se ejecutaba automáticamente cada vez que el usuario iniciaba sesión."
    print(f'Realizando consulta: "{query}"')

    results = vector_store.similarity_search(query, k=3)

    print("\nResultados de la consulta:")
    for i, doc in enumerate(results, 1):
        print(f"\nResultado {i}:")
        print(doc.page_content)

if __name__ == "__main__":
    main()