# Cargas la base de datos vectorial creada en vector_db.py y realizas una consulta de similitud semántica para probar su funcionamiento.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


db = Chroma(persist_directory='./mitre_chromadb', embedding_function=HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2"))


query = "The attacker used a list of common passwords to gain unauthorized access to the account."
print(f'\nPrueba de consulta: "{query}"')

resultados = db.similarity_search(query, k=3)

for i, res in enumerate(resultados):
    print(f'\nResultado {i+1}:')
    print(f'ID MITRE: {res.metadata["id"]}')
    print(f'Técnica: {res.metadata["name"]}')
