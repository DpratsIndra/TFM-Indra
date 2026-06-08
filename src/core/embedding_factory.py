import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    """
    Factoría para seleccionar el modelo de Embeddings.
    - REMOTE: Endpoint TEI en el servidor de inferencia (más rápido para batches grandes).
    - LOCAL: Ejecuta bge-m3 en local mediante HuggingFace (suficiente para pruebas y poco consumo).
    """
    profile = os.getenv("EXECUTION_PROFILE", "LOCAL").upper()
    
    if profile == "REMOTE":
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3"),
            base_url=os.getenv("EMBEDDINGS_BASE_URL", "http://10.0.152.198:8002/v1"),
            api_key="EMPTY"
        )
    else:
        return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
