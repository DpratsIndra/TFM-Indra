import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "False").lower() in ("true", "1", "yes")
    
    if use_remote:
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3"),
            base_url=os.getenv("EMBEDDINGS_BASE_URL", "http://10.0.152.198:8002/v1"),
            api_key="EMPTY"
        )
    else:
        return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
