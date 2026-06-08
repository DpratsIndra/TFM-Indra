import os
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Factoría centralizada para instanciar el LLM según el entorno de ejecución.
    - REMOTE: Apunta al cluster de GPUs (vLLM) usando la interfaz de OpenAI.
    - LOCAL: Usa la API de Google Gemini como modelo de desarrollo rápido.
    """
    profile = os.getenv("EXECUTION_PROFILE", "LOCAL").upper()
    
    if profile == "REMOTE":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL_NAME", "gpt-oss-20b"),
            base_url=os.getenv("VLLM_BASE_URL", "http://10.0.152.198:8000/v1"),
            api_key="EMPTY",
            temperature=temperature,
            max_retries=3
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=3
        )
