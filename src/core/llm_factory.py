import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Factoría centralizada para instanciar el LLM según el entorno de ejecución.
    - REMOTE: Apunta al cluster de GPUs (vLLM) usando la interfaz de OpenAI.
    - LOCAL: Usa la API de Google Gemini como modelo de desarrollo rápido.
    """
    profile = os.getenv("EXECUTION_PROFILE", "LOCAL").upper()

    if profile == "REMOTE":
        use_gemma = os.getenv("USE_GEMMA4", "False").lower() in ("true", "1", "yes")
        
        if use_gemma:
            vllm_base_url = os.getenv("VLLM_BASE_URL_GEMMA", "http://10.0.152.198:8003/v1")
            model_name = os.getenv("VLLM_MODEL_NAME_GEMMA", "gemma4")
        else:
            vllm_base_url = os.getenv("VLLM_BASE_URL")
            model_name = os.getenv("VLLM_MODEL_NAME", "gpt-oss-20b")

        if not vllm_base_url:
            raise ValueError("VLLM_BASE_URL is not set in .env")

        llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "120.0"))

        return ChatOpenAI(
            model=model_name,
            base_url=vllm_base_url,
            api_key="EMPTY",
            temperature=temperature,
            seed=42,
            max_retries=1,
            timeout=llm_timeout
        )
    else:
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, max_retries=3, seed=42
        )
