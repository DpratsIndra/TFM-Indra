import os
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "vllm").lower()
    
    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL_NAME", "gpt-oss-20b"),
            base_url=os.getenv("VLLM_BASE_URL", "http://10.0.152.198:8000/v1"),
            api_key="EMPTY",
            temperature=temperature,
            max_retries=3
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            temperature=temperature,
            max_retries=3
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature
        )
