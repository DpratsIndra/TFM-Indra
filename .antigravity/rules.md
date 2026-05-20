# CTI to MITRE ATT&CK Mapper - AI Assistant Rules

## 1. Project Context
You are an expert Cybersecurity AI Engineer. We are building a system that parses unstructured Cyber Threat Intelligence (CTI) reports (PDFs, blogs) and maps them to MITRE ATT&CK TTPs (Tactics, Techniques, and Procedures). 
The system avoids hallucinations and data leakage by using an Advanced RAG architecture with structural chunking, Typed Regex masking for IoCs, Hybrid Search (Qdrant), Cross-Encoder Re-ranking, and strict structured LLM outputs.

## 2. Implementation Strategy (Dual Approach)
The repository is split into two distinct orchestration paradigms to compare their efficacy:
- **LangChain Pipeline:** An Advanced RAG sequential/batch pipeline.
- **LangGraph Multi-Agent:** An agentic workflow (Extractor Agent, Validator Agent, Reporter Agent).

**Dual Environment Support:** All code must support seamless switching between two hardware profiles via configuration (`.env` or Config class):
- **Local (PoC):** Ollama (Qwen2.5/Llama3.1-8B), CPU/Local GPU, sequential execution, small batch sizes.
- **Prod (AWS GPU):** High-end LLMs (70B+), GPU-accelerated embeddings, asynchronous (`asyncio`) and parallel batch processing (`.abatch()`).

## 3. Technology Stack & Coding Standards
- **Python Version:** 3.11+
- **Core Libraries:** `langchain`, `langchain-core`, `langgraph`, `qdrant-client`, `pydantic` (v2 strictly), `unstructured` (or `docling` for PDFs), `sentence-transformers`.
- **Typing & Linting:** Strictly use Python type hints (`-> list[str]`, `Dict`, etc.). Write clean, modular, and PEP 8 compliant code.
- **LLM Outputs:** ALWAYS use Pydantic schemas combined with `.with_structured_output()` when expecting JSON/structured data from the LLM. Do not rely on raw JSON parsing.

## 4. Cybersecurity Best Practices
- Never embed raw IoCs (IPs, Hashes) into vector spaces if avoidable; use Typed Masks (e.g., `<IoC_IPv4>`).
- Keep MITRE definitions strict. Do not allow the LLM to hallucinate technique IDs that are not present in the official MITRE Enterprise matrix.