# CTI-RAG-Mapper Architecture

This repository contains the codebase for extracting MITRE ATT&CK TTPs from unstructured CTI reports using LLMs.

## Directory Structure

```text
cti-mapper/
├── .antigravity/
│   └── rules.md                  # IDE AI context rules
├── data/
│   ├── raw_reports/              # Input PDFs/txt files
│   ├── mitre_data/               # MITRE ATT&CK STIX/JSON files
│   └── output/                   # Generated JSON reports
├── infrastructure/
│   └── docker-compose.yml        # Qdrant Vector DB & Ollama setup
├── src/
│   ├── core/                     # Shared resources
│   │   ├── config.py             # Env vars (Local vs AWS profile)
│   │   ├── ioc_masker.py         # Regex logic for <IoC_IPv4>, etc.
│   │   └── schemas.py            # Pydantic models (CTI_Extraction, etc.)
│   ├── langchain_pipeline/       # Advanced RAG Approach
│   │   ├── phase1_ingestion.py   # PDF Parsing & Semantic Chunking
│   │   ├── phase2_indexer.py     # MITRE Hybrid Vectorization (Qdrant)
│   │   ├── phase3_retriever.py   # Hybrid Retrieval & Cross-Encoder Reranking
│   │   ├── phase4_inference.py   # LLM Map-Reduce Extraction
│   │   └── main_pipeline.py      # LangChain Orchestrator
│   └── langgraph_agents/         # Multi-Agent Approach (Future)
│       ├── agents/
│       ├── state.py
│       └── graph_builder.py
├── dashboard/
│   └── cti_inspector.py  
├── .env.example                  # Environment variables template
├── ARCHITECTURE.md               # This file
├── requirements.txt              # Prod dependencies
└── requirements-dev.txt          # Testing & Linting