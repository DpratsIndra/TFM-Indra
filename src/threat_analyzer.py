import json
from typing import List
from pydantic import BaseModel, Field
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from report_process import CTIReportProcessor
from consolidator import ReportConsolidator

class TechniqueMatch(BaseModel):
    technique_id: str = Field(description="ID de la técnica de MITRE (ej. T1547.001).")
    technique_name: str = Field(description="Nombre de la técnica.")
    justification: str = Field(description="Razonamiento técnico paso a paso de por qué has elegido esta técnica.")

class ChunkAnalysis(BaseModel):
    is_malicious: bool = Field(description="True si el texto describe acciones de un atacante. False si es contexto inofensivo.")
    detected_techniques: List[TechniqueMatch] = Field(description="Lista de TODAS las técnicas de MITRE presentes en el texto. Puede estar vacía si is_malicious es False.")

class ThreatAnalyzer:
    def __init__(self, model_name: str = "llama3.1:8b"):
        print("[*] Inicializando el Motor de Análisis CTI...")
        
        # Configurar Qdrant (El Cerebro)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="mitre_attack_techniques",
            embedding=self.embeddings
        )
        
        # Configurar Ollama (El Analista)
        # Usamos temperature=0 para que sea analítico y determinista (cero creatividad)
        self.llm = ChatOllama(model=model_name, temperature=0)
        # Forzamos al modelo a devolver siempre nuestro esquema Pydantic
        self.structured_llm = self.llm.with_structured_output(ChunkAnalysis)
        
        # El Prompt: La trampa de seguridad y las reglas del SOC
        self.prompt = ChatPromptTemplate.from_messages([
           ("system", """You are a Senior Cyber Threat Intelligence (CTI) Analyst.
Your task is to perform high-precision multi-label mapping between attack evidence and the MITRE ATT&CK framework.

STRICT RULES:
1. You will be provided with the 10 most likely MITRE techniques retrieved by a search engine.
2. Analyze the report fragment carefully. If there is NO clear attack, set is_malicious = False and leave detected_techniques empty.
3. If an attack is present, extract ALL techniques from the candidates list that are explicitly occurring in the fragment. A single text can contain multiple techniques.
4. Only select techniques from the provided candidates list. Do not hallucinate IDs.
5. Your justification for each technique must be technical and concise."""),
            ("human", """
--- REPORT FRAGMENT ---
{report_chunk}

--- MITRE ATT&CK CANDIDATES (Top 10 from Database) ---
{mitre_candidates}

Analyze the fragment and return the structured JSON output.
""")
        ])
        
        # Unimos el prompt y el modelo
        self.analysis_chain = self.prompt | self.structured_llm

    def format_candidates(self, docs: List) -> str:
        """Formatea los resultados de Qdrant para que el LLM los entienda fácilmente."""
        formatted = ""
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            formatted += f"Opción {i+1}:\n- ID: {metadata.get('id')}\n- Nombre: {metadata.get('name')}\n- Tácticas: {', '.join(metadata.get('tactics',[]))}\n- Descripción: {doc.page_content[:300]}...\n\n"
        return formatted

    def analyze_pdf(self, pdf_path: str):
        # 1. Fase 1: Ingesta
        processor = CTIReportProcessor()
        chunks = processor.process_pdf(pdf_path)
        
        print(f"\n[+] Iniciando análisis táctico de {len(chunks)} fragmentos...\n")
        
        final_report =[]
        
        # 2. Bucle de Análisis (Evitamos la pérdida de contexto)
        for i, chunk in enumerate(chunks):
            # if i >= 3: 
            #     break
                
            print(f"-> Analizando Fragmento {i+1}/{len(chunks)} (Pág. {chunk.metadata['page']})...")
            
            # Fase 2: Recuperar contexto de Qdrant
            retrieved_docs = self.vector_store.similarity_search(chunk.page_content, k=10)
            candidates_text = self.format_candidates(retrieved_docs)
            
            # Fase 3: Inferencia con Ollama
            try:
                result : ChunkAnalysis = self.analysis_chain.invoke({
                    "report_chunk": chunk.page_content,
                    "mitre_candidates": candidates_text
                })
                
                # Si el LLM detectó que era un ataque real y encontró técnicas, las guardamos
                if result.is_malicious and result.detected_techniques:
                    print(f"   [!] DETECTADAS {len(result.detected_techniques)} TÉCNICAS:")
                    for tech in result.detected_techniques:
                        print(f"      - {tech.technique_id} ({tech.technique_name})")
                        print(f"        Justificación: {tech.justification}\n")
                        
                        final_report.append({
                            "chunk_id": i+1,
                            "page": chunk.metadata["page"],
                            "technique_id": tech.technique_id,
                            "technique_name": tech.technique_name,
                            "justification": tech.justification
                        })
                else:
                    print(f"   [OK] Texto inofensivo. Contexto ignorado.\n")
                    
            except Exception as e:
                print(f"   [ERROR] Fallo en la inferencia del LLM: {e}\n")
        
        return final_report

def main():
    import json
    
    # 1. Configuración
    pdf_path = "data/APT29 attacks Embassies using CVE-2023-38831 - report en.pdf"
    modelo_llm = "llama3.1:8b" # Cambia esto si usas qwen2.5
    
    # 2. Analizar el PDF (Fases 1 a 3)
    analyzer = ThreatAnalyzer(model_name=modelo_llm)
    raw_results = analyzer.analyze_pdf(pdf_path)
    
    # 3. Consolidar resultados (Fase 4)
    print("\n--- INICIANDO FASE DE CONSOLIDACIÓN ---")
    consolidator = ReportConsolidator(model_name=modelo_llm)
    final_report = consolidator.consolidate(raw_results)
    
    # 4. Guardar a archivo JSON final
    output_path = "reporte_cti_final.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
        
    print(f"\n[!!!] PROCESO COMPLETADO. Informe ejecutivo guardado en '{output_path}'.")

if __name__ == "__main__":
    main()