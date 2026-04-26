import json
from typing import List
from pydantic import BaseModel, Field
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Importamos tu módulo de la Fase 1
from report_process import CTIReportProcessor

# 1. Definimos la estructura ESTRICTA que queremos que devuelva la IA
class TTPDetection(BaseModel):
    is_malicious: bool = Field(description="True si el texto describe una acción maliciosa de un atacante. False si es contexto inofensivo, explicaciones generales o acciones de un administrador.")
    technique_id: str = Field(description="ID de la técnica de MITRE (ej. T1547.001). Si is_malicious es False, devuelve 'N/A'.")
    technique_name: str = Field(description="Nombre de la técnica. Si is_malicious es False, devuelve 'N/A'.")
    justification: str = Field(description="Razonamiento técnico paso a paso de por qué has elegido esa técnica o por qué has decidido que el texto es inofensivo.")

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
        self.structured_llm = self.llm.with_structured_output(TTPDetection)
        
        # El Prompt: La trampa de seguridad y las reglas del SOC
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un analista Senior de Ciberinteligencia (CTI) Nivel 3.
Tu objetivo es analizar un fragmento de un reporte de seguridad y determinar si se describe una táctica o técnica de ataque real.

REGLAS ESTRICTAS:
1. Analiza el texto proporcionado. Si el texto habla de acciones normales de usuarios, descripciones de herramientas de defensa, o contexto inofensivo, clasifícalo como is_malicious = False.
2. Si el texto describe un ATAQUE REAL, revisa la lista de Técnicas Candidatas de MITRE ATT&CK que se te proporciona.
3. Elige la técnica candidata que MEJOR describa el ataque. NO te inventes IDs de técnicas que no estén en la lista de candidatas.
4. Tu justificación debe ser técnica y basarse en las evidencias del texto."""),
            ("human", """
--- FRAGMENTO DEL REPORTE ---
{report_chunk}

--- TÉCNICAS CANDIDATAS DE MITRE (Recuperadas de la Base de Datos) ---
{mitre_candidates}

Analiza el fragmento y devuelve el JSON correspondiente.
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
                result : TTPDetection = self.analysis_chain.invoke({
                    "report_chunk": chunk.page_content,
                    "mitre_candidates": candidates_text
                })
                
                # Si el LLM detectó que era un ataque real, lo guardamos
                if result.is_malicious:
                    print(f"   [!] DETECCIÓN: {result.technique_id} - {result.technique_name}")
                    print(f"   [>] Justificación: {result.justification}\n")
                    
                    final_report.append({
                        "chunk_id": i+1,
                        "page": chunk.metadata["page"],
                        "technique_id": result.technique_id,
                        "technique_name": result.technique_name,
                        "justification": result.justification
                    })
                else:
                    print(f"   [OK] Texto inofensivo. Contexto ignorado.\n")
                    
            except Exception as e:
                print(f"   [ERROR] Fallo en la inferencia del LLM: {e}\n")
        
        return final_report

def main():
    # Asegúrate de tener tu modelo local corriendo (ej. ollama run llama3.1:8b)
    analyzer = ThreatAnalyzer(model_name="llama3.1:8b") # Cambia a qwen2.5 si usaste ese
    
    pdf_path = "data/APT29 attacks Embassies using CVE-2023-38831 - report en.pdf"
    resultados = analyzer.analyze_pdf(pdf_path)
    
    print("\n=== RESUMEN FINAL DEL REPORTE ===")
    print(json.dumps(resultados, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()