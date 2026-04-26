from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# El LLM SOLO nos devolverá el resumen, nada de IDs ni páginas.
class SynthesizedSummary(BaseModel):
    summary_of_evidence: str = Field(
        description="A single, professional executive summary paragraph explaining how the attacker used this technique based on the provided evidence."
    )

class ReportConsolidator:
    def __init__(self, model_name: str = "llama3.1:8b"):
        print("[*] Inicializando el Consolidador de Reportes (SOC Lead)...")
        # Usamos temperature 0 para que la redacción sea analítica y no creativa
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.structured_llm = self.llm.with_structured_output(SynthesizedSummary)

    def consolidate(self, raw_detections: List[Dict]) -> List[Dict]:
        if not raw_detections:
            print("[-] No se encontraron detecciones para consolidar.")
            return[]

        # 1. Agrupar evidencias por ID de técnica de forma determinista (Python)
        grouped_data = {}
        for det in raw_detections:
            tid = det.get('technique_id', 'N/A')
            
            # Evitar procesar falsos positivos descartados previamente
            if tid == 'N/A':
                continue
                
            if tid not in grouped_data:
                grouped_data[tid] = {
                    "name": det['technique_name'],
                    "evidences": [],
                    "pages": set()
                }
            grouped_data[tid]["evidences"].append(det['justification'])
            grouped_data[tid]["pages"].add(det['page'])

        consolidated_results =[]
        print(f"[*] Consolidando {len(grouped_data)} técnicas únicas encontradas...")

        # 2. Pedir a Ollama que sintetice cada grupo
        for tid, data in grouped_data.items():
            print(f"   [+] Sintetizando evidencia ejecutiva para {tid}...")
            
            combined_evidence = "\n- ".join(data["evidences"])
            
            prompt = f"""You are a Lead Cyber Threat Intelligence (CTI) Analyst in a SOC. 
The MITRE ATT&CK technique '{tid}' ({data['name']}) has been detected multiple times in a threat report. 
Below is the partial evidence extracted by junior analysts from different pages of the report:

EVIDENCE:
- {combined_evidence}

TASK:
Write a single, cohesive, professional executive summary paragraph explaining exactly how the attacker utilized this technique across the campaign. Merge the context logically. Do not reference the 'junior analysts' or use phrases like 'The text says'. Just describe the attacker's actions."""

            try:
                # Inferencia final de síntesis
                summary_obj = self.structured_llm.invoke(prompt)
                
                # 3. Montamos el JSON final perfecto (IA + Python)
                consolidated_results.append({
                    "technique_id": tid,
                    "technique_name": data['name'],
                    "executive_summary": summary_obj.summary_of_evidence,
                    "pages_detected": sorted(list(data["pages"]))
                })
            except Exception as e:
                print(f"   [!] Error sintetizando {tid}: {e}")
        
        return consolidated_results