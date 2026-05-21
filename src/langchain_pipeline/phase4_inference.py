import os
import logging
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.schemas import TTPDetection

logger = logging.getLogger(__name__)

# Cargar variables de entorno (ej. USE_PROMPT_REPETITION, ENVIRONMENT_PROFILE)
load_dotenv()

class TTPAnalyzer:
    """
    Maneja la Fase 4 del pipeline de LangChain: Inferencia y Mapeo LLM.
    Utiliza LCEL para interrogar al modelo con salidas estructuradas, soportando
    Prompt Repetition y perfiles de ejecución dinámicos (LOCAL vs AWS).
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        Inicializa el analizador de TTPs con el LLM especificado.
        
        Args:
            llm (BaseChatModel): Una instancia inicializada de un ChatModel de LangChain (ej. ChatOllama).
        """
        self.llm = llm
        
        # Obtener variables de entorno (o usar valores por defecto seguros)
        self.use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
        
        # Soportamos ENVIRONMENT_PROFILE o EXECUTION_PROFILE como nombre de variable
        self.execution_profile = os.getenv("EXECUTION_PROFILE", os.getenv("ENVIRONMENT_PROFILE", "LOCAL")).upper()
        
        logger.info(f"TTPAnalyzer inicializado. Perfil: {self.execution_profile}, Prompt Repetition: {self.use_prompt_repetition}")

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """
        Construye la plantilla del prompt dinámicamente con los tres bloques solicitados,
        inyectando la lógica de 'Prompt Repetition' si está activada en la configuración.
        
        Returns:
            ChatPromptTemplate: Plantilla estructurada para invocar al LLM.
        """
        system_instruction = (
            "Act as an expert CTI and MITRE ATT&CK analyst. Your objective is to analyze extracts from a threat "
            "intelligence report and deterministically confirm if they demonstrate the use of the indicated technique. "
            "You must extract the 'justification' (why it was detected) and the 'procedure' (a concise explanation "
            "of the exact and specific action the attacker took in this report)."
        )
        
        # Aplicar el patrón "Prompt Repetition" doblando el prompt de sistema y el del usuario
        if self.use_prompt_repetition:
            system_instruction = f"{system_instruction}\n\n{system_instruction}"
        
        user_message = (
            "### MITRE CONTEXT\n"
            "Technique: {mitre_technique_id} - {mitre_technique_name}\n"
            "Tactics: {mitre_tactics}\n\n"
            "### REPORT EVIDENCE\n"
            "{supporting_chunks}\n\n"
            "Based EXCLUSIVELY on the provided evidence, "
            "is the use of the {mitre_technique_id} technique confirmed? Return the structured JSON."
        )
        
        if self.use_prompt_repetition:
            user_message = f"{user_message}\n\n{user_message}"
            
        return ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", user_message)
        ])

    def analyze_candidates(self, candidates_dict: Dict[str, Dict[str, Any]]) -> List[TTPDetection]:
        """
        Ejecuta la cadena LCEL para analizar los candidatos de MITRE filtrados en la Fase 3.
        Adapta la ejecución a paralelo (.batch) o secuencial según el perfil (AWS/LOCAL).
        
        Args:
            candidates_dict (Dict[str, Dict]): Diccionario de candidatos recuperados.
            
        Returns:
            List[TTPDetection]: Una lista con las detecciones donde is_present == True.
        """
        if not candidates_dict:
            logger.warning("No hay candidatos para analizar.")
            return []

        prompt = self._build_prompt_template()
        
        # Construcción de la cadena LCEL forzando el esquema Pydantic
        chain = prompt | self.llm.with_structured_output(TTPDetection)
        
        inputs = []
        for tech_id, data in candidates_dict.items():
            # Unir los chunks que soportan esta técnica para dárselos como evidencia
            joined_chunks = "\n---\n".join(data.get("supporting_chunks", []))
            
            inputs.append({
                "mitre_technique_id": tech_id,
                "mitre_technique_name": data.get("name", "Desconocida"),
                "mitre_tactics": ", ".join(data.get("tactics", [])),
                "supporting_chunks": joined_chunks,
                "_meta_tactics": data.get("tactics", []),
                "_meta_score": data.get("score", 0.0)
            })
            
        logger.info(f"Preparados {len(inputs)} candidatos para inferencia LLM.")
        
        results: List[TTPDetection] = []
        
        # Orquestación de la ejecución basada en Hardware/Perfil
        if self.execution_profile == "AWS":
            logger.info("Perfil AWS detectado: Ejecutando cadena en paralelo (Batching)...")
            try:
                # LLMs grandes en la nube o clusters GPU aguantan batching
                batch_responses = chain.batch(inputs)
                
                # Descartar nulos por si hubo fallos de red/parseo y poblar metadata
                for inp, res in zip(inputs, batch_responses):
                    if res is not None:
                        res.tactic = inp["_meta_tactics"]
                        res.technique_name = inp["mitre_technique_name"]
                        res.confidence_score = inp["_meta_score"]
                        results.append(res)
            except Exception as e:
                logger.error(f"Error durante el batching en AWS: {e}")
                
        else:
            logger.info("Perfil LOCAL detectado: Ejecutando cadena de forma secuencial (For-loop)...")
            # Para evitar OOM en Ollama o hardware modesto
            for i, inp in enumerate(inputs, 1):
                try:
                    logger.info(f"[{i}/{len(inputs)}] Consultando al LLM para la técnica: {inp['mitre_technique_id']}...")
                    response = chain.invoke(inp)
                    if response:
                        response.tactic = inp["_meta_tactics"]
                        response.technique_name = inp["mitre_technique_name"]
                        response.confidence_score = inp["_meta_score"]
                        results.append(response)
                        
                        # Opcional: mostrar si el LLM confirmó o descartó la técnica
                        estado = "CONFIRMADA" if response.is_present else "DESCARTADA"
                        logger.info(f" -> Técnica {inp['mitre_technique_id']} {estado}.")
                except Exception as e:
                    logger.error(f"Error procesando la técnica {inp['mitre_technique_id']}: {e}")
                    
        # Filtramos para devolver ÚNICAMENTE las detecciones marcadas como verdaderas por el LLM
        confirmed_ttps = [res for res in results if res.is_present]
        
        logger.info(f"Inferencia finalizada. {len(confirmed_ttps)} técnicas confirmadas.")
        return confirmed_ttps
