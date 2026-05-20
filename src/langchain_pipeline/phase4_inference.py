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
            "Actúa como analista CTI. Tu objetivo es confirmar si los chunks demuestran el uso de la técnica."
        )
        
        user_message = (
            "### CONTEXTO MITRE\n"
            "{mitre_technique_id} - {mitre_description}\n\n"
            "### EVIDENCIA DEL REPORTE\n"
            "{supporting_chunks}\n\n"
        )
        
        # Aplicar el patrón "Prompt Repetition" para mitigar alucinaciones / distracciones del LLM
        if self.use_prompt_repetition:
            user_message += (
                "Let me repeat that: Basándote EXCLUSIVAMENTE en la evidencia proporcionada, "
                "¿se confirma el uso de la técnica {mitre_technique_id}? Devuelve el JSON estructurado."
            )
        else:
            user_message += (
                "Basándote EXCLUSIVAMENTE en la evidencia proporcionada, "
                "¿se confirma el uso de la técnica {mitre_technique_id}? Devuelve el JSON estructurado."
            )
            
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
                "mitre_description": data.get("name", "Desconocida"),
                "supporting_chunks": joined_chunks
            })
            
        logger.info(f"Preparados {len(inputs)} candidatos para inferencia LLM.")
        
        results: List[TTPDetection] = []
        
        # Orquestación de la ejecución basada en Hardware/Perfil
        if self.execution_profile == "AWS":
            logger.info("Perfil AWS detectado: Ejecutando cadena en paralelo (Batching)...")
            try:
                # LLMs grandes en la nube o clusters GPU aguantan batching
                batch_responses = chain.batch(inputs)
                
                # Descartar nulos por si hubo fallos de red/parseo
                results.extend([res for res in batch_responses if res is not None])
            except Exception as e:
                logger.error(f"Error durante el batching en AWS: {e}")
                
        else:
            logger.info("Perfil LOCAL detectado: Ejecutando cadena de forma secuencial (For-loop)...")
            # Para evitar OOM en Ollama o hardware modesto
            for inp in inputs:
                try:
                    response = chain.invoke(inp)
                    if response:
                        results.append(response)
                except Exception as e:
                    logger.error(f"Error procesando la técnica {inp['mitre_technique_id']}: {e}")
                    
        # Filtramos para devolver ÚNICAMENTE las detecciones marcadas como verdaderas por el LLM
        confirmed_ttps = [res for res in results if res.is_present]
        
        logger.info(f"Inferencia finalizada. {len(confirmed_ttps)} técnicas confirmadas.")
        return confirmed_ttps
