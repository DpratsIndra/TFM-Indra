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
    Handles Phase 4: LLM Inference and Mapping.
    Uses LangChain Expression Language (LCEL) to prompt the model and enforce a structured output.
    Adapts execution to sequential or batched depending on the environment profile (LOCAL vs AWS).
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        Initializes the TTP Analyzer.
        
        Args:
            llm: A configured LangChain ChatModel (e.g., ChatOllama).
        """
        self.llm = llm
        
        # Determine environment settings
        self.use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
        self.execution_profile = os.getenv("EXECUTION_PROFILE", os.getenv("ENVIRONMENT_PROFILE", "LOCAL")).upper()
        
        logger.info(f"TTPAnalyzer started. Profile: {self.execution_profile}, Prompt Repetition: {self.use_prompt_repetition}")

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """
        Builds the prompt template dynamically.
        Injects 'Prompt Repetition' if enabled to reinforce instructions for smaller models.
        """
        system_instruction = (
            "Role: Cyber Threat Intelligence (CTI) Analyst.\n"
            "Task: Verify the presence of specific MITRE ATT&CK techniques based on provided report excerpts.\n\n"
            "Rules:\n"
            "1. Evaluate all evidence blocks objectively.\n"
            "2. Set `is_present` to true only if the evidence technically confirms the technique. Add each confirming block to the `occurrences` list.\n"
            "3. If no blocks confirm the technique, set `is_present` to false and leave `occurrences` empty.\n"
            "4. Preserve the exact `location` and `Score` provided in the evidence tags.\n"
            "5. Map contextualized masked tags (e.g., <IoC_URL>) to their tactical intent.\n"
            "6. Exclusion Criteria: Strictly map observable, technical intrusion activity. Exclude geopolitical context, analyst attribution theories, victimology, or historical background."
        )
        
        if self.use_prompt_repetition:
            system_instruction = f"{system_instruction}\n\nReminder of Rules:\n{system_instruction}"
            
        user_message = (
            "<mitre_context>\n"
            "ID: {mitre_technique_id}\n"
            "Name: {mitre_technique_name}\n"
            "Tactics: {mitre_tactics}\n"
            "Description: {mitre_description}\n"
            "</mitre_context>\n\n"
            "<evidence>\n"
            "{supporting_chunks}\n"
            "</evidence>\n\n"
            "Based exclusively on the provided evidence, return the structured JSON confirming or discarding the technique."
        )
        
        if self.use_prompt_repetition:
            user_message = f"{user_message}\n\n{user_message}"
            
        return ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", user_message)
        ])

    def analyze_candidates(self, candidates_dict: Dict[str, Dict[str, Any]]) -> List[TTPDetection]:
        """
        Runs the LLM chain to analyze the retrieved MITRE candidates.
        Uses batching in AWS or sequential execution locally to avoid Out-Of-Memory (OOM) issues.
        """
        if not candidates_dict:
            logger.warning("No candidates to analyze.")
            return []

        prompt = self._build_prompt_template()
        
        # Build the LCEL chain with forced structured output
        chain = prompt | self.llm.with_structured_output(TTPDetection)
        
        inputs = []
        for tech_id, data in candidates_dict.items():
            # Keep only the top 5 highest scored chunks to avoid LLM context limits
            top_chunks = sorted(data.get("supporting_chunks", []), key=lambda x: x.get("score", 0.0), reverse=True)[:5]
            
            formatted_chunks = []
            for idx, chunk_data in enumerate(top_chunks, 1):
                loc = chunk_data.get("location", "Unknown")
                score = chunk_data.get("score", 0.0)
                txt = chunk_data.get("text", "")
                formatted_chunks.append(f"[Evidence {idx} | Location: {loc} | Score: {score}]\n{txt}\n")
                
            joined_chunks = "\n---\n".join(formatted_chunks)
            
            inputs.append({
                "mitre_technique_id": tech_id,
                "mitre_technique_name": data.get("name", "Unknown"),
                "mitre_tactics": ", ".join(data.get("tactics", [])),
                "mitre_description": data.get("description", "No description available."),
                "supporting_chunks": joined_chunks,
                "_meta_tactics": data.get("tactics", [])
            })
            
        logger.info(f"Prepared {len(inputs)} candidates for LLM inference.")
        
        results: List[TTPDetection] = []
        
        if self.execution_profile == "AWS":
            logger.info("AWS Profile detected: Running chain in parallel (Batching)...")
            try:
                batch_responses = chain.batch(inputs)
                
                # Filter out nulls (failed parsing) and populate metadata
                for inp, res in zip(inputs, batch_responses):
                    if res is not None:
                        res.tactic = inp["_meta_tactics"]
                        res.technique_name = inp["mitre_technique_name"]
                        results.append(res)
            except Exception as e:
                logger.error(f"Error during AWS batching: {e}")
                
        else:
            logger.info("LOCAL Profile detected: Running chain sequentially...")
            import time
            is_gemini = "google" in str(type(self.llm)).lower()

            for i, inp in enumerate(inputs, 1):
                try:
                    logger.info(f"[{i}/{len(inputs)}] Querying LLM for technique: {inp['mitre_technique_id']}...")
                    response = chain.invoke(inp)
                    if response:
                        response.tactic = inp["_meta_tactics"]
                        response.technique_name = inp["mitre_technique_name"]
                        results.append(response)
                        
                        status = "CONFIRMED" if response.is_present else "DISCARDED"
                        logger.info(f" -> Technique {inp['mitre_technique_id']} {status}.")
                        
                    # Rate limit estricto para el Free Tier de Gemini (15 peticiones/minuto)
                    # 60s / 15 = 4s. Usamos 4.5s para tener margen de seguridad.
                    if is_gemini and i < len(inputs):
                        time.sleep(4.5)
                        
                except Exception as e:
                    logger.error(f"Error processing technique {inp['mitre_technique_id']}: {e}")
                    
        # Filter and return ONLY confirmed detections
        confirmed_ttps = [res for res in results if res.is_present]
        
        logger.info(f"Inference complete. {len(confirmed_ttps)} techniques confirmed.")
        return confirmed_ttps
