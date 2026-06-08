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
        system_instruction = (
            "Role: Cyber Threat Intelligence (CTI) Analyst.\n"
            "Task: Extract confirmed MITRE ATT&CK techniques from a report chunk based ONLY on the provided candidate techniques.\n\n"
            "Rules:\n"
            "0. Multilingual Support: The source text may be in any language (Spanish, Russian, Chinese, etc.). Analyze it natively, but output the requested JSON schema in English.\n"
            "1. Evaluate all evidence blocks objectively.\n"
            "2. Set `is_present` to true only if the evidence technically confirms the technique. Add each confirming block to the `occurrences` list.\n"
            "3. If no blocks confirm the technique, set `is_present` to false and leave `occurrences` empty.\n"
            "4. Preserve the exact `location` and `Score` provided in the evidence tags.\n"
            "5. Map contextualized masked tags (e.g., <IoC_URL>) to their tactical intent.\n"
            "6. Exclusion Criteria: Strictly map observable, technical intrusion activity. Exclude geopolitical context or attribution theories."
        )
        if self.use_prompt_repetition:
            system_instruction = f"{system_instruction}\n\nReminder of Rules:\n{system_instruction}"
            
        user_message = (
            "<candidate_techniques>\n{candidates_str}\n</candidate_techniques>\n\n"
            "<chunk_text>\n{chunk_text}\n</chunk_text>\n\n"
            "Based exclusively on the chunk text, extract the techniques from the candidates that are confirmed."
        )
        if self.use_prompt_repetition:
            user_message = f"{user_message}\n\n{user_message}"
            
        return ChatPromptTemplate.from_messages([("system", system_instruction), ("user", user_message)])

    def analyze_candidates(self, chunk_results: List[Dict[str, Any]]):
        if not chunk_results:
            logger.warning("No candidates to analyze.")
            return [], 0.0

        prompt = self._build_prompt_template()
        from src.core.schemas import ChunkExtraction, TTPDetection, Evidence
        chain = prompt | self.llm.with_structured_output(ChunkExtraction)
        
        inputs = []
        for data in chunk_results:
            chunk = data["chunk"]
            candidates = data["candidates"]
            
            cand_strings = []
            tech_metadata_map = {}
            for c in candidates:
                tech_id = c["technique_id"]
                cand_strings.append(f"ID: {tech_id} | Name: {c['name']} | Tactics: {', '.join(c['tactics'])}\nDescription: {c['description']}")
                tech_metadata_map[tech_id] = {"tactics": c["tactics"], "score": c["score"]}
                
            page_val = chunk.metadata.get("page")
            page_num = int(page_val) + 1 if page_val is not None else chunk.metadata.get("page_number", "Unknown")
            chunk_idx = chunk.metadata.get("chunk_index", "Unknown")
            
            inputs.append({
                "chunk_text": chunk.page_content,
                "candidates_str": "\n\n".join(cand_strings),
                "_location": f"Page {page_num}, Chunk {chunk_idx}",
                "_meta_map": tech_metadata_map
            })
            
        logger.info(f"Prepared {len(inputs)} chunks for LLM inference (Chunk-by-Chunk).")
        batch_responses = []
        artificial_delay = 0.0
        
        max_workers = int(os.getenv("MAX_CONCURRENT_CHUNKS", "2"))
        
        if self.execution_profile == "REMOTE" or max_workers > 1:
            logger.info(f"Parallel batching enabled: Running LangChain inference concurrently (max_concurrency={max_workers})...")
            try:
                batch_responses = chain.batch(inputs, config={"max_concurrency": max_workers})
            except Exception as e:
                logger.error(f"Error during parallel batching: {e}")
        else:
            logger.info("LOCAL Profile detected: Running chain sequentially...")
            import time
            is_gemini = "google" in str(type(self.llm)).lower()
            for i, inp in enumerate(inputs, 1):
                try:
                    logger.info(f"[{i}/{len(inputs)}] Querying LLM for Chunk {inp['_location']}...")
                    res = chain.invoke(inp)
                    batch_responses.append(res)
                    
                    if is_gemini and i < len(inputs):
                        sleep_time = 4.5
                        time.sleep(sleep_time)
                        artificial_delay += sleep_time
                except Exception as e:
                    logger.error(f"Error processing chunk {inp['_location']}: {e}")
                    batch_responses.append(None)
                    
        # Reducer: Consolidar ChunkExtraction de vuelta a List[TTPDetection]
        global_ttps = {}
        for inp, res in zip(inputs, batch_responses):
            if not res or not hasattr(res, 'extracted_ttps') or not res.extracted_ttps:
                continue
                
            for ext in res.extracted_ttps:
                tech_id = ext.technique_id
                meta = inp["_meta_map"].get(tech_id, {"tactics": [], "score": 0.99})
                
                if tech_id not in global_ttps:
                    global_ttps[tech_id] = TTPDetection(
                        technique_id=tech_id,
                        tactic=meta["tactics"],
                        technique_name=ext.technique_name,
                        is_present=True,
                        occurrences=[]
                    )
                    
                global_ttps[tech_id].occurrences.append(Evidence(
                    location=inp["_location"],
                    procedure=ext.procedure,
                    justification=ext.justification,
                    confidence_score=meta["score"]
                ))
                
        final_ttps = list(global_ttps.values())
        logger.info(f"Inference complete. {len(final_ttps)} global techniques confirmed.")
        return final_ttps, artificial_delay
