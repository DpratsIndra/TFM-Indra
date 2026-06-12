import os
import logging
import time
from typing import Dict, List, Any
from src.core.schemas import ChunkExtraction, TTPDetection, Evidence

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


logger = logging.getLogger(__name__)

from tenacity import retry, stop_after_attempt, wait_exponential
from src.core.schemas import ChunkExtraction, TTPDetection, Evidence
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
        self.use_prompt_repetition = os.getenv(
            "USE_PROMPT_REPETITION", "False"
        ).lower() in ("true", "1", "yes")
        self.execution_profile = os.getenv(
            "EXECUTION_PROFILE", os.getenv("ENVIRONMENT_PROFILE", "LOCAL")
        ).upper()

        logger.info(
            f"TTPAnalyzer started. Profile: {self.execution_profile}, Prompt Repetition: {self.use_prompt_repetition}"
        )

    def _build_prompt_template(self) -> ChatPromptTemplate:
        system_instruction = (
            "Role: Cyber Threat Intelligence (CTI) Analyst.\n"
            "Task: Extract confirmed MITRE ATT&CK techniques from a report chunk based ONLY on the provided candidate techniques.\n\n"
            "Rules:\n"
            "0. Multilingual Support: The source text may be in any language (Spanish, Russian, Chinese, etc.). Analyze it natively, but output the requested JSON schema in English.\n"
            "1. Evaluate all evidence blocks objectively.\n"
            "2. Only extract techniques that are unequivocally confirmed by the technical evidence.\n"
            "3. Do not include techniques that are false positives, benign, or defensive mentions. If none are valid, return an empty list.\n"
            "4. Preserve the exact `location` and `Score` provided in the evidence tags.\n"
            "5. Map contextualized masked tags (e.g., <IoC_URL>) to their tactical intent.\n"
            "6. Exclusion Criteria: Strictly map observable, technical intrusion activity. Exclude geopolitical context or attribution theories.\n"
            "7. CRITICAL JSON FORMATTING: In the 'procedure' field, you MUST write a full descriptive sentence of what the attacker did (e.g., 'The attacker used Ngrok to establish a tunnel.'). DO NOT write single words or tactic names."
        )
        if self.use_prompt_repetition:
            system_instruction = (
                f"{system_instruction}\n\nReminder of Rules:\n{system_instruction}"
            )

        user_message = (
            "<candidate_techniques>\n{candidates_str}\n</candidate_techniques>\n\n"
            "<chunk_text>\n{chunk_text}\n</chunk_text>\n\n"
            "Based exclusively on the chunk text, extract the techniques from the candidates that are confirmed."
        )
        if self.use_prompt_repetition:
            user_message = f"{user_message}\n\n{user_message}"

        return ChatPromptTemplate.from_messages(
            [("system", system_instruction), ("user", user_message)]
        )

    def analyze_candidates(self, chunk_results: List[Dict[str, Any]]):
        if not chunk_results:
            logger.warning("No candidates to analyze.")
            return [], 0.0

        prompt = self._build_prompt_template()
        chain = prompt | self.llm.with_structured_output(ChunkExtraction)

        inputs = []
        for data in chunk_results:
            chunk = data["chunk"]
            candidates = data["candidates"]

            cand_strings = []
            tech_metadata_map = {}
            for c in candidates:
                tech_id = str(c.get("technique_id", "Unknown")).strip().upper()
                cand_strings.append(
                    f"ID: {tech_id} | Name: {c['name']} | Tactics: {', '.join(c['tactics'])}\nDescription: {c['description']}"
                )
                tech_metadata_map[tech_id] = {
                    "tactics": c["tactics"],
                    "score": c["score"],
                    "name": c["name"],
                }

            page_val = chunk.metadata.get("page")
            page_num = (
                int(page_val) + 1
                if page_val is not None
                else chunk.metadata.get("page_number", "Unknown")
            )
            chunk_idx = chunk.metadata.get("chunk_index", "Unknown")

            inputs.append(
                {
                    "chunk_text": chunk.page_content,
                    "candidates_str": "\n\n".join(cand_strings),
                    "_location": f"Page {page_num}, Chunk {chunk_idx}",
                    "_meta_map": tech_metadata_map,
                }
            )

        logger.info(f"Prepared {len(inputs)} chunks for LLM inference (Chunk-by-Chunk).")
        batch_responses = [None] * len(inputs)
        artificial_delay = 0.0
        max_workers = int(os.getenv("MAX_CONCURRENT_CHUNKS", "2"))
        is_gemini = "google" in str(type(self.llm)).lower()

        @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
        def _invoke_chain(inp):
            return chain.invoke(inp)

        def _process_chunk(idx, inp):
            logger.info(f"[{idx+1}/{len(inputs)}] Querying LLM for Chunk {inp['_location']}...")
            try:
                res = _invoke_chain(inp)
                return res
            except Exception as e:
                logger.error(f"Error processing chunk {inp['_location']} after retries: {e}")
                return e

        import concurrent.futures

        if max_workers > 1:
            logger.info(f"Running LangChain inference concurrently (max_workers={max_workers})...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_chunk, i, inp) for i, inp in enumerate(inputs)]
                for i, future in enumerate(futures):
                    batch_responses[i] = future.result()
        else:
            logger.info("Running LangChain sequentially (max_workers=1)...")
            for i, inp in enumerate(inputs):
                batch_responses[i] = _process_chunk(i, inp)
                if is_gemini and i < len(inputs) - 1:
                    time.sleep(4.5)
                    artificial_delay += 4.5

        # Reducer: Consolidar ChunkExtraction de vuelta a List[TTPDetection]
        global_ttps = {}
        for inp, res in zip(inputs, batch_responses):
            # AÑADIDO: Controlar si el resultado fue una excepción (ej. Timeout)
            if isinstance(res, Exception):
                logger.error(f"  [!] Fallo/Timeout crítico en chunk {inp['_location']}: {res}")
                raise RuntimeError(f"Ejecución abortada por fallo en chunk {inp['_location']}: {res}") from res

            if not res or not hasattr(res, "extracted_ttps") or not res.extracted_ttps:
                continue

            for ext in res.extracted_ttps:
                tech_id = str(ext.technique_id).strip().upper()

                # ANTI-HALLUCINATION GUARD: Reject if LLM bypassed RAG and invented an ID
                if tech_id not in inp["_meta_map"]:
                    logger.warning(
                        f"Descartando TTP alucinado/inventado por el LLM: {tech_id}"
                    )
                    continue

                meta = inp["_meta_map"][tech_id]

                if tech_id not in global_ttps:
                    global_ttps[tech_id] = TTPDetection(
                        technique_id=tech_id,
                        tactic=meta["tactics"],
                        technique_name=meta["name"],
                        is_present=True,
                        occurrences=[],
                    )

                global_ttps[tech_id].occurrences.append(
                    Evidence(
                        location=inp["_location"],
                        procedure=ext.procedure,
                        justification=ext.justification,
                        confidence_score=meta["score"],
                    )
                )

        final_ttps = list(global_ttps.values())

        # Limpieza de redundancia jerárquica (padres e hijos)
        ids_presentes = [t.technique_id for t in final_ttps]
        final_ttps_limpios = []
        for ttp in final_ttps:
            if "." not in ttp.technique_id and any(
                child.startswith(f"{ttp.technique_id}.") for child in ids_presentes
            ):
                logger.info(
                    f"Descartando TTP padre {ttp.technique_id} porque existe una sub-técnica más precisa."
                )
                continue
            final_ttps_limpios.append(ttp)

        logger.info(
            f"Inference complete. {len(final_ttps_limpios)} global techniques confirmed."
        )
        return final_ttps_limpios, artificial_delay
