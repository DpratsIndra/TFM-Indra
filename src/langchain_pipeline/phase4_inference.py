import os
import logging
import time
from typing import Dict, List, Any
from src.core.schemas import ChunkExtraction, TTPDetection, Evidence

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
import json


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
            "7. CRITICAL JSON FORMATTING: In the 'procedure' field, you MUST write a full descriptive sentence of what the attacker did (e.g., 'The attacker used Ngrok to establish a tunnel.'). DO NOT write single words or tactic names.\n"
            "8. {format_instructions}"
        )
        if self.use_prompt_repetition:
            system_instruction = (
                f"{system_instruction}\n\nReminder of Rules:\n{system_instruction}"
            )

        user_message = (
            "<candidate_techniques>\n{candidates_str}\n</candidate_techniques>\n\n"
            "<chunk_text>\n{chunk_text}\n</chunk_text>\n\n"
            "Based exclusively on the chunk text, extract the techniques from the candidates that are confirmed. You MUST output ONLY a valid JSON object."
        )
        if self.use_prompt_repetition:
            user_message = f"{user_message}\n\n{user_message}"

        return ChatPromptTemplate.from_messages(
            [("system", system_instruction), ("user", user_message)]
        )

    def analyze_candidates(self, chunk_results: List[Dict[str, Any]], cache_key: str = None):
        if not chunk_results:
            logger.warning("No candidates to analyze.")
            return [], 0.0

        parser = PydanticOutputParser(pydantic_object=ChunkExtraction)
        prompt = self._build_prompt_template()
        chain = prompt | self.llm

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
                    "format_instructions": parser.get_format_instructions(),
                    "_location": f"Page {page_num}, Chunk {chunk_idx}",
                    "_meta_map": tech_metadata_map,
                }
            )

        logger.info(f"Prepared {len(inputs)} chunks for LLM inference (Chunk-by-Chunk).")
        batch_responses = [None] * len(inputs)
        artificial_delay = 0.0
        max_workers = int(os.getenv("MAX_CONCURRENT_CHUNKS", "2"))
        is_gemini = "google" in str(type(self.llm)).lower()

        import hashlib
        cache_path = None
        cached_responses = {}
        if cache_key:
            file_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_path = os.path.join("data", "output", "evaluations", f"cache_langchain_{file_hash}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_responses = json.load(f)
                    logger.info(f"Loaded {len(cached_responses)} chunks from LangChain cache.")
                except Exception:
                    pass

        import threading
        cache_lock = threading.Lock()
        def save_cache():
            if cache_path:
                with cache_lock:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cached_responses, f, indent=4)

        def print_retry_warning(retry_state):
            import sys
            print(f"\n[⚠️ ALARMA DE API / RATE LIMIT] El servidor LLM ha rechazado la petición o ha fallado.", file=sys.stderr)
            print(f"[⏳] Tenacity esperando {retry_state.next_action.sleep} segundos antes del reintento #{retry_state.attempt_number}...", file=sys.stderr)

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=2, min=5, max=15),
            before_sleep=print_retry_warning,
            reraise=True,
        )
        def _invoke_chain(inp):
            # PROACTIVE RATE LIMITING: Google AI Studio allows 15 RPM (1 request every 4 seconds)
            if is_gemini:
                time.sleep(4.5)
            try:
                msg = chain.invoke(inp)
                # Serialize AIMessage to plain dict
                content = msg.content if hasattr(msg, 'content') else str(msg)
                usage = getattr(msg, "usage_metadata", {})
                return {"content": content, "usage_metadata": usage}
            except Exception as e:
                # If it's a structural failure not caught by tenacity, or if it raises
                raise RuntimeError(f"Rate Limit / API Collapse: {e}") from e

        stop_event = threading.Event()

        def _process_chunk(idx, inp):
            if stop_event.is_set():
                return Exception("Aborted due to prior API crash")
            if str(idx) in cached_responses:
                logger.info(f"[{idx+1}/{len(inputs)}] Loading Chunk {inp['_location']} from cache...")
                return cached_responses[str(idx)]

            logger.info(f"[{idx+1}/{len(inputs)}] Querying LLM for Chunk {inp['_location']}...")
            try:
                res = _invoke_chain(inp)
                if cache_path:
                    with cache_lock:
                        cached_responses[str(idx)] = res
                    save_cache()
                return res
            except Exception as e:
                err_str = str(e).lower()
                api_errors = ["429", "quota", "resourceexhausted", "503", "500", "timeout", "not_found", "api", "connection", "unavailable", "rate limit"]
                if any(err in err_str for err in api_errors):
                    stop_event.set()
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
        total_input_tokens = 0
        total_output_tokens = 0
        
        api_crashed_flag = False
        for inp, res_dict in zip(inputs, batch_responses):
            if isinstance(res_dict, Exception):
                err_str = str(res_dict).lower()
                api_errors = ["429", "quota", "resourceexhausted", "503", "500", "timeout", "not_found", "api", "connection", "unavailable", "rate limit"]
                if any(err in err_str for err in api_errors):
                    if not api_crashed_flag:
                        logger.error(f"[!] ABORTANDO: Fallo de API/Rate Limit detectado en chunk {inp.get('_location')}. Guardando progreso parcial...")
                        api_crashed_flag = True
                    continue
                else:
                    logger.error(f"No TTPs extracted for chunk {inp.get('_location')} due to critical failure: {res_dict}")
                    continue

            # res_dict is now a dict containing content and usage_metadata
            raw_msg = res_dict
            if raw_msg and raw_msg.get("usage_metadata"):
                total_input_tokens += raw_msg["usage_metadata"].get("input_tokens", 0)
                total_output_tokens += raw_msg["usage_metadata"].get("output_tokens", 0)

            if not raw_msg or "content" not in raw_msg:
                logger.warning(f"  [!] LLM devolvió respuesta vacía en {inp['_location']}.")
                continue

            content = raw_msg["content"]
            if isinstance(content, list):
                # Handle Gemini returning a list of dicts
                text_parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
                content = "".join(text_parts) if text_parts else str(content)
            elif not isinstance(content, str):
                content = str(content)

            try:
                res = parser.parse(content)
            except Exception as e:
                logger.warning(f"  [!] Fallo de parseo JSON en {inp['_location']}: {e}")
                continue

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
            f"Inference complete. {len(final_ttps_limpios)} global techniques confirmed. Tokens: IN={total_input_tokens}, OUT={total_output_tokens}"
        )
        return final_ttps_limpios, artificial_delay, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens, "api_crashed": api_crashed_flag}
