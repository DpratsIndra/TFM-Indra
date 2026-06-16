from typing import List
from pydantic import BaseModel, Field

import os
import json
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import ChunkState
from .tools import get_mitre_candidates, load_mitre_json
from src.core.llm_factory import get_llm as factory_get_llm

# ==============================================================================
# LLM FACTORY
# ==============================================================================


def get_llm(tier="pro", temperature=0.0):
    return factory_get_llm(temperature=temperature)


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================


class TriageDecision(BaseModel):
    is_relevant: bool = Field(
        description="True if the text contains any cyber threat intelligence, attacker behavior, or technical indicators. False otherwise."
    )


class DraftTTP(BaseModel):
    technique_id: str = Field(
        description="The MITRE ATT&CK Technique ID (e.g., T1059.001)"
    )
    tactic: List[str] = Field(
        default_factory=list,
        description="List of associated MITRE tactics (e.g. ['initial-access', 'execution'])",
    )
    name: str = Field(description="The name of the MITRE ATT&CK Technique")
    procedure: str = Field(
        description="The specific procedure or behavior observed in the text"
    )
    justification: str = Field(
        default="Verified by Validator",
        description="Technical explanation of why this chunk demonstrates the technique.",
    )
    mitre_description: str = Field(
        default="",
        description="The official MITRE ATT&CK description for this technique provided in the candidates",
    )
    confidence_score: float = Field(
        description="The numeric score provided in the Candidate Techniques list."
    )
    location: str = Field(
        default="Unknown",
        description="The document page and chunk location of the evidence.",
    )


class DraftTTPList(BaseModel):
    ttps: List[DraftTTP] = Field(description="A list of drafted TTPs from the text.")


class ValidationResult(BaseModel):
    valid_ttps: List[DraftTTP] = Field(
        description="A list of approved TTPs that strictly match the text."
    )
    feedback_notes: List[str] = Field(
        description="A list of feedback notes explaining what was hallucinated or missing."
    )

class SimpleDraftTTP(BaseModel):
    technique_id: str = Field(
        description="The MITRE ATT&CK Technique ID (e.g., T1059.001)"
    )
    procedure: str = Field(
        description="The specific procedure or behavior observed in the text"
    )
    justification: str = Field(
        default="Verified by Validator",
        description="Technical explanation of why this chunk demonstrates the technique.",
    )

class SimpleDraftTTPList(BaseModel):
    ttps: List[SimpleDraftTTP] = Field(description="A list of drafted TTPs from the text.")

class SimpleValidationResult(BaseModel):
    valid_ttps: List[SimpleDraftTTP] = Field(
        description="A list of approved TTPs that strictly match the text."
    )
    feedback_notes: List[str] = Field(
        description="A list of feedback notes explaining what was hallucinated or missing."
    )


# ==============================================================================
# NODE FUNCTIONS
# ==============================================================================


from tenacity import retry, stop_after_attempt, wait_exponential

def print_retry_warning(retry_state):
    import sys
    import os
    profile = os.getenv("EXECUTION_PROFILE", "LOCAL").upper()
    provider = "Google AI Studio" if profile == "LOCAL" else "vLLM Remote"
    print(f"\n[⚠️ ALARMA DE CONEXIÓN] {provider} ha rechazado o fallado en la petición.", file=sys.stderr)
    print(f"[⏳] Tenacity esperando {retry_state.next_action.sleep} segundos antes del reintento #{retry_state.attempt_number}...", file=sys.stderr)

@retry(
    stop=stop_after_attempt(4), 
    wait=wait_exponential(multiplier=2, min=5, max=15),
    before_sleep=print_retry_warning,
    reraise=True
)
def _invoke_with_retry(callable_chain, input_data):
    return callable_chain.invoke(input_data)

def safe_invoke(callable_chain, input_data):
    """Safely invokes a chain or agent with exponential backoff. Raises exception if all retries fail."""
    # PROACTIVE RATE LIMITING: Google AI Studio allows 15 RPM (1 request every 4 seconds)
    # Solo aplicamos el sleep si estamos en el perfil LOCAL (Gemini API)
    import os
    if os.getenv("EXECUTION_PROFILE", "LOCAL").upper() == "LOCAL":
        time.sleep(4.5)
    
    try:
        return _invoke_with_retry(callable_chain, input_data)
    except Exception as e:
        print(f"[ERROR CRÍTICO] Model Invocation Error after all retries exhausted: {e}")
        raise RuntimeError(f"Rate Limit / API Collapse: {e}") from e


def triage_node(state: ChunkState) -> dict:
    """
    Filtro rápido.
    Lee el chunk y decide si contiene Inteligencia de Amenazas (TTPs, atacantes)
    o si es relleno. Si es irrelevante, corta la ejecución.
    """
    t0 = time.time()
    chunk_text = state["chunk_text"]

    llm = get_llm(tier="lite")
    structured_llm = llm.with_structured_output(TriageDecision, include_raw=True)

    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in (
        "true",
        "1",
        "yes",
    )

    sys_prompt = (
        "Role: Cyber Threat Intelligence Analyst.\n"
        "Task: Perform a binary classification on text chunks.\n"
        "Condition: Return True if the text contains cybersecurity threat intelligence, attacker behavior, or technical indicators. Return False otherwise.\n"
        "Note: The text may be in any language (Spanish, Russian, etc.). Analyze it natively."
    )
    if use_prompt_repetition:
        sys_prompt = f"{sys_prompt}\n\nReminder of Rules:\n{sys_prompt}"

    user_prompt = "<text>\n{chunk_text}\n</text>"
    if use_prompt_repetition:
        user_prompt = f"{user_prompt}\n\n{user_prompt}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            ("human", user_prompt),
        ]
    )

    chain = prompt | structured_llm

    result = safe_invoke(chain, {"chunk_text": chunk_text})

    in_tok = 0
    out_tok = 0
    
    if result is None:
        is_relevant = True
    else:
        raw_msg = result.get("raw")
        if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
            in_tok += raw_msg.usage_metadata.get("input_tokens", 0)
            out_tok += raw_msg.usage_metadata.get("output_tokens", 0)
        
        parsed = result.get("parsed")
        is_relevant = parsed.is_relevant if parsed else True

    c_idx = state.get("chunk_metadata", {}).get("chunk_index", "?")
    print(f"[DEBUG] [Chunk {c_idx}] Triage - Is Relevant: {is_relevant}")
    return {
        "is_relevant": is_relevant, 
        "triage_time": time.time() - t0,
        "input_tokens": state.get("input_tokens", 0) + in_tok,
        "output_tokens": state.get("output_tokens", 0) + out_tok,
    }


def extractor_node(state: ChunkState) -> dict:
    """
    Analiza el chunk, extrae comportamientos tácticos,
    consulta a la base de datos vectorial para anclar el comportamiento a
    una técnica oficial de MITRE ATT&CK, y genera un borrador del TTP.
    """
    t0 = time.time()
    chunk_text = state["chunk_text"]
    val_feedback = state.get("validation_feedback", "")
    approved_ttps = state.get("approved_ttps", [])
    approved_ids = [t.get("technique_id") for t in approved_ttps]
    
    in_tok = 0
    out_tok = 0

    # 1. QUERY TRANSFORMATION: Abstract Keywords Only
    c_idx = state.get("chunk_metadata", {}).get("chunk_index", "?")
    if state.get("abstract_keywords") and not val_feedback:
        abstract_keywords = state["abstract_keywords"]
        print(
            f"[DEBUG] [Chunk {c_idx}] Extractor - Abstract Keywords (Cached): {abstract_keywords}"
        )
    else:
        translator_llm = get_llm(tier="lite")
        sys_prompt = (
            "Role: Cyber Threat Intelligence Analyst.\n"
            "Task: Extract a comma-separated list of abstract cybersecurity behaviors, tactics, and mechanisms from the text.\n"
            "Instructions: Translate specific actions into their tactical purpose, BUT ALWAYS preserve specific technology and tool names (e.g., PowerShell, Ngrok, WinRAR, WMI). Output ONLY the keywords in English.\n\n"
            "EXAMPLE:\n"
            "Source Text: 'The actors used a PowerShell script to download a payload called invoice.pdf.'\n"
            "Output: Spearphishing Attachment, PowerShell, Malicious File, Ingress Tool Transfer, Script Execution\n"
        )

        if val_feedback:
            sys_prompt += (
                "\n\nCRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:\n"
                "The previous extraction failed or was incomplete. Review the validator's feedback and generate NEW, DIFFERENT abstract keywords to find better techniques.\n"
                f"Feedback: {val_feedback}"
            )

        use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in (
            "true",
            "1",
            "yes",
        )
        if use_prompt_repetition:
            sys_prompt = f"{sys_prompt}\n\nReminder of Rules:\n{sys_prompt}"

        user_prompt = "<text>\n{chunk_text}\n</text>"
        if use_prompt_repetition:
            user_prompt = f"{user_prompt}\n\n{user_prompt}"

        translator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt),
                ("human", user_prompt),
            ]
        )

        translation_chain = translator_prompt | translator_llm | StrOutputParser()

        abstract_keywords_result = safe_invoke(translation_chain, {"chunk_text": chunk_text})
        if abstract_keywords_result is None:
            print(f"[ERROR] [Chunk {c_idx}] Translation Error: LLM failed after retries.")
            abstract_keywords = ""
        else:
            if isinstance(abstract_keywords_result, dict):
                raw_msg = abstract_keywords_result.get("raw")
                if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
                    in_tok += raw_msg.usage_metadata.get("input_tokens", 0)
                    out_tok += raw_msg.usage_metadata.get("output_tokens", 0)
                abstract_keywords = abstract_keywords_result.get("parsed", "")
            elif hasattr(abstract_keywords_result, 'response_metadata') and 'token_usage' in abstract_keywords_result.response_metadata:
                usage = abstract_keywords_result.response_metadata['token_usage']
                in_tok += usage.get('prompt_tokens', 0)
                out_tok += usage.get('completion_tokens', 0)
                abstract_keywords = abstract_keywords_result.content
            else:
                abstract_keywords = abstract_keywords_result

        print(
            f"[DEBUG] [Chunk {c_idx}] Extractor - Abstract Keywords: {abstract_keywords}"
        )

    # 2. DYNAMIC MECHANICAL RETRIEVAL
    # Pass the RAW TEXT plus the keywords to the Oracle so no context is lost.
    # We DO NOT include val_feedback in the vector query to avoid polluting the semantic search.
    search_query = f"Raw Text: {chunk_text}\nKeywords: {abstract_keywords}"

    # Expanding Window mechanism: Use Pagination
    # Iter 0 -> Offset 0 (Top 1-25)
    # Iter 1 -> Offset 25 (Top 26-50)
    # Iter 2 -> Offset 50 (Top 51-75)
    state.get("loop_count", 0)

    current_meta_map = state.get("metadata_map", {})
    if state.get("candidates_list") and not val_feedback:
        candidates_list = state["candidates_list"]
        print(
            f"[DEBUG] [Chunk {c_idx}] Extractor - Using cached candidates_list ({len(candidates_list)} candidates)."
        )
    else:
        # Recuperamos la bolsa de 25
        candidates_list, meta_map = get_mitre_candidates(
            search_query, top_k=25
        )
        # Fusionamos con el metadata_map previo por si hay varias iteraciones
        current_meta_map.update(meta_map)

    if not candidates_list:
        print(
            f"[DEBUG] [Chunk {c_idx}] Extractor - No candidates found via Mechanical Retrieval. Returning empty drafts."
        )
        return {
            "draft_ttps": [],
            "metadata_map": current_meta_map,
            "loop_count": state.get("loop_count", 0) + 1,
            "extractor_time": state.get("extractor_time", 0.0) + (time.time() - t0),
        }

    # 3. EXTRACTION (BRAINSTORMER)
    llm = get_llm(tier="pro")
    
    use_simple_schema = os.getenv("SIMPLIFIED_JSON_SCHEMA", "False").lower() in ("true", "1", "yes")
    OutputSchema = SimpleDraftTTPList if use_simple_schema else DraftTTPList
    structured_llm = llm.with_structured_output(OutputSchema, include_raw=True)

    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in (
        "true",
        "1",
        "yes",
    )
    sys_prompt = (
        "Role: Cyber Threat Intelligence Analyst.\n"
        "Task: Extract applicable MITRE ATT&CK techniques from the provided candidate list based on the source text.\n\n"
        "Rules:\n"
        "1. Multilingual: The source text may be in any language, but write your justification in English.\n"
        "2. Extract all applicable new techniques. Do not artificially limit the output.\n"
        "3. Ignore previously approved or rejected techniques.\n"
        "4. Focus exclusively on observable actions.\n"
        "5. CRITICAL JSON FORMATTING: In the 'procedure' field, you MUST write a full descriptive sentence of what the attacker did (e.g., 'The attacker used Ngrok to establish a tunnel.'). DO NOT write single words or tactic names.\n"
        "6. Output strictly according to the requested JSON schema."
    )
    if use_prompt_repetition:
        sys_prompt = f"{sys_prompt}\n\nReminder of Rules:\n{sys_prompt}"

    prompt = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("human", "{prompt_text}")]
    )

    all_drafts = []
    candidates_str = "CANDIDATE TECHNIQUES:\n" + "\n\n".join(candidates_list)

    prompt_text = f"Global Context: {state.get('global_context', '')}\n\n"

    if approved_ids:
        prompt_text += f"ALREADY APPROVED TTPs FOR THIS CHUNK: {approved_ids}\n"
        prompt_text += "Do NOT propose these exact technique IDs again. Search the text for OTHER distinct malicious behaviors.\n"
        prompt_text += (
            "If you cannot find any NEW behaviors, you MUST return an empty list.\n\n"
        )

    if val_feedback:
        prompt_text += f"PAST VALIDATOR REJECTIONS (CRITICAL: DO NOT PROPOSE THESE REJECTED TTPs AGAIN):\n{val_feedback}\n\n"

    prompt_text += f"Candidate MITRE Techniques:\n{candidates_str}\n\n"
    prompt_text += f"Source Text:\n{chunk_text}"

    if use_prompt_repetition:
        prompt_text = f"{prompt_text}\n\n{prompt_text}"

    parsed_result_dict = safe_invoke(prompt | structured_llm, {"prompt_text": prompt_text})

    if parsed_result_dict is not None:
        raw_msg = parsed_result_dict.get("raw")
        if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
            in_tok += raw_msg.usage_metadata.get("input_tokens", 0)
            out_tok += raw_msg.usage_metadata.get("output_tokens", 0)
            
        parsed_result = parsed_result_dict.get("parsed")
        if parsed_result:
            loc = f"Page {state.get('chunk_metadata', {}).get('page_number', '?')}, Chunk {state.get('chunk_metadata', {}).get('chunk_index', '?')}"
            for ttp in parsed_result.ttps:
                draft = ttp.model_dump()
                tech_id = str(draft.get("technique_id", "")).strip().upper()

                # --- HYBRID GUARDIAN ---
                if tech_id not in current_meta_map:
                    # 1. El LLM ha propuesto un ID que no le dio Qdrant.
                    # Vamos a comprobar si es una alucinación (T9999) o si es real (T1203).
                    
                    # Cargamos el diccionario global de MITRE (está cacheado, es instantáneo)
                    global_mitre_db = load_mitre_json() 
                    
                    if tech_id in global_mitre_db:
                        print(f"[DEBUG] Model deduced {tech_id} natively. Accepted for validation.")
                        # Inyectamos la metadata oficial para que el Validator no se confunda
                        current_meta_map[tech_id] = {
                            "name": global_mitre_db[tech_id]["name"],
                            "tactics": global_mitre_db[tech_id]["tactics"],
                            "score": 0.0
                        }
                        draft["technique_id"] = tech_id
                        draft["name"] = global_mitre_db[tech_id]["name"]
                        draft["tactic"] = global_mitre_db[tech_id]["tactics"]
                        draft["confidence_score"] = 0.0 # Score 0.0 indica que viene de memoria, no de Qdrant
                        draft["location"] = loc
                        all_drafts.append(draft)
                    else:
                        print(f"[DEBUG] Discarding hallucination: {tech_id} not in MITRE.")
                    
                    continue # Saltamos a la siguiente iteración

                # 2. Si venía de Qdrant (Comportamiento normal)
                draft["technique_id"] = tech_id
                draft["name"] = current_meta_map[tech_id]["name"]
                draft["tactic"] = current_meta_map[tech_id]["tactics"]
                draft["confidence_score"] = current_meta_map[tech_id]["score"]
                draft["location"] = loc
                all_drafts.append(draft)

    draft_ids = [d.get("technique_id", "Unknown") for d in all_drafts]
    print(
        f"[DEBUG] [Chunk {c_idx}] Extractor - Draft TTPs found: {len(all_drafts)} {draft_ids}"
    )
    return {
        "draft_ttps": all_drafts,
        "metadata_map": dict(current_meta_map), # Force a new dict copy so LangGraph updates state properly
        "loop_count": state.get("loop_count", 0) + 1,
        "extractor_time": state.get("extractor_time", 0.0) + (time.time() - t0),
        "candidates_list": candidates_list,
        "abstract_keywords": abstract_keywords,
        "input_tokens": state.get("input_tokens", 0) + in_tok,
        "output_tokens": state.get("output_tokens", 0) + out_tok,
    }


def validator_node(state: ChunkState) -> dict:
    """
    Goal: Perform a 'Line-Item Veto'. Strict verification of drafted TTPs against the source text.
    """
    t0 = time.time()
    c_idx = state.get("chunk_metadata", {}).get("chunk_index", "?")

    draft_ttps_list = state.get("draft_ttps", [])
    if not draft_ttps_list:
        print(
            f"[DEBUG] [Chunk {c_idx}] Validator - No drafts to validate. Skipping LLM call."
        )
        return {
            "approved_ttps": [],
            "validation_feedback": "",
            "draft_ttps": [],
            "validator_time": state.get("validator_time", 0.0) + (time.time() - t0),
        }

    llm = get_llm(tier="pro", temperature=0.2)
    
    use_simple_schema = os.getenv("SIMPLIFIED_JSON_SCHEMA", "False").lower() in ("true", "1", "yes")
    ValidationSchema = SimpleValidationResult if use_simple_schema else ValidationResult
    structured_llm = llm.with_structured_output(ValidationSchema, include_raw=True)

    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in (
        "true",
        "1",
        "yes",
    )

    sys_prompt = (
        "Role: Cyber Threat Intelligence Analyst.\n"
        "Task: Validate drafted MITRE ATT&CK techniques against the source text.\n\n"
        "Rules:\n"
        "1. Multilingual: The source text may be in any language. Understand the native language but output your feedback in English.\n"
        "2. Read the source text and the official MITRE description for each drafted technique.\n"
        "3. Approve the technique if the text contains concrete evidence of the technical action described.\n"
        "4. Reject the technique if it is based solely on theoretical motives or implies the action did not occur.\n"
        "5. Provide specific feedback for rejected techniques to improve extraction accuracy.\n"
        "6. Validate each technique independently."
    )

    if use_prompt_repetition:
        sys_prompt = f"{sys_prompt}\n\nReminder of Rules:\n{sys_prompt}"

    user_prompt = "<context>\n{global_context}\n</context>\n\n<source_text>\n{chunk_text}\n</source_text>\n\n<drafted_techniques>\n{draft_ttps}\n</drafted_techniques>\n\nValidate these techniques and provide the approved list and feedback notes."

    if use_prompt_repetition:
        user_prompt = f"{user_prompt}\n\n{user_prompt}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            ("human", user_prompt),
        ]
    )

    chain = prompt | structured_llm

    result_dict = safe_invoke(
        chain,
        {
            "global_context": state.get("global_context", "None provided."),
            "chunk_text": state["chunk_text"],
            "draft_ttps": json.dumps(draft_ttps_list, indent=2),
        },
    )
    
    in_tok = 0
    out_tok = 0

    if result_dict is None:
        print(f"[DEBUG] [Chunk {c_idx}] Validator - Validation failed gracefully.")
        return {"approved_ttps": [], "validation_feedback": "", "draft_ttps": []}

    raw_msg = result_dict.get("raw")
    if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
        in_tok += raw_msg.usage_metadata.get("input_tokens", 0)
        out_tok += raw_msg.usage_metadata.get("output_tokens", 0)
        
    result = result_dict.get("parsed")

    current_meta_map = state.get("metadata_map", {})
    new_approved = []
    if result and result.valid_ttps:
        for ttp in result.valid_ttps:
            ttp_dict = ttp.model_dump()
            tech_id = str(ttp_dict["technique_id"]).strip().upper()
            if tech_id not in current_meta_map:
                print(
                    f"[DEBUG] [Chunk {c_idx}] Validator - Discarding hallucinated TTP: {tech_id}"
                )
                continue
            # Force metadata
            ttp_dict["technique_id"] = tech_id
            ttp_dict["name"] = current_meta_map[tech_id]["name"]
            ttp_dict["tactic"] = current_meta_map[tech_id]["tactics"]
            ttp_dict["confidence_score"] = current_meta_map[tech_id]["score"]
            ttp_dict["location"] = f"Page {state.get('chunk_metadata', {}).get('page_number', '?')}, Chunk {state.get('chunk_metadata', {}).get('chunk_index', '?')}"
            new_approved.append(ttp_dict)

    # Set feedback to current loop's rejections only
    if len(new_approved) == len(draft_ttps_list):
        feedback = ""
    else:
        feedback = (
            "\n".join(result.feedback_notes) if result and result.feedback_notes else ""
        )

    print(
        f"[DEBUG] [Chunk {c_idx}] Validator - Approved {len(new_approved)}/{len(draft_ttps_list)} drafts."
    )

    return {
        "approved_ttps": new_approved,
        "validation_feedback": feedback,
        "draft_ttps": [],  # Limpiamos los drafts para la siguiente iteración
        "validator_time": state.get("validator_time", 0.0) + (time.time() - t0),
        "input_tokens": state.get("input_tokens", 0) + in_tok,
        "output_tokens": state.get("output_tokens", 0) + out_tok,
    }
