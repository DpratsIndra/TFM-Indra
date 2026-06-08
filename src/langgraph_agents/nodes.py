from typing import List, Any
from pydantic import BaseModel, Field

import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import ChunkState
from .tools import mitre_oracle, get_mitre_candidates

# ==============================================================================
# LLM FACTORY
# ==============================================================================

def get_llm(tier="pro", temperature=0.0):
    from src.core.llm_factory import get_llm as factory_get_llm
    return factory_get_llm(temperature=temperature)

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class TriageDecision(BaseModel):
    is_relevant: bool = Field(
        description="True if the text contains any cyber threat intelligence, attacker behavior, or technical indicators. False otherwise."
    )

class DraftTTP(BaseModel):
    technique_id: str = Field(description="The MITRE ATT&CK Technique ID (e.g., T1059.001)")
    tactic: List[str] = Field(default_factory=list, description="List of associated MITRE tactics (e.g. ['initial-access', 'execution'])")
    name: str = Field(description="The name of the MITRE ATT&CK Technique")
    procedure: str = Field(description="The specific procedure or behavior observed in the text")
    justification: str = Field(default="Verified by Validator", description="Technical explanation of why this chunk demonstrates the technique.")
    mitre_description: str = Field(default="", description="The official MITRE ATT&CK description for this technique provided in the candidates")
    confidence_score: float = Field(description="The numeric score provided in the Candidate Techniques list.")
    location: str = Field(default="Unknown", description="The document page and chunk location of the evidence.")

class DraftTTPList(BaseModel):
    ttps: List[DraftTTP] = Field(description="A list of drafted TTPs from the text.")

class ValidationResult(BaseModel):
    valid_ttps: List[DraftTTP] = Field(
        description="A list of approved TTPs that strictly match the text."
    )
    feedback_notes: List[str] = Field(
        description="A list of feedback notes explaining what was hallucinated or missing."
    )

# ==============================================================================
# NODE FUNCTIONS
# ==============================================================================

def safe_invoke(callable_chain, input_data):
    """Safely invokes a chain or agent, returning None on failure."""
    try:
        return callable_chain.invoke(input_data)
    except Exception as e:
        print(f"[ERROR] Model Invocation Error: {e}")
        return None

def triage_node(state: ChunkState) -> dict:
    """
    Objetivo: Actuar como un filtro rápido de bajo coste.
    Este agente lee el chunk y decide si contiene Inteligencia de Amenazas (TTPs, atacantes) 
    o si es pura paja (índices, bibliografía, relleno). Si es irrelevante, cortamos la ejecución 
    del chunk aquí mismo y ahorramos tokens en los agentes pesados.
    """
    import time
    t0 = time.time()
    chunk_text = state["chunk_text"]
    
    llm = get_llm(tier="lite")
    structured_llm = llm.with_structured_output(TriageDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Role: Cyber Threat Intelligence Analyst.\n"
                   "Task: Perform a binary classification on text chunks.\n"
                   "Condition: Return True if the text contains cybersecurity threat intelligence, attacker behavior, or technical indicators. Return False otherwise.\n"
                   "Note: The text may be in any language (Spanish, Russian, etc.). Analyze it natively."),
        ("human", "<text>\n{chunk_text}\n</text>")
    ])
    
    chain = prompt | structured_llm
    
    result = safe_invoke(chain, {"chunk_text": chunk_text})
    
    if result is None:
        is_relevant = True
    else:
        is_relevant = result.is_relevant
        
    c_idx = state.get("chunk_metadata", {}).get("chunk_index", "?")
    print(f"[DEBUG] [Chunk {c_idx}] Triage - Is Relevant: {is_relevant}")
    return {"is_relevant": is_relevant, "triage_time": time.time() - t0}

def extractor_node(state: ChunkState) -> dict:
    """
    Objetivo: El corazón del sistema. Analiza el chunk, extrae comportamientos tácticos,
    consulta a la base de datos vectorial (MITRE_Oracle) para anclar el comportamiento a 
    una técnica oficial de MITRE ATT&CK, y genera un borrador del TTP.
    Si viene rebotado del Validator con feedback, intenta buscar técnicas alternativas.
    """
    import time
    t0 = time.time()
    chunk_text = state["chunk_text"]
    val_feedback = state.get("validation_feedback", "")
    approved_ttps = state.get("approved_ttps", [])
    approved_ids = [t.get("technique_id") for t in approved_ttps]
    
    # 1. QUERY TRANSFORMATION: Abstract Keywords Only
    translator_llm = get_llm(tier="lite")
    translator_prompt = ChatPromptTemplate.from_messages([
        ("system", "Role: Cyber Threat Intelligence Analyst.\n"
                   "Task: Extract a comma-separated list of abstract cybersecurity behaviors, tactics, and mechanisms from the text.\n"
                   "Instructions: The text may be in any language. Translate specific tools into their tactical purpose (e.g., 'Ngrok' -> 'Protocol Tunneling'). Output ONLY the keywords in English."),
        ("human", "<text>\n{chunk_text}\n</text>")
    ])
    
    translation_chain = translator_prompt | translator_llm | StrOutputParser()
    c_idx = state.get("chunk_metadata", {}).get("chunk_index", "?")
    
    try:
        abstract_keywords = translation_chain.invoke({"chunk_text": chunk_text})
    except Exception as e:
        print(f"[ERROR] [Chunk {c_idx}] Translation Error: {e}")
        abstract_keywords = ""
        
    print(f"[DEBUG] [Chunk {c_idx}] Extractor - Abstract Keywords: {abstract_keywords}")
    
    # 2. DYNAMIC MECHANICAL RETRIEVAL
    # Pass the RAW TEXT plus the keywords to the Oracle so no context is lost.
    search_query = f"Raw Text: {chunk_text}\nKeywords: {abstract_keywords}"
    if val_feedback:
        search_query += f"\nPast Rejections & Feedback: {val_feedback}"
        
    candidates_list = get_mitre_candidates(search_query)
    
    if not candidates_list:
        print(f"[DEBUG] [Chunk {c_idx}] Extractor - No candidates found via Mechanical Retrieval. Returning empty drafts.")
        return {"draft_ttps": [], "loop_count": state.get("loop_count", 0) + 1, "extractor_time": state.get("extractor_time", 0.0) + (time.time() - t0)}
    
    # 3. EXTRACTION (BRAINSTORMER)
    llm = get_llm(tier="pro")
    structured_llm = llm.with_structured_output(DraftTTPList)
    
    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
    sys_prompt = (
        "Role: Cyber Threat Intelligence Analyst.\n"
        "Task: Extract applicable MITRE ATT&CK techniques from the provided candidate list based on the source text.\n\n"
        "Rules:\n"
        "1. Multilingual: The source text may be in Spanish, Russian, Chinese, or any other language. Comprehend it natively but write your justification in English.\n"
        "2. Extract all applicable new techniques. Do not artificially limit the output.\n"
        "3. Ignore previously approved or rejected techniques.\n"
        "4. Focus exclusively on observable actions (e.g., executing commands, dropping files). Do not map analytical context to techniques.\n"
        "5. Output strictly according to the requested JSON schema."
    )
    if use_prompt_repetition:
        sys_prompt = f"{sys_prompt}\n\nReminder of Rules:\n{sys_prompt}"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{prompt_text}")
    ])
    
    # Chunk the candidates into blocks of 10 to avoid context overload
    candidates_batches = [candidates_list[i:i+10] for i in range(0, len(candidates_list), 10)]
    all_drafts = []
    
    for batch in candidates_batches:
        candidates_str = "CANDIDATE TECHNIQUES:\n" + "\n\n".join(batch)
        
        prompt_text = f"Global Context: {state.get('global_context', '')}\n\n"
        
        if approved_ids:
            prompt_text += f"ALREADY APPROVED TTPs FOR THIS CHUNK: {approved_ids}\n"
            prompt_text += "Do NOT propose these exact technique IDs again. Search the text for OTHER distinct malicious behaviors.\n"
            prompt_text += "If you cannot find any NEW behaviors, you MUST return an empty list.\n\n"
            
        if val_feedback:
            prompt_text += f"PAST VALIDATOR REJECTIONS (CRITICAL: DO NOT PROPOSE THESE REJECTED TTPs AGAIN):\n{val_feedback}\n\n"
            
        prompt_text += f"Candidate MITRE Techniques:\n{candidates_str}\n\n"
        prompt_text += f"Source Text:\n{chunk_text}"
        
        parsed_result = safe_invoke(prompt | structured_llm, {"prompt_text": prompt_text})
        
        if parsed_result is not None:
            loc = f"Page {state.get('chunk_metadata', {}).get('page_number', '?')}, Chunk {state.get('chunk_metadata', {}).get('chunk_index', '?')}"
            for ttp in parsed_result.ttps:
                draft = ttp.model_dump()
                draft["location"] = loc
                all_drafts.append(draft)
        
    draft_ids = [d.get("technique_id", "Unknown") for d in all_drafts]
    print(f"[DEBUG] [Chunk {c_idx}] Extractor - Draft TTPs found: {len(all_drafts)} {draft_ids}")
    return {"draft_ttps": all_drafts, "loop_count": state.get("loop_count", 0) + 1, "extractor_time": state.get("extractor_time", 0.0) + (time.time() - t0)}

def validator_node(state: ChunkState) -> dict:
    """
    Goal: Perform a 'Line-Item Veto'. Strict verification of drafted TTPs against the source text.
    """
    import time
    t0 = time.time()
    c_idx = state.get("chunk_metadata", {}).get("chunk_index", "?")
    
    draft_ttps_list = state.get("draft_ttps", [])
    if not draft_ttps_list:
        print(f"[DEBUG] [Chunk {c_idx}] Validator - No drafts to validate. Skipping LLM call.")
        return {"approved_ttps": [], "validation_feedback": "", "draft_ttps": [], "validator_time": state.get("validator_time", 0.0) + (time.time() - t0)}

    llm = get_llm(tier="pro", temperature=0.2)
    structured_llm = llm.with_structured_output(ValidationResult)
    
    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
    
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
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "<context>\n{global_context}\n</context>\n\n<source_text>\n{chunk_text}\n</source_text>\n\n<drafted_techniques>\n{draft_ttps}\n</drafted_techniques>\n\nValidate these techniques and provide the approved list and feedback notes.")
    ])
    
    chain = prompt | structured_llm
    
    result = safe_invoke(chain, {
        "global_context": state.get("global_context", "None provided."),
        "chunk_text": state["chunk_text"],
        "draft_ttps": json.dumps(draft_ttps_list, indent=2)
    })
    
    if result is None:
        print(f"[DEBUG] [Chunk {c_idx}] Validator - Validation failed gracefully.")
        return {"approved_ttps": [], "validation_feedback": "", "draft_ttps": []}
        
    new_approved = [ttp.model_dump() for ttp in result.valid_ttps] if result and result.valid_ttps else []
    
    # Accumulate approved TTPs with deduplication
    current_approved = state.get("approved_ttps", [])
        
    # Set feedback to current loop's rejections only
    if len(new_approved) == len(draft_ttps_list):
        feedback = ""
    else:
        feedback = "\n".join(result.feedback_notes) if result and result.feedback_notes else ""
        
    print(f"[DEBUG] [Chunk {c_idx}] Validator - Approved {len(new_approved)}/{len(draft_ttps_list)} drafts.")
    
    return {
        "approved_ttps": current_approved + new_approved,
        "validation_feedback": feedback,
        "draft_ttps": [], # Limpiamos los drafts para la siguiente iteración
        "validator_time": state.get("validator_time", 0.0) + (time.time() - t0)
    }
