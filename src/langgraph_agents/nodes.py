from typing import List, Any
from pydantic import BaseModel, Field

import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import ChunkState
from .tools import mitre_oracle

# ==============================================================================
# LLM FACTORY
# ==============================================================================

def get_llm(tier="pro", temperature=0.0):
    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=3)

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
        print(f"[!] Model Invocation Error: {e}")
        return None

def triage_node(state: ChunkState) -> dict:
    """
    Goal: Act as a cheap, fast filter to drop irrelevant chunks 
    (e.g., legal disclaimers, marketing noise).
    """
    llm = get_llm(tier="lite")
    structured_llm = llm.with_structured_output(TriageDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cyber threat intelligence triage analyst. Your job is to determine if a given text chunk contains any valuable CTI, attacker behavior, or technical indicators."),
        ("human", "Does this text contain any cyber threat intelligence, attacker behavior, or technical indicators? Answer True or False.\n\nText: {chunk_text}")
    ])
    
    chain = prompt | structured_llm
    
    result = safe_invoke(chain, {"chunk_text": state["chunk_text"]})
    
    if result is None:
        is_relevant = True
    else:
        is_relevant = result.is_relevant
        
    print(f"[DEBUG - Triage] Chunk relevant: {is_relevant}")
    return {"is_relevant": is_relevant}

def extractor_node(state: ChunkState) -> dict:
    """
    Goal: Translate text to a Dense Tactical Summary, use Mechanical Retrieval, and draft TTPs.
    """
    chunk_text = state["chunk_text"]
    val_feedback = state.get("validation_feedback", "")
    approved_ttps = state.get("approved_ttps", [])
    approved_ids = [t.get("technique_id") for t in approved_ttps]
    
    # 1. QUERY TRANSFORMATION: Abstract Keywords Only
    translator_llm = get_llm(tier="lite")
    translator_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a CTI conceptual translator. Read the raw text and output ONLY a comma-separated list of abstract cybersecurity behaviors, tactics, and mechanisms present. "
                   "Translate specific tools into their tactical purpose (e.g., 'Ngrok' -> 'Protocol Tunneling'). "
                   "If you see fake identities or spoofing, output 'Impersonation'. "
                   "Output ONLY the keywords, no other text."),
        ("human", "Raw Text:\n{chunk_text}")
    ])
    
    translation_chain = translator_prompt | translator_llm | StrOutputParser()
    try:
        abstract_keywords = translation_chain.invoke({"chunk_text": chunk_text})
    except Exception as e:
        print(f"[!] Translation Error: {e}")
        abstract_keywords = ""
        
    print(f"[DEBUG - Extractor] Abstract Keywords: {abstract_keywords}")
    
    # 2. DYNAMIC MECHANICAL RETRIEVAL
    # Pass the RAW TEXT plus the keywords to the Oracle so no context is lost.
    search_query = f"Raw Text: {chunk_text}\nKeywords: {abstract_keywords}"
    if val_feedback:
        search_query += f"\nPast Rejections & Feedback: {val_feedback}"
        
    candidates_str = mitre_oracle.invoke(search_query)
    
    if "No matching MITRE techniques found" in candidates_str or "Error" in candidates_str or "No highly confident MITRE techniques" in candidates_str:
        print("[DEBUG - Extractor] No candidates found via Mechanical Retrieval. Returning empty drafts.")
        return {"draft_ttps": [], "loop_count": state.get("loop_count", 0) + 1}
    
    # 3. EXTRACTION (BRAINSTORMER)
    llm = get_llm(tier="pro")
    structured_llm = llm.with_structured_output(DraftTTPList)
    
    prompt_text = f"Global Context: {state.get('global_context', '')}\n\n"
    
    if approved_ids:
        prompt_text += f"ALREADY APPROVED TTPs FOR THIS CHUNK: {approved_ids}\n"
        prompt_text += "Do NOT propose these exact technique IDs again. Search the text for OTHER distinct malicious behaviors.\n"
        prompt_text += "If you cannot find any NEW behaviors, you MUST return an empty list.\n\n"
        
    if val_feedback:
        prompt_text += f"PAST VALIDATOR REJECTIONS (CRITICAL: DO NOT PROPOSE THESE REJECTED TTPs AGAIN):\n{val_feedback}\n\n"
        
    prompt_text += f"Candidate MITRE Techniques:\n{candidates_str}\n\n"
    prompt_text += f"Source Text:\n{chunk_text}"
    
    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
    
    sys_prompt = (
        "You are a CTI Brainstormer Extractor. Draft candidate TTPs from the Candidate Techniques list.\n"
        "Extract ALL applicable NEW TTPs. Do not artificially limit your output.\n"
        "CRITICAL: If you see ALREADY APPROVED TTPs or PREVIOUSLY REJECTED TTPs, you MUST ignore them.\n"
        "RULE - FOCUS ON OBSERVABLE ACTIONS: Your primary goal is to map technical execution. Focus on what the attacker or malware actually DID (e.g., executing commands, dropping files, establishing tunnels, sending phishing emails). "
        "Do not map the analyst's background theories (e.g., geopolitics, victim industry, historical motives) to MITRE techniques.\n"
        "Convert your brainstormed analysis into the strict DraftTTPList JSON schema. Do not hallucinate IDs."
    )
    
    if use_prompt_repetition:
        sys_prompt = f"{sys_prompt}\n\nCRITICAL REMINDER OF RULES:\n{sys_prompt}"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{prompt_text}")
    ])
    
    parsed_result = safe_invoke(prompt | structured_llm, {"prompt_text": prompt_text})
    
    if parsed_result is None:
        drafts = []
    else:
        drafts = []
        loc = f"Page {state.get('chunk_metadata', {}).get('page_number', '?')}, Chunk {state.get('chunk_metadata', {}).get('chunk_index', '?')}"
        for ttp in parsed_result.ttps:
            draft = ttp.model_dump()
            draft["location"] = loc
            drafts.append(draft)
        
    draft_ids = [d.get("technique_id", "Unknown") for d in drafts]
    print(f"[DEBUG - Extractor] Draft TTPs found: {len(drafts)} {draft_ids}")
    return {"draft_ttps": drafts, "loop_count": state.get("loop_count", 0) + 1}

def validator_node(state: ChunkState) -> dict:
    """
    Goal: Perform a 'Line-Item Veto'. Strict verification of drafted TTPs against the source text.
    """
    draft_ttps_list = state.get("draft_ttps", [])
    if not draft_ttps_list:
        print("[DEBUG - Validator] No drafts to validate. Skipping LLM call.")
        return {"approved_ttps": [], "validation_feedback": "", "draft_ttps": []}

    llm = get_llm(tier="pro", temperature=0.2)
    structured_llm = llm.with_structured_output(ValidationResult)
    
    use_prompt_repetition = os.getenv("USE_PROMPT_REPETITION", "False").lower() in ("true", "1", "yes")
    
    sys_prompt = (
        "You are a CTI Semantic Auditor. Your job is to independently verify drafted MITRE ATT&CK TTPs against the source text.\n"
        "Read the source text and the official MITRE description for each drafted TTP. "
        "Approve the TTP if the text contains concrete evidence of the attacker or malware performing the technical action described.\n"
        "Reject the TTP if it is based solely on theoretical motives, target demographics (like business relationships), or if the text implies the action didn't actually happen.\n"
        "If you reject a TTP, provide specific feedback explaining why so the Extractor can learn.\n"
        "Validate each technique on its own merits."
    )
    
    if use_prompt_repetition:
        sys_prompt = f"{sys_prompt}\n\nCRITICAL REMINDER OF RULES:\n{sys_prompt}"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Global Context:\n{global_context}\n\nSource Text:\n{chunk_text}\n\nDrafted TTPs (including official descriptions):\n{draft_ttps}\n\nValidate these TTPs and provide the approved valid list and any feedback notes.")
    ])
    
    chain = prompt | structured_llm
    
    result = safe_invoke(chain, {
        "global_context": state.get("global_context", "None provided."),
        "chunk_text": state["chunk_text"],
        "draft_ttps": json.dumps(draft_ttps_list, indent=2)
    })
    
    if result is None:
        print("[DEBUG - Validator] Validation failed gracefully.")
        return {"approved_ttps": [], "validation_feedback": "", "draft_ttps": []}
        
    valid_ttps = [ttp.model_dump() for ttp in result.valid_ttps] if result and result.valid_ttps else []
    
    # Accumulate approved TTPs with deduplication
    current_approved = state.get("approved_ttps", [])
    existing_ids = {t.get("technique_id") for t in current_approved}
    
    for ttp in valid_ttps:
        if ttp.get("technique_id") not in existing_ids:
            current_approved.append(ttp)
            
    # Set feedback to current loop's rejections only
    if len(valid_ttps) == len(draft_ttps_list):
        feedback = ""
    else:
        feedback = "\n".join(result.feedback_notes) if result and result.feedback_notes else ""
        
    valid_ids = [v.get("technique_id", "Unknown") for v in current_approved]
    print(f"[DEBUG - Validator] Total Approved TTPs so far: {len(current_approved)} {valid_ids}")
    if feedback:
        print(f"[DEBUG - Validator] Rejection Feedback:\n{feedback}")
    
    return {
        "approved_ttps": current_approved,
        "validation_feedback": feedback,
        "draft_ttps": []
    }
