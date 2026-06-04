from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END

from .state import ChunkState, GlobalState
from .nodes import triage_node, extractor_node, validator_node

# ==============================================================================
# ROUTING FUNCTIONS (CONDITIONAL EDGES)
# ==============================================================================

def triage_router(state: ChunkState) -> str:
    """
    Routes the chunk after the triage node.
    If it's irrelevant, stop processing this chunk. Otherwise, proceed to extraction.
    """
    if not state.get("is_relevant", True):
        return "end"
    return "extractor_node"

def extractor_router(state: ChunkState) -> str:
    """
    Routes the chunk after the extractor node.
    If the extractor returned 0 drafts, it means all TTPs are exhausted. We are done.
    """
    if len(state.get("draft_ttps", [])) == 0:
        return "end"
    return "validator_node"

def validator_router(state: ChunkState) -> str:
    """
    ALWAYS loop back to the extractor to find more TTPs, 
    unless we hit the safety loop limit (e.g., 4 loops max).
    """
    loop_count = state.get("loop_count", 0)
    if loop_count >= 4:
        return "end"
    return "extractor_node"


# ==============================================================================
# BUILD THE SUB-GRAPH (THE "MAP" PROCESS)
# ==============================================================================

def build_chunk_graph():
    """
    Builds and compiles the per-chunk sub-graph.
    This graph processes a single chunk of text to extract and validate TTPs.
    """
    # Initialize the state graph
    builder = StateGraph(ChunkState)
    
    # Add the nodes
    builder.add_node("triage_node", triage_node)
    builder.add_node("extractor_node", extractor_node)
    builder.add_node("validator_node", validator_node)
    
    # Add normal edge from START
    builder.add_edge(START, "triage_node")
    
    # Add conditional edge from triage node
    builder.add_conditional_edges(
        "triage_node",
        triage_router,
        {
            "end": END,
            "extractor_node": "extractor_node"
        }
    )
    
    # Add conditional edge from extractor to validator
    builder.add_conditional_edges(
        "extractor_node",
        extractor_router,
        {
            "end": END,
            "validator_node": "validator_node"
        }
    )
    
    # Add conditional edge from validator node (the feedback loop)
    builder.add_conditional_edges(
        "validator_node",
        validator_router,
        {
            "end": END,
            "extractor_node": "extractor_node"
        }
    )
    
    # Compile and return the graph
    return builder.compile()


# ==============================================================================
# THE CONSOLIDATOR NODE (THE "REDUCE" PROCESS)
# ==============================================================================

def consolidator_node(state: GlobalState) -> dict:
    """
    Goal: Merge the outputs of all chunks into a final, clean list of TTPs 
    matching the standard JSON schema.
    """
    all_ttps = state.get("all_approved_ttps", [])
    deduplicated: Dict[str, Dict[str, Any]] = {}
    
    for ttp in all_ttps:
        tech_id = ttp.get("technique_id")
        if not tech_id: continue
        tech_id_upper = tech_id.upper()
        
        if tech_id_upper not in deduplicated:
            deduplicated[tech_id_upper] = {
                "technique_id": tech_id_upper,
                "tactic": ttp.get("tactic", []),
                "technique_name": ttp.get("name", "Unknown"),
                "occurrences": []
            }
        
        proc_text = ttp.get("procedure", "").strip()
        loc_text = ttp.get("location", "Unknown")
        justification = ttp.get("justification", "Verified by Validator Agent")
        conf_score = ttp.get("confidence_score", 0.95)
        
        is_duplicate = any(o["procedure"] == proc_text for o in deduplicated[tech_id_upper]["occurrences"])
        if proc_text and not is_duplicate:
            deduplicated[tech_id_upper]["occurrences"].append({
                "location": loc_text,
                "procedure": proc_text,
                "justification": justification,
                "confidence_score": conf_score
            })
            
    final_ttps = list(deduplicated.values())
    
    # Return ONLY the list of TTPs, the main script will build the outer shell
    return {"final_json": final_ttps}


# ==============================================================================
# THE MAIN ORCHESTRATOR (MAP-REDUCE EXECUTION)
# ==============================================================================

import time

def process_full_report(source_file: str, global_context: str, sanitized_chunks: List[Dict[str, Any]]) -> dict:
    """
    Main orchestrator that executes the Map-Reduce flow.
    Returns both the extracted TTPs and the internal timing metrics for Phase 3.
    """
    print(f"[*] Starting LangGraph Map-Reduce execution for: {source_file}")
    
    # Initialize the compiled chunk graph (Map Phase)
    chunk_graph = build_chunk_graph()
    master_ttp_list = []
    
    # Acumuladores de tiempo
    triage_time_total = 0.0
    extraction_time_total = 0.0
    validation_time_total = 0.0
    
    # 1. MAP: Loop over sanitized_chunks sequentially
    for i, chunk_data in enumerate(sanitized_chunks):
        print(f"  -> Processing chunk {i+1}/{len(sanitized_chunks)}...")
        
        chunk_text = chunk_data.get("text", str(chunk_data))
        chunk_metadata = chunk_data.get("metadata", {})
        
        # Initialize ChunkState
        initial_chunk_state: ChunkState = {
            "chunk_text": chunk_text,
            "chunk_metadata": chunk_metadata,
            "global_context": global_context,
            "is_relevant": True,  # Will be assessed by triage
            "draft_ttps": [],
            "approved_ttps": [],
            "validation_feedback": "",
            "loop_count": 0
        }
        
        try:
            t0 = time.time()
            # Invoke the graph for this single chunk
            chunk_result = chunk_graph.invoke(initial_chunk_state)
            chunk_time = time.time() - t0
            
            # Aproximación heurística del tiempo según el camino que tomó en el grafo
            if not chunk_result.get("is_relevant"):
                # Murió en el Triage
                triage_time_total += chunk_time
            elif len(chunk_result.get("draft_ttps", [])) == 0 and len(chunk_result.get("approved_ttps", [])) == 0:
                # Pasó Triage, fue al Extractor, pero no propuso nada (No llegó al Validator)
                triage_time_total += (chunk_time * 0.15)
                extraction_time_total += (chunk_time * 0.85)
            else:
                # Recorrió todo el camino (Triage -> Extractor -> Validator)
                triage_time_total += (chunk_time * 0.10)
                extraction_time_total += (chunk_time * 0.60)
                validation_time_total += (chunk_time * 0.30)
                
            approved = chunk_result.get("approved_ttps", [])
            if approved:
                print(f"     [+] Found {len(approved)} valid TTP(s) in chunk {i+1}")
                master_ttp_list.extend(approved)
            else:
                print(f"     [-] No valid TTPs found in chunk {i+1}")
                
        except Exception as e:
            print(f"     [!] Error processing chunk {i+1}: {str(e)}")
            
    print(f"[*] Map phase complete. Consolidating {len(master_ttp_list)} total TTP(s)...")
    
    # 2. REDUCE: Instantiate GlobalState and run consolidator
    global_state: GlobalState = {
        "source_file": source_file,
        "global_context": global_context,
        "chunks": sanitized_chunks,
        "all_approved_ttps": master_ttp_list,
        "final_json": {}
    }
    
    t0_red = time.time()
    final_state_update = consolidator_node(global_state)
    consolidator_time = time.time() - t0_red
    
    print("[*] Reduce phase complete. Report ready.")
    
    # Devolver estructura empaquetada con TTPs y Tiempos
    return {
        "extracted_ttps": final_state_update.get("final_json", []),
        "timing_breakdown_phase3": {
            "triage_node_seconds": round(triage_time_total, 2),
            "extraction_oracle_node_seconds": round(extraction_time_total, 2),
            "validator_node_seconds": round(validation_time_total, 2),
            "consolidator_seconds": round(consolidator_time, 2)
        }
    }
