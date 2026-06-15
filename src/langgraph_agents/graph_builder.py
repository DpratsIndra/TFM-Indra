import os
import time
import concurrent.futures
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END

from .state import ChunkState, GlobalState
from .nodes import triage_node, extractor_node, validator_node

# ==============================================================================
# ROUTING FUNCTIONS (CONDITIONAL EDGES)
# ==============================================================================


def triage_router(state: ChunkState) -> str:
    """
    Router: Decide el siguiente paso tras el Triage.
    Si el chunk es basura (is_relevant=False), matamos el proceso (END).
    Si hay contexto útil, pasamos al nodo Extractor.
    """
    if not state.get("is_relevant", True):
        return "end"
    return "extractor_node"


def extractor_router(state: ChunkState) -> str:
    """
    Router: Si el extractor devuelve 0 borradores, significa que ya hemos "agotado"
    todos los TTPs posibles de este texto. Fin del procesamiento.
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
    if loop_count >= 3:
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
        "triage_node", triage_router, {"end": END, "extractor_node": "extractor_node"}
    )

    # Add conditional edge from extractor to validator
    builder.add_conditional_edges(
        "extractor_node",
        extractor_router,
        {"end": END, "validator_node": "validator_node"},
    )

    # Add conditional edge from validator node (the feedback loop)
    builder.add_conditional_edges(
        "validator_node",
        validator_router,
        {"end": END, "extractor_node": "extractor_node"},
    )

    # Compile and return the graph
    return builder.compile()


# ==============================================================================
# THE CONSOLIDATOR NODE (THE "REDUCE" PROCESS)
# ==============================================================================


def limpiar_redundancia_jerarquica(
    ttps_aprobados: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Si en la lista final de un chunk tengo una técnica padre (TXXXX) y también una
    sub-técnica hija (TXXXX.YYY), elimino la técnica padre porque la hija es más precisa.
    """
    ids_presentes = [ttp.get("technique_id", "") for ttp in ttps_aprobados]
    ttps_limpios = []

    for ttp in ttps_aprobados:
        tech_id = ttp.get("technique_id", "")
        # Si es un padre (no tiene punto) y alguna subtecnica suya está en la lista
        if "." not in tech_id and any(
            child.startswith(f"{tech_id}.") for child in ids_presentes
        ):
            continue  # Lo descartamos
        ttps_limpios.append(ttp)

    return ttps_limpios


def consolidator_node(state: GlobalState) -> dict:
    """
    Objetivo: Fase de "Reduce" del pipeline.
    Recoge todos los TTPs aprobados de todos los chunks procesados en paralelo,
    elimina duplicados (fusionando evidencias de distintas páginas) y devuelve
    un JSON limpio y consolidado con el reporte final.
    matching the standard JSON schema.
    """
    all_ttps = state.get("all_approved_ttps", [])

    # 1. Limpieza Algorítmica de redundancia jerárquica
    all_ttps = limpiar_redundancia_jerarquica(all_ttps)

    deduplicated: Dict[str, Dict[str, Any]] = {}

    for ttp in all_ttps:
        tech_id = ttp.get("technique_id")
        if not tech_id:
            continue
        tech_id_upper = tech_id.upper()

        if tech_id_upper not in deduplicated:
            deduplicated[tech_id_upper] = {
                "technique_id": tech_id_upper,
                "tactic": ttp.get("tactic", []),
                "technique_name": ttp.get("name", "Unknown"),
                "occurrences": [],
            }

        proc_text = ttp.get("procedure", "").strip()
        loc_text = ttp.get("location", "Unknown")
        justification = ttp.get("justification", "Verified by Validator Agent")
        conf_score = ttp.get("confidence_score", 0.95)

        is_duplicate = any(
            o["procedure"] == proc_text
            for o in deduplicated[tech_id_upper]["occurrences"]
        )
        if proc_text and not is_duplicate:
            deduplicated[tech_id_upper]["occurrences"].append(
                {
                    "location": loc_text,
                    "procedure": proc_text,
                    "justification": justification,
                    "confidence_score": conf_score,
                }
            )

    final_ttps = list(deduplicated.values())

    # Return ONLY the list of TTPs, the main script will build the outer shell
    return {"final_json": final_ttps}


# ==============================================================================
# THE MAIN ORCHESTRATOR (MAP-REDUCE EXECUTION)
# ==============================================================================


def process_full_report(
    source_file: str, global_context: str, sanitized_chunks: List[Dict[str, Any]]
) -> dict:
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
    input_tokens_total = 0
    output_tokens_total = 0

    # 1. MAP: Loop over sanitized_chunks using ThreadPoolExecutor for parallelism
    max_workers = int(os.getenv("MAX_CONCURRENT_CHUNKS", "2"))
    print(f"[*] Starting MAP phase across {len(sanitized_chunks)} chunks (Concurrent workers: {max_workers})...")

    # FIX: Pre-warm the caches on the MAIN thread to prevent Race Conditions.
    # If 2 threads try to load the models at the same time, it duplicates RAM usage and triggers the OOM Killer.
    from src.langgraph_agents.tools import get_retriever, load_mitre_json
    get_retriever()
    load_mitre_json()

    def _process_single_chunk(args):
        i, chunk_data = args
        chunk_text = chunk_data.get("text", str(chunk_data))
        chunk_metadata = chunk_data.get("metadata", {})

        initial_chunk_state: ChunkState = {
            "chunk_text": chunk_text,
            "chunk_metadata": chunk_metadata,
            "global_context": global_context,
            "is_relevant": True,
            "draft_ttps": [],
            "approved_ttps": [],
            "validation_feedback": "",
            "loop_count": 0,
            "triage_time": 0.0,
            "extractor_time": 0.0,
            "validator_time": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        try:
            chunk_result = chunk_graph.invoke(initial_chunk_state)
            
            # LATIDO: Solo imprimimos si extrajo algo útil o si terminó bien
            aprobados = len(chunk_result.get("approved_ttps", []))
            print(f"     [✓] Chunk {i + 1}/{len(sanitized_chunks)} procesado. TTPs válidos: {aprobados}")
            
            return i, chunk_result
        except Exception as e:
            print(f"     [!] Error Crítico/Timeout procesando chunk {i + 1}: {str(e)}")
            return i, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        args_list = [(i, chunk) for i, chunk in enumerate(sanitized_chunks)]
        results = list(executor.map(_process_single_chunk, args_list))

    api_crashed_flag = False
    for i, chunk_result in results:
        if chunk_result is None:
            continue
            
        if isinstance(chunk_result, Exception):
            err_str = str(chunk_result).lower()
            api_errors = ["429", "quota", "resourceexhausted", "503", "500", "timeout", "not_found", "api", "connection", "unavailable", "rate limit"]
            if any(err in err_str for err in api_errors):
                print(f"[!] ABORTANDO LangGraph: Fallo de API detectado en chunk {i + 1}. Guardando progreso parcial...")
                api_crashed_flag = True
                break
            else:
                continue

        triage_time_total += chunk_result.get("triage_time", 0.0)
        extraction_time_total += chunk_result.get("extractor_time", 0.0)
        validation_time_total += chunk_result.get("validator_time", 0.0)
        input_tokens_total += chunk_result.get("input_tokens", 0)
        output_tokens_total += chunk_result.get("output_tokens", 0)

        approved = chunk_result.get("approved_ttps", [])
        if approved:
            print(f"     [+] Found {len(approved)} valid TTP(s) in chunk {i + 1}")
            master_ttp_list.extend(approved)

    print(
        f"[*] Map phase complete. Consolidating {len(master_ttp_list)} total TTP(s)..."
    )

    # 2. REDUCE: Instantiate GlobalState and run consolidator
    global_state: GlobalState = {
        "source_file": source_file,
        "global_context": global_context,
        "chunks": sanitized_chunks,
        "all_approved_ttps": master_ttp_list,
        "final_json": {},
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
    }

    t0_red = time.time()
    final_state_update = consolidator_node(global_state)
    consolidator_time = time.time() - t0_red

    final_count = len(final_state_update.get("final_json", []))
    print(f"[*] Reduce phase complete. Consolidated {len(master_ttp_list)} raw extractions into {final_count} distinct TTPs. Report ready.")

    # Devolver estructura empaquetada con TTPs y Tiempos
    return {
        "extracted_ttps": final_state_update.get("final_json", []),
        "timing_breakdown_phase3": {
            "triage_node_seconds": round(triage_time_total, 2),
            "extraction_oracle_node_seconds": round(extraction_time_total, 2),
            "validator_node_seconds": round(validation_time_total, 2),
            "consolidator_seconds": round(consolidator_time, 2),
            "input_tokens": input_tokens_total,
            "output_tokens": output_tokens_total,
            "api_crashed": api_crashed_flag
        },
    }
