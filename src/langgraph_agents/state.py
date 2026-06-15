import operator
from typing import TypedDict, Annotated, List, Dict, Any


class GlobalState(TypedDict):
    source_file: str
    global_context: str
    chunks: List[Dict[str, Any]]
    all_approved_ttps: Annotated[List[Any], operator.add]
    final_json: Dict[str, Any]
    input_tokens: int
    output_tokens: int


class ChunkState(TypedDict):
    chunk_text: str
    chunk_metadata: Dict[str, Any]
    global_context: str
    is_relevant: bool
    draft_ttps: List[Any]
    approved_ttps: Annotated[List[Any], operator.add]
    validation_feedback: str
    # Loop Control & Metrics
    loop_count: int
    triage_time: float
    extractor_time: float
    validator_time: float
    input_tokens: int
    output_tokens: int

    # Official Metadata map injected by Oracle
    metadata_map: dict

    # Cached items to prevent re-running tools on loops
    candidates_list: List[str]
    abstract_keywords: str
