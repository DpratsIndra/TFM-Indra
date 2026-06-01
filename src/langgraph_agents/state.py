import operator
from typing import TypedDict, Annotated, List, Dict, Any

class GlobalState(TypedDict):
    source_file: str
    global_context: str
    chunks: List[Dict[str, Any]]
    all_approved_ttps: Annotated[List[Any], operator.add]
    final_json: Dict[str, Any]

class ChunkState(TypedDict):
    chunk_text: str
    chunk_metadata: Dict[str, Any]
    global_context: str
    is_relevant: bool
    draft_ttps: List[Any]
    approved_ttps: List[Any]
    validation_feedback: str
    loop_count: int
