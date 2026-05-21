from typing import List
from pydantic import BaseModel, Field

class TTPDetection(BaseModel):
    """
    Pydantic schema to enforce structured LLM output for MITRE ATT&CK technique detection.
    """
    technique_id: str = Field(
        ...,
        description="The MITRE ATT&CK technique ID (e.g. T1059)."
    )
    tactic: List[str] = Field(
        default_factory=list,
        description="List of associated MITRE tactics obtained from context."
    )
    technique_name: str = Field(
        default="",
        description="The full name of the MITRE ATT&CK technique."
    )
    is_present: bool = Field(
        ...,
        description="True if the evidence unequivocally demonstrates the attacker used this technique. False if it is a false positive, benign, or defensive mention."
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence score coming from the retrieval phase."
    )
    justification: str = Field(
        ...,
        description="Technical explanation of why this technique was detected (or why it was rejected), quoting the original provided text."
    )
    procedure: str = Field(
        ...,
        description="A concise explanation of the exact, specific action the attacker took in this report (The MITRE ATT&CK Procedure)."
    )
