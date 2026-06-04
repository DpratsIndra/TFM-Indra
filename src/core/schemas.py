from typing import List
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    location: str = Field(
        ...,
        description="The location of the evidence (e.g., 'Page 3, Chunk 5')."
    )
    procedure: str = Field(
        ...,
        description="A concise explanation of the exact, specific action the attacker took in this report (The MITRE ATT&CK Procedure)."
    )
    justification: str = Field(
        ...,
        description="Technical explanation of why this chunk demonstrates the technique, quoting the original provided text."
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score coming from the retrieval phase."
    )

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
    occurrences: List[Evidence] = Field(
        default_factory=list,
        description="A list of specific occurrences/evidence blocks where the technique was confirmed."
    )

class ChunkEvidence(BaseModel):
    technique_id: str = Field(..., description="The MITRE ATT&CK technique ID (e.g. T1059)")
    technique_name: str = Field(..., description="The name of the technique")
    procedure: str = Field(..., description="Exact attacker action described in this chunk")
    justification: str = Field(..., description="Why this chunk matches the technique")

class ChunkExtraction(BaseModel):
    extracted_ttps: List[ChunkEvidence] = Field(
        default_factory=list, 
        description="List of confirmed techniques in this chunk. Return an empty list if no candidate techniques are present."
    )
