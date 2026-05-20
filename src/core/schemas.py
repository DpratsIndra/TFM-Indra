from typing import List
from pydantic import BaseModel, Field

class TTPDetection(BaseModel):
    """
    Pydantic schema to enforce structured LLM output for MITRE ATT&CK technique detection.
    """
    technique_id: str = Field(
        ...,
        description="El ID de la técnica de MITRE ATT&CK (ej. T1059)."
    )
    is_present: bool = Field(
        ...,
        description="True si la evidencia demuestra inequívocamente el ataque. False si es un falso positivo o una mención benigna/defensiva."
    )
    technical_justification: str = Field(
        ...,
        description="Explicación técnica de por qué se detectó (o por qué se descartó) esta técnica, citando el texto original proporcionado."
    )
    iocs_found: List[str] = Field(
        default_factory=list,
        description="Lista de las etiquetas enmascaradas encontradas en la evidencia (ej. ['<IoC_IPv4>', '<IoC_HASH>'])."
    )
