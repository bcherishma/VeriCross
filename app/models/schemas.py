from pydantic import BaseModel
from typing import List, Dict

class EvaluationRequest(BaseModel):
    source_text: str
    summary_text: str
    source_lang: str = "en"
    summary_lang: str = "en"

class EntailmentResult(BaseModel):
    score: float
    label: str  # entailment, contradiction, neutral

class EntityAlignment(BaseModel):
    source_entities: List[Dict]
    summary_entities: List[Dict]
    mismatches: List[Dict]

class HeatmapData(BaseModel):
    sentence_scores: List[float]

class EvaluationResponse(BaseModel):
    entailment: EntailmentResult
    entity_alignment: EntityAlignment
    heatmap: HeatmapData
    overall_confidence: float