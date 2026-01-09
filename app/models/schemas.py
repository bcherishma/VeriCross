from pydantic import BaseModel
from typing import List, Dict

class EvaluationRequest(BaseModel):
    source_text: str
    summary_text: str
    source_lang: str = "auto"  # "auto" for auto-detection
    summary_lang: str = "auto"  # "auto" for auto-detection

class EntailmentResult(BaseModel):
    score: float
    label: str  # entailment, contradiction, neutral

class EntityAlignment(BaseModel):
    source_entities: List[Dict]
    summary_entities: List[Dict]
    matches: List[Dict]  # Entities that match across languages
    mismatches: List[Dict]  # Entities in source but not in summary
    extra_entities: List[Dict]  # Entities in summary but not in source

class HeatmapData(BaseModel):
    sentence_scores: List[float]

class EvaluationResponse(BaseModel):
    entailment: EntailmentResult
    entity_alignment: EntityAlignment
    heatmap: HeatmapData
    overall_confidence: float
    detected_source_lang: str = "en"
    detected_summary_lang: str = "en"
    executive_summary: str = ""