import pytest
from app.utils.detector import HallucinationDetector

def test_semantic_entailment():
    detector = HallucinationDetector()
    result = detector.semantic_entailment("The cat is on the mat.", "A feline is resting on a rug.")
    assert result['label'] in ['entailment', 'neutral', 'contradiction']

def test_entity_alignment():
    detector = HallucinationDetector()
    result = detector.entity_alignment("John went to Paris.", "John visited London.", "en", "en")
    assert 'mismatches' in result