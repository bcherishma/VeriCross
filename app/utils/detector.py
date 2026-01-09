from transformers import pipeline
from sentence_transformers import SentenceTransformer
import bert_score
import spacy
from typing import List, Dict
import torch

class HallucinationDetector:
    def __init__(self):
        # Use a proper NLI model
        self.nli_model = pipeline("text-classification", model="microsoft/DialoGPT-medium", return_all_scores=True)  # Placeholder, need better NLI
        self.embedder = SentenceTransformer('sentence-transformers/LaBSE')
        self.bert_scorer = bert_score.BERTScorer(lang="en")  # Will adjust for languages
        self.ner_models = {}  # Cache for different languages

    def load_ner_model(self, lang: str):
        if lang not in self.ner_models:
            if lang == "en":
                self.ner_models[lang] = spacy.load("en_core_web_sm")
            else:
                # For other languages, use multilingual model or download
                self.ner_models[lang] = spacy.load("xx_ent_wiki_sm")  # Multilingual
        return self.ner_models[lang]

    def semantic_entailment(self, premise: str, hypothesis: str) -> Dict:
        # Use a proper entailment model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        labels = ['contradiction', 'neutral', 'entailment']  # Assuming order
        scores = {label: prob.item() for label, prob in zip(labels, probs[0])}
        label = max(scores, key=scores.get)
        return {"score": scores[label], "label": label}

    def entity_alignment(self, source: str, summary: str, source_lang: str, summary_lang: str) -> Dict:
        source_nlp = self.load_ner_model(source_lang)
        summary_nlp = self.load_ner_model(summary_lang)
        
        source_entities = [(ent.text, ent.label_) for ent in source_nlp(source).ents]
        summary_entities = [(ent.text, ent.label_) for ent in summary_nlp(summary).ents]
        
        # Simple mismatch check
        source_set = set(source_entities)
        summary_set = set(summary_entities)
        mismatches = list(source_set - summary_set)
        
        return {
            "source_entities": [{"text": t, "label": l} for t, l in source_entities],
            "summary_entities": [{"text": t, "label": l} for t, l in summary_entities],
            "mismatches": [{"text": t, "label": l} for t, l in mismatches]
        }

    def confidence_heatmap(self, source: str, summary: str) -> List[float]:
        source_sentences = source.split('. ')
        summary_sentences = summary.split('. ')
        
        scores = []
        for s_sent in summary_sentences:
            max_score = 0
            for src_sent in source_sentences:
                emb1 = self.embedder.encode([src_sent], convert_to_tensor=True)
                emb2 = self.embedder.encode([s_sent], convert_to_tensor=True)
                score = torch.cosine_similarity(emb1, emb2).item()
                max_score = max(max_score, score)
            scores.append(max_score)
        return scores

    def evaluate(self, source: str, summary: str, source_lang: str, summary_lang: str) -> Dict:
        entailment = self.semantic_entailment(source, summary)
        entities = self.entity_alignment(source, summary, source_lang, summary_lang)
        heatmap = self.confidence_heatmap(source, summary)
        overall = (entailment['score'] + (1 - len(entities['mismatches']) / max(1, len(entities['source_entities'])))) / 2
        return {
            "entailment": entailment,
            "entity_alignment": entities,
            "heatmap": {"sentence_scores": heatmap},
            "overall_confidence": overall
        }