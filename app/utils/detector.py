from transformers import pipeline
from sentence_transformers import SentenceTransformer
import bert_score
import spacy
from typing import List, Dict, Optional
import torch
from langdetect import detect, DetectorFactory, LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

class HallucinationDetector:
    def __init__(self):
        # Use a proper NLI model
        self.nli_model = pipeline("text-classification", model="microsoft/DialoGPT-medium", return_all_scores=True)  # Placeholder, need better NLI
        self.embedder = SentenceTransformer('sentence-transformers/LaBSE')
        self.bert_scorer = bert_score.BERTScorer(lang="en")  # Will adjust for languages
        self.ner_models = {}  # Cache for different languages

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text"""
        try:
            # Remove very short texts that might cause errors
            if len(text.strip()) < 3:
                return "en"  # Default to English
            
            detected = detect(text)
            
            # Map Indian languages and other common variations
            lang_mapping = {
                'hi': 'hi',  # Hindi
                'ta': 'ta',  # Tamil
                'te': 'te',  # Telugu
                'kn': 'kn',  # Kannada
                'ml': 'ml',  # Malayalam
                'mr': 'mr',  # Marathi
                'gu': 'gu',  # Gujarati
                'pa': 'pa',  # Punjabi
                'bn': 'bn',  # Bengali
                'ur': 'ur',  # Urdu
                'or': 'or',  # Odia
                'as': 'as',  # Assamese
                'ne': 'ne',  # Nepali
                'si': 'si',  # Sinhala
                'de': 'de',  # German
                'fr': 'fr',  # French
                'es': 'es',  # Spanish
                'it': 'it',  # Italian
                'pt': 'pt',  # Portuguese
                'ru': 'ru',  # Russian
                'zh': 'zh',  # Chinese
                'ja': 'ja',  # Japanese
                'ko': 'ko',  # Korean
                'ar': 'ar',  # Arabic
                'en': 'en',  # English
            }
            
            # Return mapped language or default to detected language
            return lang_mapping.get(detected, detected)
        except (LangDetectException, Exception):
            # Default to English if detection fails
            return "en"
    
    def load_ner_model(self, lang: str):
        if lang not in self.ner_models:
            if lang == "en":
                self.ner_models[lang] = spacy.load("en_core_web_sm")
            else:
                # For other languages, use multilingual model or download
                # The multilingual model supports many languages including Indian languages
                self.ner_models[lang] = spacy.load("xx_ent_wiki_sm")  # Multilingual
        return self.ner_models[lang]

    def semantic_entailment(self, premise: str, hypothesis: str, source_lang: str = "en", summary_lang: str = "en") -> Dict:
        # For cross-lingual cases, use semantic similarity instead of English-only NLI model
        if source_lang != summary_lang or source_lang != "en":
            # Use LaBSE embeddings for cross-lingual semantic similarity
            try:
                source_emb = self.embedder.encode([premise], convert_to_tensor=True)
                summary_emb = self.embedder.encode([hypothesis], convert_to_tensor=True)
                similarity = torch.cosine_similarity(source_emb, summary_emb).item()
                
                # Convert similarity to entailment scores
                # High similarity (>0.8) = entailment, Medium (0.5-0.8) = neutral, Low (<0.5) = contradiction
                if similarity >= 0.8:
                    return {"score": similarity, "label": "entailment"}
                elif similarity >= 0.5:
                    return {"score": similarity, "label": "neutral"}
                else:
                    return {"score": 1 - similarity, "label": "contradiction"}
            except Exception:
                # Fallback to neutral if encoding fails
                return {"score": 0.5, "label": "neutral"}
        
        # For same-language (especially English), use RoBERTa-large-MNLI
        try:
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
        except Exception:
            # Fallback to semantic similarity if NLI model fails
            try:
                source_emb = self.embedder.encode([premise], convert_to_tensor=True)
                summary_emb = self.embedder.encode([hypothesis], convert_to_tensor=True)
                similarity = torch.cosine_similarity(source_emb, summary_emb).item()
                if similarity >= 0.8:
                    return {"score": similarity, "label": "entailment"}
                elif similarity >= 0.5:
                    return {"score": similarity, "label": "neutral"}
                else:
                    return {"score": 1 - similarity, "label": "contradiction"}
            except Exception:
                return {"score": 0.5, "label": "neutral"}

    def entity_alignment(self, source: str, summary: str, source_lang: str, summary_lang: str) -> Dict:
        source_nlp = self.load_ner_model(source_lang)
        summary_nlp = self.load_ner_model(summary_lang)
        
        source_entities = [(ent.text, ent.label_) for ent in source_nlp(source).ents]
        summary_entities = [(ent.text, ent.label_) for ent in summary_nlp(summary).ents]
        
        # Convert to dict format
        source_entity_dicts = [{"text": t, "label": l} for t, l in source_entities]
        summary_entity_dicts = [{"text": t, "label": l} for t, l in summary_entities]
        
        # Use embeddings for cross-lingual matching
        matches = []
        matched_source_indices = set()
        matched_summary_indices = set()
        
        # Similarity threshold for cross-lingual matching (lowered for better matching)
        SIMILARITY_THRESHOLD = 0.65
        
        # Define entity type groups (entities in same group can match)
        location_types = {'LOC', 'GPE', 'LOCATION', 'GEO'}
        person_types = {'PERSON', 'PER'}
        org_types = {'ORG', 'ORGANIZATION'}
        
        def are_compatible_labels(label1, label2):
            """Check if two entity labels are compatible (same type category)"""
            if label1 == label2:
                return True
            # Check if both are location types
            if label1 in location_types and label2 in location_types:
                return True
            # Check if both are person types
            if label1 in person_types and label2 in person_types:
                return True
            # Check if both are organization types
            if label1 in org_types and label2 in org_types:
                return True
            return False
        
        # Match entities using semantic similarity (works across languages)
        for i, src_ent in enumerate(source_entities):
            if i in matched_source_indices:
                continue
                
            src_text = src_ent[0]
            src_label = src_ent[1]
            best_match = None
            best_score = 0
            best_idx = -1
            best_label_match = False
            
            for j, sum_ent in enumerate(summary_entities):
                if j in matched_summary_indices:
                    continue
                    
                sum_text = sum_ent[0]
                sum_label = sum_ent[1]
                
                # Check if labels are compatible (same or similar entity type)
                label_compatible = are_compatible_labels(src_label, sum_label)
                
                # Use embeddings for semantic similarity (cross-lingual)
                try:
                    src_emb = self.embedder.encode([src_text], convert_to_tensor=True)
                    sum_emb = self.embedder.encode([sum_text], convert_to_tensor=True)
                    similarity = torch.cosine_similarity(src_emb, sum_emb).item()
                    
                    # Prefer exact label matches, but allow compatible labels
                    # Boost score slightly if labels match exactly
                    adjusted_score = similarity
                    if src_label == sum_label:
                        adjusted_score = min(1.0, similarity + 0.05)
                    
                    # Accept match if similarity is high enough and labels are compatible
                    if (similarity >= SIMILARITY_THRESHOLD and label_compatible and 
                        (adjusted_score > best_score or (best_match and src_label == sum_label and not best_label_match))):
                        best_score = similarity
                        best_match = sum_ent
                        best_idx = j
                        best_label_match = (src_label == sum_label)
                except Exception as e:
                    # Fallback: exact text match (case-insensitive)
                    if src_text.lower() == sum_text.lower() and label_compatible:
                        best_score = 1.0
                        best_match = sum_ent
                        best_idx = j
                        best_label_match = (src_label == sum_label)
            
            if best_match:
                matches.append({
                    "source_text": src_text,
                    "source_label": src_label,
                    "summary_text": best_match[0],
                    "summary_label": best_match[1],
                    "similarity": round(best_score, 3)
                })
                matched_source_indices.add(i)
                matched_summary_indices.add(best_idx)
        
        # Find mismatches (entities in source but not matched)
        mismatches = []
        for i, src_ent in enumerate(source_entities):
            if i not in matched_source_indices:
                mismatches.append({"text": src_ent[0], "label": src_ent[1]})
        
        # Find extra entities (entities in summary but not in source)
        extra_entities = []
        for j, sum_ent in enumerate(summary_entities):
            if j not in matched_summary_indices:
                extra_entities.append({"text": sum_ent[0], "label": sum_ent[1]})
        
        return {
            "source_entities": source_entity_dicts,
            "summary_entities": summary_entity_dicts,
            "matches": matches,
            "mismatches": mismatches,
            "extra_entities": extra_entities
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
        entailment = self.semantic_entailment(source, summary, source_lang, summary_lang)
        entities = self.entity_alignment(source, summary, source_lang, summary_lang)
        heatmap = self.confidence_heatmap(source, summary)
        
        # Calculate semantic similarity for overall text (cross-lingual)
        try:
            source_emb = self.embedder.encode([source], convert_to_tensor=True)
            summary_emb = self.embedder.encode([summary], convert_to_tensor=True)
            semantic_similarity = torch.cosine_similarity(source_emb, summary_emb).item()
        except Exception:
            semantic_similarity = 0.5  # Default if encoding fails
        
        # Calculate entity alignment score
        entity_score = 1.0
        if len(entities['source_entities']) > 0:
            match_ratio = len(entities['matches']) / len(entities['source_entities'])
            mismatch_penalty = len(entities['mismatches']) / max(1, len(entities['source_entities']))
            entity_score = match_ratio * (1 - mismatch_penalty * 0.5)
        else:
            # If no entities detected, rely more on semantic similarity
            # This handles cases like "tumhara naam kya hai" vs "what is your name"
            entity_score = semantic_similarity
        
        # Calculate overall confidence
        # Weight: 40% entailment, 30% entity alignment, 30% semantic similarity
        overall = (entailment['score'] * 0.4 + entity_score * 0.3 + semantic_similarity * 0.3)
        
        # Add semantic similarity to response for display
        entities['semantic_similarity'] = round(semantic_similarity, 3)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            entailment, entities, overall, semantic_similarity, source_lang, summary_lang
        )
        
        return {
            "entailment": entailment,
            "entity_alignment": entities,
            "heatmap": {"sentence_scores": heatmap},
            "overall_confidence": overall,
            "detected_source_lang": source_lang,
            "detected_summary_lang": summary_lang,
            "executive_summary": executive_summary
        }
    
    def _generate_executive_summary(self, entailment: Dict, entities: Dict, overall_confidence: float, 
                                    semantic_similarity: float, source_lang: str, summary_lang: str) -> str:
        """Generate a natural language executive summary of the evaluation results"""
        
        # Determine cross-lingual status
        is_cross_lingual = source_lang != summary_lang
        
        # Base assessment on overall confidence
        if overall_confidence >= 0.9:
            if semantic_similarity >= 0.95:
                summary = "The summary is an exact match of the source text, indicating perfect factual consistency and no hallucinations."
            else:
                summary = "The summary demonstrates excellent alignment with the source text with minimal discrepancies. The content is highly consistent with no significant hallucinations detected."
        elif overall_confidence >= 0.75:
            summary = "The summary shows strong alignment with the source text. Minor discrepancies may exist, but the overall factual consistency is good with minimal risk of hallucinations."
        elif overall_confidence >= 0.6:
            summary = "The summary has moderate alignment with the source text. Some discrepancies or potential hallucinations may be present and should be reviewed."
        elif overall_confidence >= 0.4:
            summary = "The summary shows weak alignment with the source text. Significant discrepancies and potential hallucinations are likely present and require careful review."
        else:
            summary = "The summary has poor alignment with the source text. Major discrepancies and hallucinations are highly likely, indicating the summary may not accurately represent the source content."
        
        # Add entailment-specific information
        if entailment['label'] == 'entailment' and entailment['score'] >= 0.8:
            summary += " The semantic entailment analysis confirms that the summary logically follows from the source."
        elif entailment['label'] == 'contradiction':
            summary += " However, the semantic entailment analysis indicates contradictions between the source and summary, suggesting potential factual errors."
        elif entailment['label'] == 'neutral':
            summary += " The semantic relationship between source and summary is neutral, indicating the summary may not fully capture the source's meaning."
        
        # Add entity alignment information
        total_entities = len(entities.get('source_entities', []))
        total_matches = len(entities.get('matches', []))
        total_mismatches = len(entities.get('mismatches', []))
        total_extra = len(entities.get('extra_entities', []))
        
        if total_entities > 0:
            match_rate = total_matches / total_entities
            if match_rate >= 0.9:
                summary += " Entity alignment is excellent, with nearly all named entities correctly matched across languages."
            elif match_rate >= 0.7:
                summary += " Most entities are correctly aligned, though some minor mismatches exist."
            elif match_rate >= 0.5:
                summary += " Entity alignment shows moderate success, with several entities not properly matched."
            else:
                summary += " Entity alignment is poor, with many entities missing or incorrectly matched."
            
            if total_mismatches > 0:
                summary += f" {total_mismatches} entity(ies) from the source were not found in the summary."
            if total_extra > 0:
                summary += f" {total_extra} entity(ies) appear in the summary but not in the source, which may indicate hallucination."
        else:
            # No entities detected - rely on semantic similarity
            if semantic_similarity >= 0.85:
                summary += " While no named entities were detected, the semantic similarity is very high, indicating strong content alignment."
            elif semantic_similarity >= 0.7:
                summary += " No named entities were detected, but semantic similarity suggests reasonable content alignment."
            else:
                summary += " No named entities were detected, and semantic similarity is low, suggesting potential content mismatch."
        
        # Add cross-lingual note if applicable
        if is_cross_lingual:
            summary += f" This evaluation was performed across languages ({source_lang} → {summary_lang}), using cross-lingual semantic matching."
        
        return summary