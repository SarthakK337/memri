"""Query processing: entity extraction + temporal detection without LLM calls."""

import re
from typing import List, Dict

try:
    import spacy
    _nlp = None

    def _get_nlp():
        global _nlp
        if _nlp is None:
            _nlp = spacy.load("en_core_web_sm")
        return _nlp

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


def extract_entities(text: str) -> List[str]:
    """Extract entities using spaCy NER + proper nouns. Falls back to noun extraction."""
    if not HAS_SPACY:
        return []

    nlp = _get_nlp()
    doc = nlp(text)
    entities = set()

    # Named entities
    for ent in doc.ents:
        entities.add(ent.text)

    # Proper nouns and nouns (catch names spaCy misses)
    skip = {"what", "who", "when", "where", "how", "which", "did", "does",
            "would", "could", "should", "was", "were", "is", "are", "has", "have"}
    for token in doc:
        if token.pos_ in ("PROPN",) and token.text.lower() not in skip:
            entities.add(token.text)

    return list(entities)


def detect_temporal_intent(query: str) -> Dict:
    """Detect if query requires temporal reasoning. Regex-based, no LLM."""
    q = query.lower()

    temporal_keywords = {
        "before": "before",
        "after": "after",
        "when did": "when",
        "when was": "when",
        "when is": "when",
        "during": "during",
        "since": "after",
        "until": "before",
        "prior to": "before",
        "following": "after",
        "how long ago": "when",
        "how many years": "when",
        "what date": "when",
        "what day": "when",
        "what month": "when",
        "what year": "when",
    }

    for keyword, direction in temporal_keywords.items():
        if keyword in q:
            return {"has_temporal": True, "direction": direction, "needs_llm": False}

    if re.search(r'\b\d{1,2}[\s/-]\w+[\s/-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b', q):
        return {"has_temporal": True, "direction": "when", "needs_llm": False}

    # Ambiguous — let LLM handle
    ambiguous = ["recently", "lately", "the other day", "a while back",
                 "around the time", "the same week", "that weekend"]
    for phrase in ambiguous:
        if phrase in q:
            return {"has_temporal": True, "direction": None, "needs_llm": True}

    return {"has_temporal": False, "direction": None, "needs_llm": False}
