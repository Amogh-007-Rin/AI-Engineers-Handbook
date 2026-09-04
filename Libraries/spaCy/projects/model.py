"""Offline spaCy entity pipeline with a serializable output contract."""

from pathlib import Path
import spacy


PATTERNS = [
    {"label": "LANGUAGE", "pattern": "Python"},
    {"label": "LIBRARY", "pattern": [{"LOWER": "scikit"}, {"ORTH": "-"}, {"LOWER": "learn"}]},
]


def build_pipeline():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(PATTERNS)
    return nlp


def annotate(nlp, texts):
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of documents, not one string")
    return [
        {"text": doc.text, "tokens": [t.text for t in doc],
         "entities": [(e.text, e.label_, e.start_char, e.end_char) for e in doc.ents]}
        for doc in nlp.pipe(texts)
    ]


def save(nlp, path):
    path = Path(path)
    nlp.to_disk(path)
    return path


def load(path):
    return spacy.load(Path(path))
