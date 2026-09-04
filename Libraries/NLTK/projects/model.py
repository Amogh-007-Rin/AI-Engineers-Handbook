"""Offline-safe NLTK token frequency baseline."""

from collections import Counter
from nltk.tokenize import RegexpTokenizer

TOKENIZER = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def tokens(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [token.casefold() for token in TOKENIZER.tokenize(text)]


def vocabulary(documents, minimum_count=1):
    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    counts = Counter(token for document in documents for token in tokens(document))
    return {token: count for token, count in sorted(counts.items()) if count >= minimum_count}
