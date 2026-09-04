"""Small Gensim TF-IDF pipeline with explicit dictionary ownership."""

from pathlib import Path
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess


def tokenize(documents):
    return [simple_preprocess(document, deacc=True) for document in documents]


def train(documents):
    texts = tokenize(documents)
    dictionary = Dictionary(texts)
    if not dictionary:
        raise ValueError("training documents produced an empty vocabulary")
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, TfidfModel(corpus, normalize=True)


def transform(dictionary, model, document):
    vector = model[dictionary.doc2bow(simple_preprocess(document, deacc=True))]
    return [(dictionary[token_id], float(weight)) for token_id, weight in vector]


def save(dictionary, model, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(directory / "dictionary.gensim"))
    model.save(str(directory / "tfidf.gensim"))


def load(directory):
    directory = Path(directory)
    return Dictionary.load(str(directory / "dictionary.gensim")), TfidfModel.load(str(directory / "tfidf.gensim"))
