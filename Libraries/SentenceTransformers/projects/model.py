"""SentenceTransformers similarity contract without model downloads."""

import torch
from sentence_transformers.util import cos_sim, normalize_embeddings


def similarities(queries, documents):
    queries, documents = torch.as_tensor(queries, dtype=torch.float32), torch.as_tensor(documents, dtype=torch.float32)
    if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
        raise ValueError("query/document embeddings must be 2D with equal dimensions")
    if not torch.isfinite(queries).all() or not torch.isfinite(documents).all():
        raise ValueError("embeddings must be finite")
    return cos_sim(normalize_embeddings(queries), normalize_embeddings(documents))
