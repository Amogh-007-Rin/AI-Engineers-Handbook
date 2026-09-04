---
title: SentenceTransformers embeddings similarity retrieval and evaluation
slug: sentence-transformers-foundations
level: practitioner
stage: nlp
estimated_hours: 14
prerequisites:
  - transformers-foundations
learning_objectives:
  - Define embedding dimension normalization and similarity contracts
  - Build retrieval evaluation with relevant hard negatives
  - Batch and cache embeddings without stale model or text identity
  - Audit semantic failure bias privacy latency and index drift
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: SentenceTransformers
supported_versions: 3.x
---

# SentenceTransformers foundations

An embedding contract includes model revision, pooling, dimension, dtype,
normalization, maximum input length, and similarity function. Cosine similarity
and dot product are equivalent only for normalized vectors. Never mix index and
query embeddings from incompatible versions without a migration test.

Evaluate retrieval using query/document groups and hard negatives that reflect
production confusion. Report recall/MRR/nDCG at relevant cutoffs, latency,
memory, empty/long text, multilingual slices, duplicates, and near-duplicates.
Cache by normalized text plus complete model configuration; protect embedded
sensitive data and define index deletion/rebuild behavior.

## Completion criteria

- [ ] Dimension, normalization, similarity, and revision are explicit.
- [ ] Retrieval metrics use held-out queries and hard negatives.
- [ ] Cache/index identities prevent mixed-version embeddings.
- [ ] Bias, privacy, latency, and drift risks are evaluated.
