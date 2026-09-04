---
title: Grounded RAG evaluation project
slug: grounded-rag-project
level: advanced
stage: generative-ai
estimated_hours: 20
prerequisites:
  - rag-grounded-evaluation
learning_objectives:
  - Build a provenance-preserving retrieval and answer pipeline
  - Measure retrieval grounding refusal latency and cost
  - Test prompt injection permissions and stale evidence failures
formats:
  - project
  - assessment
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Grounded RAG evaluation project

Run `python3 -m unittest -v test_grounding.py`. Build a small RAG system over versioned, openly licensed documents. Include lexical retrieval baseline, relevance judgments, chunk provenance, access filtering, answer citations, and refusal behavior.

Evaluate retrieval recall/ranking, answer correctness, citation precision/coverage, unsupported claims, unanswerable questions, conflicting/stale documents, prompt injection, latency, and token/cost estimates. Passing requires 80/100 across data/provenance, retrieval, grounding, security, and reproducibility; leaked restricted content or fabricated supporting citations require revision.
