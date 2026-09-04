---
title: Retrieval augmented generation and grounded evaluation
slug: rag-grounded-evaluation
level: advanced
stage: generative-ai
estimated_hours: 16
prerequisites:
  - foundation-model-systems
learning_objectives:
  - Design ingestion retrieval reranking and generation components
  - Evaluate retrieval and answer grounding independently
  - Defend RAG systems against untrusted content and stale evidence
formats:
  - lesson
  - project
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Retrieval-augmented generation and grounded evaluation

A RAG system has independently fallible stages: acquire and parse sources, segment them, attach provenance, index representations, retrieve candidates, rerank, assemble context, generate, cite, and validate. End-to-end answer scores cannot reveal which stage failed.

Build relevance judgments and measure retrieval recall/ranking at the context budget. For generation, measure answer correctness, evidence entailment, citation precision/coverage, refusal on insufficient evidence, and robustness to conflicting or malicious documents. Evaluate freshness and access controls at retrieval time.

Retrieved text is untrusted data, not an instruction channel. Preserve trust boundaries, sanitize active content, enforce document permissions before retrieval, label provenance, constrain tools outside the model, and test prompt injection and data-exfiltration attempts.

## Completion criteria

- [ ] Corpus version, chunk lineage, and permissions are traceable.
- [ ] Retrieval and generation have separate test sets and metrics.
- [ ] Unanswerable and conflicting-evidence cases are required.
- [ ] Citations are checked against actual supporting passages.
- [ ] Injection tests cannot expand tool or data permissions.
