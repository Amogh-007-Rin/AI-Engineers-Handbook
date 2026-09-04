---
title: Language models retrieval and speech systems
slug: nlp-language-systems
level: practitioner
stage: nlp-and-speech
estimated_hours: 14
prerequisites:
  - nlp-text-foundations
learning_objectives:
  - Compare sequence transformer retrieval and speech pipelines
  - Design evaluations for grounded language behavior
  - Identify multilingual privacy safety and latency risks
formats:
  - lesson
  - project
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Language models, retrieval, and speech systems

Sequence models compress or expose prior state; attention lets outputs weight relevant representations; transformers parallelize token interactions but require explicit positional information and careful context budgeting. Retrieval systems separate corpus preparation, candidate retrieval, reranking, and downstream use—evaluate each layer independently.

Speech recognition maps audio to text and may fail by accent, noise, device, language, or terminology. Synthesis quality includes intelligibility, naturalness, identity/consent, latency, and misuse risk. Text normalization must be consistent across training and evaluation without erasing dialect or meaningful form.

Grounded generation requires evidence attribution tests, unanswerable cases, contradiction handling, and human review. Never treat fluent wording as evidence of truth.

## Completion criteria

- [ ] Component metrics isolate retrieval/model/speech failures.
- [ ] Evaluation includes unanswerable and adversarial cases.
- [ ] Multilingual and accessibility behavior is measured.
- [ ] Privacy, consent, provenance, and abuse controls are documented.
