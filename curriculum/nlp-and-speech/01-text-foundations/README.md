---
title: Text representation tokenization and evaluation
slug: nlp-text-foundations
level: foundation
stage: nlp-and-speech
estimated_hours: 10
prerequisites:
  - neural-architecture-families
learning_objectives:
  - Explain sparse dense and contextual text representations
  - Test tokenization normalization and truncation behavior
  - Select task metrics and analyze language errors by slice
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Text representation, tokenization, and evaluation

Text is converted into model inputs through choices that can discard meaning. Unicode normalization, casing, punctuation, whitespace, language identification, vocabulary, subword splitting, special tokens, padding, and truncation all belong to the model contract.

Bag-of-words and TF–IDF represent lexical evidence and remain strong transparent baselines. Static embeddings encode one vector per token; contextual models condition representations on surrounding tokens. Dense representations enable similarity search but similarity is not factual correctness or task relevance.

Evaluate the decision, not generic “language quality.” Classification needs class-aware metrics and calibration. Extraction needs exact and span-level analysis. Retrieval needs relevance judgments, recall at cutoffs, ranking measures, and negative quality. Generation often requires human criteria because lexical overlap misses meaning and fluent text may be unsupported.

## Exercise

Build a tokenizer contract test containing Unicode variants, emoji, mixed scripts, URLs, whitespace, empty text, long text, and domain terminology. Compare a whitespace baseline with one library tokenizer. Record round-trip limitations, unknown handling, token counts, truncation, and cost implications.

## Completion criteria

- [ ] Normalization decisions preserve raw text and are justified.
- [ ] Token and character offsets are tested where spans matter.
- [ ] Metrics match the downstream task.
- [ ] Slices include language, length, and domain-relevant groups.
