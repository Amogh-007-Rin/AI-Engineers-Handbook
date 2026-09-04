---
title: Gensim streaming corpora topic models and vector semantics
slug: gensim-foundations
level: practitioner
stage: nlp
estimated_hours: 14
prerequisites:
  - nlp-text-foundations
learning_objectives:
  - Stream sparse corpora without materializing the entire dataset
  - Build dictionaries from training documents and control rare tokens
  - Train and evaluate topic and vector models with seeded experiments
  - Persist models and verify vocabulary and inference after reload
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Gensim
supported_versions: 4.4.x
---

# Gensim foundations

Gensim is designed around streamed documents and sparse vector spaces. A
corpus is an iterable of bag-of-words vectors; a `Dictionary` maps tokens to
integer identifiers. Fit that mapping on training documents only. Filtering
rare or ubiquitous tokens changes every downstream vector, so persist the
dictionary with the model and record the filtering thresholds.

Bag-of-words discards order and context. TF-IDF changes document weighting;
LSI projects into a linear latent space; LDA estimates a topic mixture;
Word2Vec learns distributional neighborhoods. None produces inherently human
topics or unbiased semantic similarity. Choose a representation from the task
contract and compare it with a simple lexical baseline.

Topic-model evaluation needs more than coherence: inspect stability across
seeds, topic prevalence, representative and counterexample documents, and
downstream usefulness. Vector evaluation must guard against vocabulary leakage
and social bias. Save/load tests should cover token IDs, dimensions,
out-of-vocabulary behavior, and inference tolerances.

## Lab sequence

1. Tokenize documents and build a train-only dictionary.
2. Convert a reiterable corpus to sparse bag-of-words vectors.
3. Fit TF-IDF and compare common versus discriminative terms.
4. Test unknown-token behavior on validation text.
5. Save and reload dictionary and model artifacts.

## Completion criteria

- [ ] Corpus iteration is memory bounded and repeatable.
- [ ] Dictionary filtering and OOV policy are documented.
- [ ] Evaluation includes stability and a lexical baseline.
- [ ] Artifact reload preserves IDs, dimensions, and outputs.
