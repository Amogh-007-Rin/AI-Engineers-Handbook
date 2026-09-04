---
title: spaCy pipelines documents components and production packaging
slug: spacy-foundations
level: practitioner
stage: nlp
estimated_hours: 14
prerequisites:
  - nlp-text-foundations
learning_objectives:
  - Explain token document vocabulary and pipeline component contracts
  - Build deterministic rule and statistical processing pipelines
  - Evaluate span and document predictions without train test leakage
  - Package and reload a pipeline with versioned metadata
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: spaCy
supported_versions: 3.8.x
---

# spaCy foundations

spaCy represents text as a `Doc` whose tokens reference a shared `Vocab`.
Pipeline components read and write annotations such as sentence boundaries,
parts of speech, entities, and custom extensions. Component order is therefore
an interface: declare the attributes each component requires and assigns, then
test the complete ordered pipeline rather than calling components in isolation.

Start with `spacy.blank(language)` and rule-based components. This keeps the
tokenization and annotation contract visible before adding downloaded models.
Use `nlp.pipe` for batches, disable unused components deliberately, and avoid
thread-unsafe mutation during inference. Statistical packages are separate
versioned artifacts; installation and licensing belong in environment setup,
not request handling.

For span tasks, exact-match scores can hide boundary and label failures. Report
precision, recall, and F1 by label, inspect overlapping/nested cases, and split
by source or entity identity when memorization is possible. Package the whole
pipeline with metadata and reload it in a clean process. Test tokens, span
offsets, labels, and custom attributes—not just rendered text.

## Lab sequence

1. Compare token boundaries on contractions, punctuation, URLs, and Unicode.
2. Add an `EntityRuler` and validate character offsets.
3. Stream a batch through `nlp.pipe` and preserve document order.
4. Compute label-level span errors on a held-out fixture.
5. Save and reload the pipeline, then compare structured outputs.

## Completion criteria

- [ ] Component requirements, assignments, and order are documented.
- [ ] Tests include empty, Unicode, and overlapping-pattern inputs.
- [ ] Evaluation reports boundary and label errors separately.
- [ ] A disk round trip preserves the annotation contract.
