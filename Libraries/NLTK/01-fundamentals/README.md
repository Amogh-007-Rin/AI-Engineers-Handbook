---
title: NLTK corpora tokenization tagging and evaluation
slug: nltk-foundations
level: foundation
stage: nlp
estimated_hours: 10
prerequisites:
  - nlp-text-foundations
learning_objectives:
  - Normalize and tokenize text without silently discarding meaning
  - Manage NLTK data packages as versioned dependencies
  - Build corpus statistics and classical NLP baselines
  - Evaluate token and document outputs on adversarial examples
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: NLTK
supported_versions: 3.9.x
---

# NLTK foundations

NLTK is a teaching toolkit and a practical source of classical NLP algorithms,
corpus readers, and linguistic resources. Begin by defining the unit your task
needs: Unicode code point, token, sentence, document, or corpus. Lowercasing,
stemming, punctuation removal, and stop-word filtering are modeling decisions;
they can erase negation, names, code, emoji, or language cues.

Many tokenizers and taggers require separately downloaded data. Pin those
resources, place them in a controlled data directory, and fail with an
actionable message when missing. For offline and security-sensitive systems,
do not download during request handling. The project uses `RegexpTokenizer` so
its base test has no hidden network dependency.

Classical frequency and n-gram baselines remain useful sanity checks. Split by
document, speaker, or time before building vocabulary. Report out-of-vocabulary
behavior and inspect contractions, URLs, apostrophes, numbers, empty text, and
non-English samples.

## Completion criteria

- [ ] Normalization choices are justified with counterexamples.
- [ ] Resource versions and licenses are recorded.
- [ ] Train-only vocabulary construction is tested.
- [ ] A baseline and error taxonomy accompany the score.
