# Gensim assessment

Passing score: 80/100; vocabulary leakage and non-reiterable corpora are gates.

- 25 points: bounded, repeatable corpus and dictionary construction.
- 20 points: explicit filtering, OOV, and feature-identity policies.
- 20 points: seeded evaluation with stability and a simple baseline.
- 20 points: dictionary/model persistence and reload verification.
- 15 points: memory evidence, error analysis, and documentation.

Automatic fail conditions: validation mutates the dictionary, token IDs change
after reload, or semantic claims are made from a single qualitative example.
