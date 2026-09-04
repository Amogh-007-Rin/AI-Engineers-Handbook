# spaCy assessment

Passing score: 80/100; pipeline-order and artifact-round-trip tests are gates.

- 25 points: token, span, label, and component contracts.
- 20 points: leakage-safe evaluation with per-label boundary errors.
- 20 points: batched inference and representative edge cases.
- 20 points: versioned pipeline packaging and clean reload.
- 15 points: decision log, performance evidence, and maintainable tests.

Automatic fail conditions: hidden model downloads, offsets not checked, or a
reported aggregate score without label-level error analysis.
