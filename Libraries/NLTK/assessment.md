# NLTK assessment

Passing score: 80/100, with no leakage or runtime downloads.

- 25 points: tokenizer behavior is specified and adversarially tested.
- 20 points: corpus assets, versions, licenses, and offline setup are recorded.
- 20 points: vocabulary and transforms are fitted on training records only.
- 20 points: baseline comparison and error taxonomy are reproducible.
- 15 points: code quality, deterministic execution, and concise documentation.

Automatic fail conditions: test data changes vocabulary, requested resources
download during inference, or preprocessing silently drops a protected signal.
