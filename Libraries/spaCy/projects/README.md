# spaCy entity-pipeline contract

Build an offline English pipeline, add deterministic entity rules, batch texts,
and verify exact character offsets after a disk round trip. Run from this
directory with `python -W error -m unittest -v`.

Extend the project by introducing a conflicting rule, defining precedence, and
computing exact-span precision/recall against a held-out JSONL fixture.
