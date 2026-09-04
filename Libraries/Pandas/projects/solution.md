# Auditable data-quality pipeline solution notes

The reference pipeline copies selected columns, validates keys before aggregation, normalizes explicit dtypes, parses timestamps as UTC, rejects negative amounts, reports orphan transactions, uses a validated one-to-one feature join, retains customers without activity, and sorts output deterministically.

Run:

```bash
python3 -m unittest -v test_data_quality_pipeline.py
```

The zero transaction count and total are meaningful identities; median and timestamps remain missing because inventing them would misrepresent knowledge. Orphans are counted and excluded from customer features rather than silently disappearing. Production extensions should write rejected records, include input checksums, validate schema before loading full files, and make the orphan policy configurable and approved.
