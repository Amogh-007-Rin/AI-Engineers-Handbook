# Event data-quality lab

This dependency-free lab separates row inspection, batch reconciliation, split
overlap, and join cardinality. Run from the repository root:

```bash
python3 curriculum/data/01-data-quality/lab/data_quality.py
python3 -m unittest discover -s curriculum/data/01-data-quality/lab -v
```

Expected output reports four received rows, two accepted, two quarantined, and
one issue each for `range`, `temporal_availability`, and
`cross_field_completeness`. Eight tests pass.

## Investigation

Predict which issues can coexist on one row. Trigger every validator branch and
preserve one quarantined example with invented, non-personal values. Create a
many-to-many join where two rows on each side produce four joined rows; then
declare one-to-one and confirm the contract fails before the join.

## Limitations

The fixture demonstrates contracts, not population quality. It does not infer
missingness mechanisms, detect semantic duplicates, authorize data use, measure
representation, or replace a schema engine. A production pipeline also needs
lineage storage, privacy controls, monitoring, ownership, and incident response.
