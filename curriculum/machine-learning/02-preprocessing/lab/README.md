# Leakage-safe preprocessing lab

The lab makes fitted state and fixed output order visible. Run from the
repository root:

```bash
python3 curriculum/machine-learning/02-preprocessing/lab/preprocessing.py
python3 -m unittest discover -s curriculum/machine-learning/02-preprocessing/lab -v
```

Expected state has median `30`, scale `10`, and ordered categories `basic` and
`pro`. An unseen plan activates `plan=other`; a missing age activates
`age_missing`. Eight tests pass.

## Investigation

Append `{"age": 1000, "plan": "enterprise"}` to validation data and compare
features when state is correctly frozen versus incorrectly refitted. Trigger
every schema, numeric, category, and artifact failure. Preserve the feature-name
tuple with results so a consumer cannot silently reinterpret column positions.

## Limitations

Max-deviation scaling was chosen for transparent arithmetic, not as a universal
recommendation. The JSON format avoids executable deserialization but still
needs integrity, provenance, access control, and schema evolution. The fixture
does not prove population validity or train-serving parity beyond tested cases.
