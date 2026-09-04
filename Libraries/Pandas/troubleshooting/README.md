# Pandas troubleshooting

- **Unexpected rows after a merge:** state each table’s grain, check key uniqueness, use `validate`, and inspect `indicator=True` before dropping anything.
- **Chained-assignment warning:** perform selection and assignment in one `.loc[...]` operation or make an intentional copy.
- **Unexpected missing values:** inspect index alignment, unmatched join keys, parsing failures, and the semantic meaning of absence.
- **Object dtype:** convert intentionally to string, category, numeric, Boolean, or datetime and test invalid values.
- **Dates shift:** parse and store timezone-aware timestamps; define the display timezone separately.
- **Slow row loop:** express column operations, joins, or group transforms; verify equality before benchmarking.
- **High memory:** read required columns, fix dtypes, filter early, measure deeply, and evaluate DuckDB/Polars/database execution.
- **Training/serving mismatch:** persist one tested feature transformation and validate schema/order at both boundaries.

Always reduce a failure to a tiny DataFrame showing index, dtypes, values, expected grain, and expected output.
