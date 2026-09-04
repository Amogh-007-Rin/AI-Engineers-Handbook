# DuckDB analytical pipeline project

Run `python3 -m unittest -v test_analytics.py`. Extend the fixture to ingest versioned Parquet data, validate schemas and keys, create a feature table, export deterministic results, and record row-count/null/orphan audits. Compare equivalent Pandas and DuckDB workflows for correctness before runtime and memory. Include `EXPLAIN`, parameterization tests, persistence/concurrency limits, and an engine decision. A pass requires no silent row loss or join multiplication.
