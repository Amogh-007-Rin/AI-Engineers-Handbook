# Polars lazy feature pipeline project

Run `python3 -m unittest -v test_lazy_pipeline.py`. Extend the pipeline to scan versioned Parquet, enforce schemas, audit orphans, normalize null/NaN values, create time-bounded features, and write deterministic output. Inspect optimized plans and compare correctness, runtime, and peak memory with Pandas or DuckDB. Passing requires no silent row loss, join multiplication, eager full-data materialization without justification, or unsupported performance claim.
