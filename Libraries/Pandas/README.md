# Pandas Academy

> Status: Pilot · Core target: Pandas 2.x · Compute: CPU · Last verified: 2026-09-03

Pandas provides labeled, heterogeneous tabular data structures. This academy teaches explicit schemas, index alignment, tidy transformations, validation, performance, and reproducible analysis.

## Use it when

- Data fits on one machine and labels, mixed types, joins, time series, or exploratory transformations matter.
- You need broad interoperability with Python analytics and ML tools.

## Avoid it when

- Dense homogeneous numerical work dominates; prefer NumPy.
- Data volume or latency exceeds a single machine; evaluate Polars, DuckDB, Dask, or Spark using measurements.
- A database can perform the filtering or aggregation more safely and efficiently.

## Prerequisites

- [Python foundations](../../curriculum/foundations/01-python-foundations/README.md)
- [Data-quality foundations](../../curriculum/data/01-data-quality/README.md)
- [NumPy foundation](../NumPy/01-fundamentals/README.md)

## Outcomes

- **Foundation:** Construct, inspect, select, filter, and summarize Series and DataFrames without ambiguous chained operations.
- **Practitioner:** Build tested joins, groupings, reshaping, time-series, missing-data, and validation workflows.
- **Advanced:** Diagnose alignment, dtype, memory, copy, vectorization, and scaling problems and select alternatives honestly.

## Learning path

1. [Series, DataFrames, indexes, and selection](01-fundamentals/README.md)
2. [Cleaning, joining, grouping, and validation](02-core-workflows/README.md)
3. [Performance, memory, and production boundaries](04-advanced/README.md)
4. [Project and practical assessment](projects/data-quality-pipeline.md)

```bash
python -m pip install "pandas>=2,<3"
python -c "import pandas as pd; print(pd.__version__); print(pd.DataFrame({'x': [1, 2]}).sum())"
```

Official reference: [Pandas documentation](https://pandas.pydata.org/docs/).
