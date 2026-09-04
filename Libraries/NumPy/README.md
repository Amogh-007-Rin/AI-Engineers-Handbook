# NumPy Academy

> Status: Pilot · Core target: NumPy 2.x · Compute: CPU · Last verified: 2026-09-03

NumPy provides n-dimensional arrays and numerical operations that underpin much of Python’s scientific and ML ecosystem. This academy emphasizes shape reasoning, correctness, vectorization, memory, and interoperability—not API memorization.

## Use it when

- You need dense numerical arrays, vectorized calculations, linear algebra, random sampling, or interoperability with scientific Python.
- You are implementing mathematical ideas or preparing numeric data for another ML framework.

## Avoid it when

- Labeled heterogeneous tables are central; begin with [Pandas](../Pandas/README.md) or Polars.
- Data exceeds one machine or requires lazy distributed execution; evaluate Dask, Spark, or Ray.
- You need automatic differentiation and accelerator training; use PyTorch, TensorFlow, or JAX.

## Prerequisites

- [Python foundations](../../curriculum/foundations/01-python-foundations/README.md)
- [Linear-algebra foundations](../../curriculum/mathematics/01-linear-algebra/README.md)

## Outcomes

- **Foundation:** Construct, inspect, index, reshape, and combine arrays while predicting shapes and dtypes.
- **Practitioner:** Build correct vectorized pipelines and test broadcasting, missing-value, and numerical edge cases.
- **Advanced:** Reason about views, copies, strides, memory layout, numerical stability, profiling, and interoperability.

## Learning path

1. [Arrays, shapes, dtypes, and indexing](01-fundamentals/README.md)
2. [Vectorization, broadcasting, and numerical reliability](02-core-workflows/README.md)
3. [Memory, performance, and interoperability](04-advanced/README.md)
4. [Project and practical assessment](projects/vectorized-features.md)

Install and verify in an isolated environment:

```bash
python -m pip install "numpy>=2,<3"
python -c "import numpy as np; print(np.__version__); print(np.arange(3) ** 2)"
```

Official reference: [NumPy documentation](https://numpy.org/doc/stable/).
