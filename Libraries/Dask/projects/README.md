# Dask lazy-computation contract

Construct a lazy array reduction, inspect its task graph, compute it, and test
partition validation. Run `python -W error -m unittest -v`; extend it with a
partitioned dataframe and bounded write path.
