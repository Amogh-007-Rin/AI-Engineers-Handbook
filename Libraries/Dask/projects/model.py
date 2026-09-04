"""Small lazy Dask reduction with explicit partition contract."""

import dask.array as da


def mean_range(stop, chunks=4):
    if stop <= 0 or chunks <= 0:
        raise ValueError("stop and chunks must be positive")
    array = da.arange(stop, chunks=chunks)
    return array.mean()


def graph_size(stop, chunks=4):
    return len(mean_range(stop, chunks).dask)
