# NumPy troubleshooting

- **Broadcasting error:** write both shapes right-aligned and locate the first pair that is neither equal nor one.
- **Unexpected extra dimensions:** inspect integer indexing versus slices and use `keepdims=True` when preserving a reduction axis is intentional.
- **Changes affect another array:** check `np.shares_memory`; basic slices are commonly views.
- **Silent overflow or truncation:** inspect dtype before arithmetic/casting and test boundary values.
- **NaN or infinity:** assert finite inputs, then isolate division, logarithm, exponentiation, and reductions over empty data.
- **Vectorized code uses too much memory:** identify temporary arrays, use in-place work only when safe, or process chunks.
- **Unexpected copy at integration boundary:** inspect contiguity, strides, dtype, device, and ownership requirements.

Reduce failures to a small array with explicit values, shape, dtype, and expected output before seeking help.
