# Transparent linear-algebra lab

This lab favors visible contracts over library speed. Predict every shape and
hand-calculate at least one result before executing:

```bash
python3 curriculum/mathematics/01-linear-algebra/lab/linear_algebra.py
python3 -m unittest discover -s curriculum/mathematics/01-linear-algebra/lab -v
```

Expected predictions are `[2.5, -3.0, 1.5]`; projection of `(3,4)` onto the
first axis is `(3,0)`; the sensitivity ratio exceeds 100,000. Eight tests pass.

## Investigation

Trace `matmul` from output entry to a row/column dot product. Then perturb the
near-dependent system's second target by `1e-8`, `1e-7`, and `1e-6`. Record
input and output change using scientific notation. The example uses a tighter
solver tolerance deliberately so the ill-conditioned system can be studied;
that is not a universal production default.

Trigger ragged, empty, non-finite, mismatched, zero-direction, and singular
inputs. Preserve one traceback and explain why validation occurs at that layer.

## Limitations

The code is educational, not a replacement for NumPy/LAPACK. It lacks general
factorizations, dtype control, optimized kernels, robust scaling, sparse data,
and error-bound estimates. It demonstrates semantics and tests; it does not
benchmark production numerical quality.
