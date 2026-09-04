# Flax explicit-state training project

Run `python3 -m unittest -v test_model.py`. Extend the fixture with dropout and batch normalization, an explicit optimizer state, named PRNG streams, train/eval steps, gradient checks, best-state restoration, and Orbax or supported checkpointing. Test mutable batch statistics, missing keys, shape changes, and clean restore. Passing requires 80/100 and no hidden state mutation or key reuse.
