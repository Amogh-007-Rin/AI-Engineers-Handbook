# JAX pure training project

Run `python3 -m unittest -v test_model.py`. Extend the fixture into a batched MLP with explicit random keys and optimizer state. Test gradients, `vmap` parity, seeded initialization, invalid shapes, JIT recompilation behavior, and checkpoint tree round-trip. Report compile versus steady-state time with blocked execution. Passing requires 80/100 and no hidden mutation, reused random key, tracer-dependent Python branch, or asynchronous benchmark error.
