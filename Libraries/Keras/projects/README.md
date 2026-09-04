# Keras backend-portable training project

Run `KERAS_BACKEND=jax python3 -m unittest -v test_model.py`. Extend the fixture to a Functional-API classifier with `keras.ops`, validation checkpointing, early stopping, metrics, custom portable layer, and native round trip. Run equivalent tests on JAX and TensorFlow backends, document numerical tolerances and backend-only code, and compare compile/steady-state performance. Passing requires 80/100 and no test tuning, mode error, or undocumented backend dependency.
