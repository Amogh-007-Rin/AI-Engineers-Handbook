import unittest
import jax
from flax import serialization
from model import data, train


class FlaxTests(unittest.TestCase):
    def test_training_and_state_round_trip(self):
        model, params, value = train(); self.assertLess(value, 1e-5)
        encoded = serialization.to_bytes(params)
        restored = serialization.from_bytes(params, encoded)
        x, _ = data()
        self.assertTrue(jax.numpy.allclose(model.apply({"params": params}, x), model.apply({"params": restored}, x)))

    def test_seeded_initialization_and_training(self):
        _, first, _ = train(5); _, second, _ = train(5)
        self.assertTrue(all(jax.numpy.allclose(a, b) for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)))

if __name__ == "__main__": unittest.main()
