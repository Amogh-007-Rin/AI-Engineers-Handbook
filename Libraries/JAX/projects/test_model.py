import unittest
import jax
import jax.numpy as jnp
from model import data, loss, train


class JAXTests(unittest.TestCase):
    def test_training(self):
        params, value = train(); self.assertLess(value, 1e-6)
        self.assertAlmostEqual(float(params["weight"]), 2, places=3)
        self.assertAlmostEqual(float(params["bias"]), 1, places=3)

    def test_gradient_matches_finite_difference(self):
        x, y = data(); params = {"weight": jnp.array(.4), "bias": jnp.array(-.2)}
        gradient = jax.grad(loss)(params, x, y)["weight"]
        h = 1e-3
        plus = loss({**params, "weight": params["weight"] + h}, x, y)
        minus = loss({**params, "weight": params["weight"] - h}, x, y)
        self.assertAlmostEqual(float(gradient), float((plus - minus) / (2*h)), places=3)


if __name__ == "__main__": unittest.main()
