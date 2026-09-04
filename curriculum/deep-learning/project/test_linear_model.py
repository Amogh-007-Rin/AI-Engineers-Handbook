import unittest

from linear_model import LinearModel, loss_and_gradients, train


class LinearModelTests(unittest.TestCase):
    def test_analytical_gradient_matches_centered_difference(self) -> None:
        model = LinearModel(0.4, -0.2)
        xs, ys, h = [1.0, 2.0], [3.0, 5.0], 1e-6
        _, gradient, _ = loss_and_gradients(model, xs, ys)
        plus = loss_and_gradients(LinearModel(model.weight + h, model.bias), xs, ys)[0]
        minus = loss_and_gradients(LinearModel(model.weight - h, model.bias), xs, ys)[0]
        self.assertAlmostEqual(gradient, (plus - minus) / (2 * h), places=6)

    def test_training_learns_line(self) -> None:
        model = LinearModel()
        history = train(model, [-1.0, 0.0, 1.0], [-1.0, 1.0, 3.0], 0.1, 100)
        self.assertLess(history[-1], history[0])
        self.assertAlmostEqual(model.weight, 2.0, places=3)
        self.assertAlmostEqual(model.bias, 1.0, places=3)

    def test_invalid_training_input_fails(self) -> None:
        with self.assertRaises(ValueError):
            loss_and_gradients(LinearModel(), [], [])
        with self.assertRaises(ValueError):
            train(LinearModel(), [1.0], [1.0], 0.0, 1)


if __name__ == "__main__":
    unittest.main()
