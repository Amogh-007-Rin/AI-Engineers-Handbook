import unittest
from inference import fit_line, prediction_at


class InferenceTests(unittest.TestCase):
    def test_exact_line_coefficients_and_prediction(self) -> None:
        result = fit_line([0, 1, 2, 3], [1, 3, 5, 7])
        self.assertAlmostEqual(result.params[0], 1)
        self.assertAlmostEqual(result.params[1], 2)
        mean, lower, upper = prediction_at(result, 4)
        self.assertAlmostEqual(mean, 9)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)

    def test_invalid_pairs_fail(self) -> None:
        with self.assertRaises(ValueError):
            fit_line([1], [2])
        with self.assertRaises(ValueError):
            fit_line([1, 2, 3], [1, 2])


if __name__ == "__main__":
    unittest.main()
