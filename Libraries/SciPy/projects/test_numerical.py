import math
import unittest
from numerical import constrained_minimum, normal_mass, positive_root


class NumericalTests(unittest.TestCase):
    def test_root_and_residual(self) -> None:
        root = positive_root(2)
        self.assertAlmostEqual(root, math.sqrt(2), places=10)
        self.assertAlmostEqual(root * root, 2, places=10)

    def test_constraint_is_active(self) -> None:
        point, objective = constrained_minimum()
        self.assertAlmostEqual(point, 2)
        self.assertAlmostEqual(objective, 1)

    def test_integration_error_contract(self) -> None:
        value, error = normal_mass()
        self.assertAlmostEqual(value, 0.682689492, places=8)
        self.assertLess(error, 1e-10)

    def test_invalid_root_input(self) -> None:
        with self.assertRaises(ValueError):
            positive_root(-1)


if __name__ == "__main__":
    unittest.main()
