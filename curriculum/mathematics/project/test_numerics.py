import unittest
from numerics import central_difference, dot


class NumericsTests(unittest.TestCase):
    def test_dot_and_shape_contract(self):
        self.assertEqual(dot([1, 2], [3, 4]), 11)
        with self.assertRaises(ValueError): dot([1], [1, 2])

    def test_gradient_verification(self):
        self.assertAlmostEqual(central_difference(lambda x: 3 * x * x + 2, 4), 24, places=6)
        with self.assertRaises(ValueError): central_difference(lambda x: x, 1, 0)


if __name__ == "__main__": unittest.main()
