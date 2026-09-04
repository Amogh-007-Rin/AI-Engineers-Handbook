import unittest
from reliability import availability, burn_rate, error_budget


class ReliabilityTests(unittest.TestCase):
    def test_metrics(self) -> None:
        self.assertEqual(availability(99, 100), 0.99)
        self.assertAlmostEqual(error_budget(10_000, 0.999), 10)
        self.assertAlmostEqual(burn_rate(20, 10_000, 0.999), 2)

    def test_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            availability(2, 1)


if __name__ == "__main__":
    unittest.main()
