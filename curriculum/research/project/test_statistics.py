import unittest
from statistics import mean, paired_differences, sample_standard_deviation


class ExperimentStatisticsTests(unittest.TestCase):
    def test_summary(self) -> None:
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(sample_standard_deviation([1, 2, 3]), 1)

    def test_pairing(self) -> None:
        self.assertEqual(paired_differences([1, 3], [2, 2]), [1, -1])
        with self.assertRaises(ValueError):
            paired_differences([1], [])


if __name__ == "__main__":
    unittest.main()
