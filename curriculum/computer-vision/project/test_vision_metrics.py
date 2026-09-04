import unittest

from vision_metrics import area, intersection_over_union


class BoxMetricTests(unittest.TestCase):
    def test_identical_and_disjoint(self) -> None:
        self.assertEqual(intersection_over_union((0, 0, 2, 2), (0, 0, 2, 2)), 1.0)
        self.assertEqual(intersection_over_union((0, 0, 1, 1), (1, 1, 2, 2)), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(intersection_over_union((0, 0, 2, 2), (1, 1, 3, 3)), 1 / 7)

    def test_invalid_and_empty_boxes(self) -> None:
        with self.assertRaises(ValueError):
            area((2, 0, 1, 1))
        with self.assertRaises(ValueError):
            intersection_over_union((0, 0, 0, 0), (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
