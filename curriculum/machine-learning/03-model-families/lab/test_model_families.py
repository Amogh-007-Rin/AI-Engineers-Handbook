import math
import unittest

from model_families import fit_line, fit_stump, kmeans_1d, knn_classify


class ModelFamilyTests(unittest.TestCase):
    def test_line_recovers_a_known_relationship(self):
        line = fit_line([1, 2, 3], [2, 4, 6])
        self.assertEqual((line.intercept, line.slope, line.predict(4)), (0.0, 2.0, 8.0))

    def test_line_rejects_invalid_shape_and_constant_features(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            fit_line([1], [1, 2])
        with self.assertRaisesRegex(ValueError, "constant"):
            fit_line([1, 1], [1, 2])

    def test_numeric_inputs_must_be_nonempty_and_finite(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            fit_line([], [])
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_line([1, math.nan], [1, 2])

    def test_knn_uses_local_neighbors_and_stable_ties(self):
        self.assertEqual(knn_classify([0, 1, 9, 10], ["cold", "cold", "warm", "warm"], 8, 3), "warm")
        self.assertEqual(knn_classify([0, 2], ["z", "a"], 1, 2), "a")

    def test_knn_validates_k_and_alignment(self):
        with self.assertRaisesRegex(ValueError, "between one"):
            knn_classify([1], ["a"], 1, 0)
        with self.assertRaisesRegex(ValueError, "equal"):
            knn_classify([1], [], 1, 1)

    def test_stump_finds_a_separating_threshold(self):
        stump = fit_stump([0, 1, 2, 3], [0, 0, 1, 1])
        self.assertEqual((stump.threshold, stump.errors), (1.5, 0))
        self.assertEqual([stump.predict(value) for value in (1, 2)], [0, 1])

    def test_stump_rejects_invalid_labels_and_constant_features(self):
        with self.assertRaisesRegex(ValueError, "zero or one"):
            fit_stump([0, 1], [0, 2])
        with self.assertRaisesRegex(ValueError, "two unique"):
            fit_stump([1, 1], [0, 1])

    def test_kmeans_is_seeded_ordered_and_validated(self):
        first = kmeans_1d([0, 1, 9, 10], 2, seed=4)
        self.assertEqual(first, (0.5, 9.5))
        self.assertEqual(first, kmeans_1d([0, 1, 9, 10], 2, seed=4))
        with self.assertRaisesRegex(ValueError, "unique"):
            kmeans_1d([1, 1], 2)
        with self.assertRaisesRegex(ValueError, "iteration"):
            kmeans_1d([0, 1], 1, max_iterations=0)


if __name__ == "__main__":
    unittest.main()
