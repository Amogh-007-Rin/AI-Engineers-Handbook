import math
import unittest

from evaluation import (
    Confusion, bootstrap_accuracy, brier_score, choose_threshold, confusion,
    metrics, predictions, slice_report, threshold_cost, validate,
)


class EvaluationTests(unittest.TestCase):
    def test_validation_rejects_shape_labels_and_probabilities(self):
        for actual, probabilities in (([], []), ([1], []), ([2], [0.5]), ([True], [0.5]), ([1], [math.nan]), ([1], [1.1])):
            with self.subTest(actual=actual, probabilities=probabilities), self.assertRaises(ValueError):
                validate(actual, probabilities)

    def test_predictions_include_the_threshold(self):
        self.assertEqual(predictions([0.2, 0.5, 0.9], 0.5), (0, 1, 1))
        with self.assertRaisesRegex(ValueError, "threshold"):
            predictions([0.5], -0.1)

    def test_confusion_counts_each_case(self):
        result = confusion([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
        self.assertEqual(result, Confusion(1, 1, 1, 1))

    def test_metrics_preserve_undefined_denominators(self):
        result = metrics(Confusion(0, 0, 3, 2))
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["specificity"], 1.0)

    def test_brier_score_rewards_exact_probabilities(self):
        self.assertEqual(brier_score([0, 1], [0.0, 1.0]), 0.0)
        self.assertEqual(brier_score([0, 1], [1.0, 0.0]), 1.0)

    def test_threshold_selection_uses_declared_cost(self):
        actual, scores = [1, 1, 0, 0], [0.9, 0.65, 0.55, 0.1]
        self.assertEqual(choose_threshold(actual, scores, [0.4, 0.6, 0.8], 2, 3),
                         {"threshold": 0.6, "cost": 0})
        self.assertEqual(threshold_cost(actual, scores, 0.8, 2, 3), 3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            threshold_cost(actual, scores, 0.5, -1, 1)

    def test_bootstrap_is_seeded_and_validates_repetitions(self):
        first = bootstrap_accuracy([1, 0, 1], [0.9, 0.2, 0.4], 0.5, 100, 4)
        self.assertEqual(first, bootstrap_accuracy([1, 0, 1], [0.9, 0.2, 0.4], 0.5, 100, 4))
        self.assertLessEqual(first[0], first[1])
        with self.assertRaisesRegex(ValueError, "20"):
            bootstrap_accuracy([1], [0.9], 0.5, 5)

    def test_slice_report_includes_support_and_validates_alignment(self):
        result = slice_report([1, 0, 1], [0.9, 0.2, 0.3], ["b", "a", "b"], 0.5)
        self.assertEqual(list(result), ["a", "b"])
        self.assertEqual(result["b"]["support"], 2)
        with self.assertRaisesRegex(ValueError, "slice labels"):
            slice_report([1], [0.9], [], 0.5)


if __name__ == "__main__":
    unittest.main()
