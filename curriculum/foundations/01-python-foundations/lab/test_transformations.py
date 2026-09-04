import math
import unittest

from transformations import Observation, iter_observations, parse_observation, summarize


class TransformationTests(unittest.TestCase):
    def test_parse_normalizes_and_types_a_record(self):
        result = parse_observation({"subject_id": " A ", "group": "Treatment", "score": "0.75"})
        self.assertEqual(result, Observation("A", "treatment", 0.75))

    def test_parse_rejects_identifier_and_group_boundaries(self):
        with self.assertRaisesRegex(ValueError, "subject_id"):
            parse_observation({"subject_id": " ", "group": "control", "score": 1})
        with self.assertRaisesRegex(ValueError, "group"):
            parse_observation({"subject_id": "A", "group": "unknown", "score": 1})

    def test_parse_preserves_numeric_failure_context(self):
        with self.assertRaisesRegex(ValueError, "score must be numeric") as raised:
            parse_observation({"subject_id": "A", "group": "control", "score": "high"})
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_parse_rejects_missing_and_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "score is required"):
            parse_observation({"subject_id": "A", "group": "control"})
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                parse_observation({"subject_id": "A", "group": "control", "score": value})

    def test_iterator_is_lazy_and_rejects_duplicates_when_consumed(self):
        records = [
            {"subject_id": "A", "group": "control", "score": 1},
            {"subject_id": "A", "group": "treatment", "score": 2},
        ]
        iterator = iter_observations(records)
        self.assertEqual(next(iterator).subject_id, "A")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            next(iterator)

    def test_summary_is_sorted_and_rejects_empty_input(self):
        result = summarize([
            Observation("B", "treatment", 0.8),
            Observation("A", "control", 0.4),
            Observation("C", "treatment", 1.0),
        ])
        self.assertEqual(list(result), ["control", "treatment"])
        self.assertEqual(result["treatment"], {"count": 2, "mean": 0.9})
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
