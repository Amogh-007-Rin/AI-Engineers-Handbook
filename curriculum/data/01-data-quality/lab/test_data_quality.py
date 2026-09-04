import math
import unittest

from data_quality import audit_events, inspect_event, split_overlap, validate_join


def row(**changes):
    value = {"event_id": "E1", "entity_id": "A", "observed_at": 10,
             "prediction_time": 10, "score": 0.5, "status": "complete"}
    value.update(changes)
    return value


class DataQualityTests(unittest.TestCase):
    def test_valid_event_is_normalized(self):
        event, issues = inspect_event(row(event_id=" E1 ", score=1))
        self.assertEqual(issues, ())
        self.assertEqual(event.event_id, "E1")
        self.assertEqual(event.score, 1.0)

    def test_schema_and_identity_are_explicit(self):
        event, issues = inspect_event({"event_id": "E1"})
        self.assertIsNone(event); self.assertEqual(issues, ("schema",))
        self.assertIn("identity", inspect_event(row(entity_id=""))[1])

    def test_range_finiteness_and_types_are_distinct(self):
        self.assertIn("range", inspect_event(row(score=1.1))[1])
        self.assertIn("nonfinite", inspect_event(row(score=math.nan))[1])
        self.assertIn("score_type", inspect_event(row(score="0.5"))[1])

    def test_temporal_and_cross_field_rules(self):
        self.assertIn("temporal_availability", inspect_event(row(observed_at=11))[1])
        self.assertIn("cross_field_completeness", inspect_event(row(score=None))[1])
        self.assertEqual(inspect_event(row(score=None, status="pending"))[1], ())

    def test_audit_reconciles_counts_and_duplicates(self):
        result = audit_events([row(), row(entity_id="B")])
        self.assertEqual(result["received"], result["accepted"] + result["quarantined"])
        self.assertEqual(result["issue_counts"], {"duplicate_event": 1})
        with self.assertRaisesRegex(ValueError, "non-empty"):
            audit_events([])

    def test_audit_example_has_multiple_issue_kinds(self):
        result = audit_events([row(score=1.2, observed_at=12), row(event_id="E2", score=None)])
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["quarantined"], 2)
        self.assertEqual(set(result["issue_counts"]), {"range", "temporal_availability", "cross_field_completeness"})

    def test_split_overlap_exposes_group_leakage(self):
        self.assertEqual(split_overlap(["A", "B"], ["B", "C"]), {"B"})
        self.assertEqual(split_overlap(["A"], ["C"]), set())

    def test_join_cardinality_and_expected_rows(self):
        self.assertEqual(validate_join(["A", "B"], ["A", "A"], "one-to-many"), 2)
        with self.assertRaisesRegex(ValueError, "right keys"):
            validate_join(["A"], ["A", "A"], "one-to-one")
        with self.assertRaisesRegex(ValueError, "left keys"):
            validate_join(["A", "A"], ["A"], "one-to-many")


if __name__ == "__main__":
    unittest.main()
