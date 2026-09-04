import unittest
from quality import audit


SCHEMA = {"id": int, "score": float}


class QualityTests(unittest.TestCase):
    def test_valid_and_duplicate_rows(self):
        self.assertTrue(audit([{"id": 1, "score": 2.0}], SCHEMA, "id")["valid"])
        result = audit([{"id": 1, "score": 2.0}, {"id": 1, "score": 3.0}], SCHEMA, "id")
        self.assertEqual(result["issues"][0]["kind"], "duplicate")

    def test_schema_and_nonfinite(self):
        self.assertFalse(audit([{"id": 1, "other": 2.0}], SCHEMA, "id")["valid"])
        self.assertFalse(audit([{"id": 1, "score": float("nan")}], SCHEMA, "id")["valid"])
        with self.assertRaises(ValueError): audit([], SCHEMA, "id")


if __name__ == "__main__": unittest.main()
