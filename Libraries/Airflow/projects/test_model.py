import unittest
from model import validate_dag


GOOD = {"dag_id": "daily_features", "owner": "ml-platform", "schedule": "0 3 * * *", "timezone": "UTC",
        "tasks": {"extract": {"retries": 2, "timeout_s": 300},
                  "publish": {"retries": 1, "timeout_s": 120, "side_effects": True, "idempotent": True}},
        "dependencies": [("extract", "publish")]}


class AirflowProjectTests(unittest.TestCase):
    def test_valid_dag(self): self.assertTrue(validate_dag(GOOD))

    def test_retry_and_dependency_gates(self):
        bad = {**GOOD, "tasks": {**GOOD["tasks"], "publish": {"retries": 1, "timeout_s": 10, "side_effects": True}}}
        with self.assertRaises(ValueError): validate_dag(bad)
        with self.assertRaises(ValueError): validate_dag({**GOOD, "dependencies": [("missing", "publish")]})


if __name__ == "__main__": unittest.main()
