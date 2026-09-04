import importlib.util
import tempfile
import unittest
from pathlib import Path
from model import record_offline_run, sanitize, validate_sweep


class WandbProjectTests(unittest.TestCase):
    def test_secret_redaction(self):
        result = sanitize({"lr": .1, "api_key": "do-not-log",
                           "nested": {"password": "also-secret"}})
        self.assertEqual(result["api_key"], "[REDACTED]"); self.assertEqual(result["lr"], .1)
        self.assertEqual(result["nested"]["password"], "[REDACTED]")

    def test_sweep_budget_and_metric(self):
        self.assertTrue(validate_sweep({"method": "random", "count": 3, "metric": {"name": "val"}}))
        with self.assertRaises(ValueError): validate_sweep({"method": "random", "count": 0})

    @unittest.skipUnless(importlib.util.find_spec("wandb"), "academy dependency not installed")
    def test_real_offline_run_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            result = record_offline_run(
                directory, {"lr": .1, "credentials": {"api_key": "never-log"}},
                validation_score=.9,
            )
            self.assertEqual(result["config"]["credentials"]["api_key"], "[REDACTED]")
            self.assertTrue(result["run_id"])
            self.assertTrue(Path(result["run_dir"]).is_dir())


if __name__ == "__main__": unittest.main()
