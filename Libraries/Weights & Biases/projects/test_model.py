import unittest
from model import sanitize, validate_sweep


class WandbProjectTests(unittest.TestCase):
    def test_secret_redaction(self):
        result = sanitize({"lr": .1, "api_key": "do-not-log"})
        self.assertEqual(result["api_key"], "[REDACTED]"); self.assertEqual(result["lr"], .1)

    def test_sweep_budget_and_metric(self):
        self.assertTrue(validate_sweep({"method": "random", "count": 3, "metric": {"name": "val"}}))
        with self.assertRaises(ValueError): validate_sweep({"method": "random", "count": 0})


if __name__ == "__main__": unittest.main()
