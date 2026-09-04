import unittest
from model import validate_job


class RayProjectTests(unittest.TestCase):
    def test_valid_job(self):
        self.assertTrue(validate_job({"name": "features", "retries": 2, "resources": {"cpu": 1}}))

    def test_failure_and_side_effect_gates(self):
        with self.assertRaises(ValueError): validate_job({"name": "x", "retries": -1, "resources": {"cpu": 1}})
        with self.assertRaises(ValueError): validate_job({"name": "x", "retries": 1, "resources": {"cpu": 1}, "side_effects": True})
        with self.assertRaises(ValueError): validate_job({"name": "x", "retries": 1, "resources": {"cpu": 0}})


if __name__ == "__main__": unittest.main()
