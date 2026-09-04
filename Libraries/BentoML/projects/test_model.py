import unittest
from model import validate_service


GOOD = {"name": "predict", "model_ref": "model@v3", "input_schema": {"features": "float[]"},
        "output_schema": {"score": "float"}, "timeout_ms": 500, "max_batch": 16,
        "readiness": "/readyz", "metrics": ["latency_ms"]}


class BentoMLProjectTests(unittest.TestCase):
    def test_valid_contract(self): self.assertTrue(validate_service(GOOD))

    def test_version_and_limits(self):
        for key, value in (("model_ref", "model"), ("timeout_ms", 0), ("max_batch", 0)):
            with self.assertRaises(ValueError): validate_service({**GOOD, key: value})

    def test_operational_fields(self):
        with self.assertRaises(ValueError): validate_service({**GOOD, "metrics": []})


if __name__ == "__main__": unittest.main()
