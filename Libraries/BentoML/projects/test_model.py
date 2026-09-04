import importlib.util
import unittest
from model import build_native_service, predict_score, validate_service


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

    def test_input_contract(self):
        self.assertEqual(predict_score([1, 2, 3]), 2.0)
        for invalid in ([], [float("nan")], [True], "1,2"):
            with self.assertRaises(ValueError):
                predict_score(invalid)

    @unittest.skipUnless(importlib.util.find_spec("bentoml"), "academy dependency not installed")
    def test_real_service_schema_and_execution(self):
        service = build_native_service()
        self.assertEqual(service.name, "handbook_predict")
        self.assertEqual(service.config["traffic"], {"timeout": 5, "max_concurrency": 8})
        self.assertEqual(service.apis["predict"].route, "/predict")
        self.assertEqual(service.inner().predict([1, 2, 3]), {"score": 2.0})


if __name__ == "__main__": unittest.main()
