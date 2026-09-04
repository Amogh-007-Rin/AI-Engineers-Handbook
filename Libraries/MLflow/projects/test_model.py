import importlib.util
import tempfile
import unittest
from model import can_promote, record_local_run, validate_run


GOOD = {"run_id": "r-1", "git_sha": "abcdef1", "data_hash": "12345678",
        "environment": "lock.txt", "metrics": {"validation_score": .9},
        "artifacts": {"model_signature": "sig.json"}}


class MLflowProjectTests(unittest.TestCase):
    def test_manifest_and_promotion(self):
        self.assertTrue(validate_run(GOOD)); self.assertTrue(can_promote(GOOD, minimum_score=.8))

    def test_gates(self):
        self.assertFalse(can_promote({**GOOD, "metrics": {"validation_score": .7}}, minimum_score=.8))
        with self.assertRaises(ValueError): validate_run({"run_id": "x"})
        with self.assertRaises(ValueError): validate_run({**GOOD, "metrics": {"validation_score": float("nan")}})

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "academy dependency not installed")
    def test_real_local_tracking_and_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = record_local_run(directory, GOOD)
            self.assertTrue(can_promote(manifest, minimum_score=.8))
            self.assertEqual(manifest["git_sha"], GOOD["git_sha"])
            self.assertTrue(manifest["artifacts"]["model_signature"].endswith("signature.json"))


if __name__ == "__main__": unittest.main()
