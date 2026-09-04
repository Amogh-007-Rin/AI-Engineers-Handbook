import unittest
from model import validate_config


class TensorFlowServingProjectTests(unittest.TestCase):
    def test_pinned_model_config(self):
        self.assertTrue(validate_config({"models": [{"name": "predict", "base_path": "/models/predict", "version_policy": "specific:7"}]}))

    def test_config_gates(self):
        with self.assertRaises(ValueError): validate_config({})
        with self.assertRaises(ValueError): validate_config({"models": [{"name": "x", "base_path": "/x", "version_policy": "all"}]})


if __name__ == "__main__": unittest.main()
