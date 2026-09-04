import unittest
from model import validate_pipeline


class PySparkProjectTests(unittest.TestCase):
    def test_valid_pipeline(self):
        self.assertTrue(validate_pipeline({"input_schema": {"id": "long"}, "output_schema": {"x": "double"}, "join_keys": ["id"], "null_policy": "reject", "writes": True, "idempotent": True}))

    def test_distributed_safety_gates(self):
        for bad in ({"input_schema": {"x": "int"}, "output_schema": {"y": "int"}, "join_keys": ["x"]},
                    {"input_schema": {"x": "int"}, "output_schema": {"y": "int"}, "writes": True},
                    {"input_schema": {"x": "int"}, "output_schema": {"y": "int"}, "driver_collect": True}):
            with self.assertRaises(ValueError): validate_pipeline(bad)


if __name__ == "__main__": unittest.main()
