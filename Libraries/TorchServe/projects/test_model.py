import unittest
from model import validate_archive


GOOD = {"name": "classifier", "version": "7", "serialized_file": "weights.pt",
        "handler": "handler.py", "runtime": "python3", "signature": "schema.json"}


class TorchServeProjectTests(unittest.TestCase):
    def test_valid_archive(self): self.assertTrue(validate_archive(GOOD))

    def test_archive_gates(self):
        with self.assertRaises(ValueError): validate_archive({**GOOD, "version": "latest"})
        with self.assertRaises(ValueError): validate_archive({**GOOD, "management_public": True})
        with self.assertRaises(ValueError): validate_archive({**GOOD, "handler": ""})


if __name__ == "__main__": unittest.main()
