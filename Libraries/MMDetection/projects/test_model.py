import unittest
from model import validate_config


GOOD = {"model": {"bbox_head": {"num_classes": 2}}, "train_dataloader": {"batch_size": 2},
        "val_dataloader": {"batch_size": 1}, "val_evaluator": {"type": "CocoMetric"},
        "train_cfg": {"max_epochs": 2}, "default_hooks": {"checkpoint": {}},
        "metainfo": {"classes": ["cat", "dog"]}, "load_from": "model@sha256:abc"}


class MMDetectionProjectTests(unittest.TestCase):
    def test_valid_resolved_config(self): self.assertTrue(validate_config(GOOD))

    def test_class_and_artifact_gates(self):
        with self.assertRaises(ValueError): validate_config({**GOOD, "model": {"bbox_head": {"num_classes": 3}}})
        with self.assertRaises(ValueError): validate_config({**GOOD, "load_from": "latest.pth"})


if __name__ == "__main__": unittest.main()
