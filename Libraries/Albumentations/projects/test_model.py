import os, unittest

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import albumentations as A
from model import augment


class AlbumentationsProjectTests(unittest.TestCase):
    def test_box_geometry_and_replay(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        result = augment(image, [(2, 1, 8, 6)], ["object"])
        self.assertTrue(np.allclose(result["bboxes"][0], (12, 1, 18, 6)))
        replayed = A.ReplayCompose.replay(result["replay"], image=image, bboxes=[(2, 1, 8, 6)], labels=["object"])
        self.assertTrue(np.array_equal(replayed["image"], result["image"]))

    def test_target_alignment(self):
        with self.assertRaises(ValueError): augment(np.zeros((2, 2, 3), np.uint8), [(0, 0, 1, 1)], [])


if __name__ == "__main__": unittest.main()
