import unittest
from model import validate_labels


class YOLOProjectTests(unittest.TestCase):
    def test_valid_labels_and_boundaries(self):
        self.assertTrue(validate_labels([(0, .5, .5, 1.0, 1.0), (1, .25, .25, .2, .2)], 2))

    def test_invalid_class_and_geometry(self):
        for row in ((2, .5, .5, .2, .2), (0, .1, .5, .4, .2), (0, .5, .5, 0, .2)):
            with self.assertRaises(ValueError): validate_labels([row], 2)


if __name__ == "__main__": unittest.main()
