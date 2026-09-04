import unittest
from model import validate_record


GOOD = {"file_name": "image.jpg", "image_id": 1, "height": 100, "width": 200,
        "annotations": [{"bbox": [10, 20, 50, 70], "category_id": 0}]}


class Detectron2ProjectTests(unittest.TestCase):
    def test_valid_dataset_dictionary(self): self.assertTrue(validate_record(GOOD, 2))

    def test_box_and_category_gates(self):
        for annotation in ({"bbox": [50, 20, 10, 70], "category_id": 0},
                           {"bbox": [10, 20, 250, 70], "category_id": 0},
                           {"bbox": [10, 20, 50, 70], "category_id": 2}):
            with self.assertRaises(ValueError): validate_record({**GOOD, "annotations": [annotation]}, 2)


if __name__ == "__main__": unittest.main()
