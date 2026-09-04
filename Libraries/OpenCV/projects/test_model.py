import unittest
import numpy as np
from model import normalize, resize


class OpenCVProjectTests(unittest.TestCase):
    def test_color_range_and_resize(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8); image[0, 0] = [0, 0, 255]
        output = normalize(image)
        self.assertEqual(output.shape, (2, 3, 3)); self.assertTrue(np.allclose(output[0, 0], [1, 0, 0]))
        self.assertEqual(resize(image, 4, 5).shape, (5, 4, 3))

    def test_input_contract(self):
        with self.assertRaises(ValueError): normalize(np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError): resize(np.zeros((2, 2, 3), dtype=np.uint8), 0, 1)


if __name__ == "__main__": unittest.main()
