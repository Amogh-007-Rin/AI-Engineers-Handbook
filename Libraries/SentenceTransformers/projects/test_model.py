import unittest
from model import similarities


class SentenceTransformerProjectTests(unittest.TestCase):
    def test_ranking_and_shape(self):
        scores = similarities([[1, 0]], [[1, 0], [0, 1]])
        self.assertEqual(tuple(scores.shape), (1, 2)); self.assertGreater(scores[0, 0], scores[0, 1])

    def test_dimension_and_finite_gates(self):
        with self.assertRaises(ValueError): similarities([[1, 0]], [[1, 0, 0]])
        with self.assertRaises(ValueError): similarities([[float("nan"), 0]], [[1, 0]])


if __name__ == "__main__": unittest.main()
