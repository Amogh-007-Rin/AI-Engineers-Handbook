import unittest

from text_metrics import recall_at_k, token_f1


class TextMetricTests(unittest.TestCase):
    def test_token_f1_counts_duplicates(self) -> None:
        precision, recall, f1 = token_f1(["a", "a", "b"], ["a", "b", "b"])
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_empty_prediction(self) -> None:
        self.assertEqual(token_f1(["answer"], []), (0.0, 0.0, 0.0))

    def test_recall_at_k(self) -> None:
        self.assertEqual(recall_at_k({"a", "c"}, ["a", "b", "c"], 2), 0.5)
        with self.assertRaises(ValueError):
            recall_at_k(set(), [], 1)


if __name__ == "__main__":
    unittest.main()
