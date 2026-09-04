import unittest
from ranking import ndcg_at_k, precision_recall_at_k


class RankingTests(unittest.TestCase):
    def test_metrics(self):
        precision, recall = precision_recall_at_k(["a", "b", "c"], {"a", "c"}, 2)
        self.assertEqual(precision, .5); self.assertEqual(recall, .5)
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "c"}, 3), 1.5 / (1 + 1 / 1.584962500721156), places=6)

    def test_invalid_evidence(self):
        with self.assertRaises(ValueError): precision_recall_at_k(["a", "a"], {"a"}, 1)
        with self.assertRaises(ValueError): ndcg_at_k(["a"], set(), 1)


if __name__ == "__main__": unittest.main()
