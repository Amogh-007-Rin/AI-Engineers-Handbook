import unittest

from grounding import citation_coverage, citation_precision, retrieval_recall


class GroundingMetricTests(unittest.TestCase):
    def test_precision_penalizes_unsupported_citations(self) -> None:
        claims = [{"a", "b"}, {"c"}]
        self.assertAlmostEqual(citation_precision(claims, {(0, "a"), (1, "c")}), 2 / 3)

    def test_coverage_counts_required_claims(self) -> None:
        self.assertEqual(citation_coverage([{"a"}, set(), {"c"}], {0, 1}), 0.5)
        with self.assertRaises(ValueError):
            citation_coverage([], set())

    def test_retrieval_recall(self) -> None:
        self.assertEqual(retrieval_recall({"a", "b"}, ["x", "a"]), 0.5)


if __name__ == "__main__":
    unittest.main()
