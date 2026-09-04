import unittest
from split import audit_split


class GraphSplitTests(unittest.TestCase):
    def test_crossing_and_unknown_edges(self):
        result = audit_split([(1, 2), (2, 3), (3, 4)], {1, 2}, {3})
        self.assertEqual(result["crossing_edges"], [(2, 3)])
        self.assertEqual(result["unknown_nodes"], [4])

    def test_overlap_fails(self):
        with self.assertRaises(ValueError): audit_split([], {1, 2}, {2, 3})


if __name__ == "__main__": unittest.main()
