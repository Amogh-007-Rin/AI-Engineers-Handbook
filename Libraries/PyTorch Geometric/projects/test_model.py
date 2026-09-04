import unittest
from model import batch_graphs, make_graph


class PyGProjectTests(unittest.TestCase):
    def test_batch_offsets_and_assignment(self):
        batch = batch_graphs([make_graph([1, 2], [(0, 1)]), make_graph([3, 4, 5], [(0, 2)])])
        self.assertEqual(batch.num_nodes, 5)
        self.assertEqual(batch.edge_index.tolist(), [[0, 2], [1, 4]])
        self.assertEqual(batch.batch.tolist(), [0, 0, 1, 1, 1])

    def test_invalid_edges(self):
        with self.assertRaises(ValueError): make_graph([1, 2], [(0, 2)])
        with self.assertRaises(ValueError): batch_graphs([])


if __name__ == "__main__": unittest.main()
