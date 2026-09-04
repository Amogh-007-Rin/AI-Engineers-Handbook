import unittest
from model import graph_size, mean_range


class DaskProjectTests(unittest.TestCase):
    def test_lazy_mean_and_graph(self):
        result = mean_range(10, 3)
        self.assertTrue(hasattr(result, "compute"))
        self.assertEqual(result.compute(), 4.5)
        self.assertGreater(graph_size(10, 3), 0)

    def test_invalid_partition_contract(self):
        with self.assertRaises(ValueError): mean_range(0)
        with self.assertRaises(ValueError): mean_range(4, 0)


if __name__ == "__main__": unittest.main()
