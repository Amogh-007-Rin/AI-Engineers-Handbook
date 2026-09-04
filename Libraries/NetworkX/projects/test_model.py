import unittest
import networkx as nx

from model import build, features, round_trip


class NetworkXProjectTests(unittest.TestCase):
    def test_features_and_components(self):
        graph = build([("a", "b", 2), ("b", "c", 1)])
        self.assertEqual(features(graph)["b"]["degree"], 2)
        self.assertEqual(list(nx.connected_components(graph)), [{"a", "b", "c"}])

    def test_schema_round_trip(self):
        graph = build([("a", "b", 2)])
        restored = round_trip(graph)
        self.assertEqual(set(restored.edges), set(graph.edges))
        self.assertEqual(restored["a"]["b"]["weight"], 2.0)

    def test_invalid_edges_fail(self):
        with self.assertRaises(ValueError): build([("a", "a", 1)])
        with self.assertRaises(ValueError): build([("a", "b", 0)])


if __name__ == "__main__": unittest.main()
