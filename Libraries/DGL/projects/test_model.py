import unittest
import dgl
import torch
from model import graph_and_features, incoming_sum


class DGLProjectTests(unittest.TestCase):
    def test_hand_computed_message_passing(self):
        graph = graph_and_features()
        self.assertTrue(torch.equal(incoming_sum(graph), torch.tensor([[4.0], [1.0], [2.0]])))

    def test_alignment_gate(self):
        graph = dgl.graph(([0], [1]), num_nodes=2)
        with self.assertRaises(ValueError): incoming_sum(graph)


if __name__ == "__main__": unittest.main()
