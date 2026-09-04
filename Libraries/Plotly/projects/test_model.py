import tempfile
import unittest
from model import export, figure


class PlotlyProjectTests(unittest.TestCase):
    def test_trace_contract(self):
        fig = figure(["a", "b"], [1, 2])
        self.assertEqual(list(fig.data[0].x), ["a", "b"])
        self.assertIn("hovertemplate", fig.data[0])

    def test_export_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(export(["a"], [1], f"{directory}/plot.html").exists())
        with self.assertRaises(ValueError): figure([], [])


if __name__ == "__main__": unittest.main()
