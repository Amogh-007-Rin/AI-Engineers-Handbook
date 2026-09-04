import tempfile
import unittest
from model import chart, export


class BokehProjectTests(unittest.TestCase):
    def test_source_and_labels(self):
        plot = chart([0, 1], [2, 3])
        self.assertEqual(plot.xaxis[0].axis_label, "sample")
        self.assertEqual(plot.renderers[0].data_source.data["y"], [2, 3])

    def test_export_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = export([0, 1], [2, 3], f"{directory}/plot.html")
            self.assertIn("Measured values", output.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError): chart([], [])


if __name__ == "__main__": unittest.main()
