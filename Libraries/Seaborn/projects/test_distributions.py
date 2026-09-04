import tempfile
import unittest
from pathlib import Path
import pandas as pd

from distributions import build_report, example_data, save_report


class DistributionTests(unittest.TestCase):
    def test_data_contract_and_labels(self) -> None:
        data = example_data()
        self.assertEqual(list(data["group"].cat.categories), ["control", "treatment"])
        fig, ax = build_report(data)
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["control", "treatment"])
        self.assertEqual(ax.get_ylabel(), "Score (points)")
        fig.clear()

    def test_invalid_data_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_report(pd.DataFrame())

    def test_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distribution.png"
            save_report(path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
