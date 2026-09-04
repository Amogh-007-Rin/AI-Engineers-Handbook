import tempfile
import unittest
from pathlib import Path

from report import build_report, draw_series, save_report


class ReportTests(unittest.TestCase):
    def test_semantic_artists_and_labels(self) -> None:
        fig, ax = build_report()
        self.assertEqual(ax.get_xlabel(), "Time (day)")
        self.assertEqual(ax.get_ylabel(), "Rate (%)")
        self.assertEqual(len(ax.lines), 1)
        self.assertGreaterEqual(len(ax.collections), 1)
        fig.clear()

    def test_invalid_lengths_fail(self) -> None:
        _, ax = build_report()
        with self.assertRaises(ValueError):
            draw_series(ax, [1], [2, 3], [1], [4])

    def test_png_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.png"
            save_report(path)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            self.assertGreater(path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
