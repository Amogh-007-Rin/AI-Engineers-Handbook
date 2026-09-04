import unittest
from backtest import last_value_forecast, mae, rolling_origins


class BacktestTests(unittest.TestCase):
    def test_temporal_windows(self):
        origins = rolling_origins([1, 2, 3, 4, 5], 2, 2)
        self.assertEqual(origins, [([1, 2], [3, 4]), ([1, 2, 3], [4, 5])])
        self.assertTrue(all(max(range(len(history)), default=-1) < len(history) for history, _ in origins))

    def test_baseline_and_metric(self):
        self.assertEqual(last_value_forecast([1, 2], 2), [2, 2])
        self.assertEqual(mae([3, 4], [2, 2]), 1.5)

    def test_invalid_windows(self):
        with self.assertRaises(ValueError): rolling_origins([1, 2], 2)


if __name__ == "__main__": unittest.main()
