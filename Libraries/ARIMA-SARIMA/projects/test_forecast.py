import unittest
from forecast import mae, rolling_forecasts


class ForecastTests(unittest.TestCase):
    def test_rolling_origin_has_no_future_access(self) -> None:
        values = [2 * x + (-1) ** x * .1 for x in range(18)]
        actual, naive, model = rolling_forecasts(values, 10)
        self.assertEqual(len(actual), 8)
        self.assertLess(mae(actual, model), mae(actual, naive))

    def test_metric_and_input_contracts(self) -> None:
        self.assertEqual(mae([1, 3], [2, 1]), 1.5)
        with self.assertRaises(ValueError):
            rolling_forecasts([1] * 5, 4)
        with self.assertRaises(ValueError):
            mae([], [])


if __name__ == "__main__": unittest.main()
