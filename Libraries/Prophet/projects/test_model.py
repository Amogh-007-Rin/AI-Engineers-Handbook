import unittest
from model import fit_and_forecast, training_frame


class ProphetTests(unittest.TestCase):
    def test_future_dates_and_trend(self) -> None:
        result = fit_and_forecast(3)
        self.assertEqual(len(result), 3)
        self.assertGreater(result["yhat"].iloc[-1], result["yhat"].iloc[0])
        self.assertGreater(result["ds"].min(), training_frame()["ds"].max())

    def test_invalid_horizon(self) -> None:
        with self.assertRaises(ValueError):
            fit_and_forecast(0)


if __name__ == "__main__": unittest.main()
