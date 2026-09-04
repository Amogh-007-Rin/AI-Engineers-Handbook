import unittest
from orbit_model import fit_predict


class OrbitSmokeTests(unittest.TestCase):
    def test_future_forecast(self):
        result = fit_predict()
        self.assertEqual(len(result), 3)
        self.assertIn("prediction", result.columns)


if __name__ == "__main__": unittest.main()
