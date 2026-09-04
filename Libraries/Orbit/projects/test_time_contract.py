import unittest
from datetime import datetime, timedelta
from time_contract import rolling_origins, validate_daily_dates


class TimeContractTests(unittest.TestCase):
    def test_dates_and_origins(self):
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(6)]
        validate_daily_dates(dates)
        origins = rolling_origins(6, 3, 2)
        self.assertEqual([(list(a), list(b)) for a, b in origins], [([0, 1, 2], [3, 4]), ([0, 1, 2, 3], [4, 5])])

    def test_gaps_and_overlap_fail(self):
        with self.assertRaises(ValueError):
            validate_daily_dates([datetime(2025, 1, 1), datetime(2025, 1, 3), datetime(2025, 1, 4)])
        with self.assertRaises(ValueError):
            rolling_origins(3, 3, 1)


if __name__ == "__main__": unittest.main()
