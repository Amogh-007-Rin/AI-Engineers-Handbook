import unittest
from analytics import build_connection, customer_summary, orphan_count


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = build_connection()

    def tearDown(self) -> None:
        self.connection.close()

    def test_summary_preserves_zero_activity_customer(self) -> None:
        self.assertEqual(customer_summary(self.connection), [(1, "GB", 2, 10.0), (2, "US", 0, 0.0)])

    def test_orphans_are_visible(self) -> None:
        self.assertEqual(orphan_count(self.connection), 1)

    def test_constraints_reject_invalid_data(self) -> None:
        with self.assertRaises(Exception):
            self.connection.execute("INSERT INTO transactions VALUES (13, 1, -1)")


if __name__ == "__main__":
    unittest.main()
