import unittest
import polars as pl
from lazy_pipeline import customer_features


class PipelineTests(unittest.TestCase):
    def test_features_preserve_customers(self) -> None:
        customers = pl.DataFrame({"customer_id": [2, 1], "country": ["US", "GB"]})
        transactions = pl.DataFrame({"transaction_id": [10, 11], "customer_id": [1, 1], "amount": [2.0, 3.0]})
        result = customer_features(customers, transactions)
        self.assertEqual(result["customer_id"].to_list(), [1, 2])
        self.assertEqual(result["transaction_count"].to_list(), [2, 0])
        self.assertEqual(result["total_amount"].to_list(), [5.0, 0.0])

    def test_duplicate_and_negative_fail(self) -> None:
        customers = pl.DataFrame({"customer_id": [1, 1], "country": ["GB", "GB"]})
        transactions = pl.DataFrame({"transaction_id": [1], "customer_id": [1], "amount": [-1.0]})
        with self.assertRaises(ValueError):
            customer_features(customers, transactions)


if __name__ == "__main__":
    unittest.main()
