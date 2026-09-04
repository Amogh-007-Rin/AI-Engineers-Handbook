import unittest
import pandas as pd

from data_quality_pipeline import customer_features


class PipelineTests(unittest.TestCase):
    def customers(self) -> pd.DataFrame:
        return pd.DataFrame({
            "customer_id": [2, 1], "country": [" gb ", "us"],
            "signup_at": ["2025-01-02", "2025-01-01"],
        })

    def transactions(self) -> pd.DataFrame:
        return pd.DataFrame({
            "transaction_id": [10, 11, 12], "customer_id": [1, 1, 99],
            "amount": [2.5, 7.5, 100], "occurred_at": ["2025-02-01", "2025-02-02", "2025-02-03"],
        })

    def test_pipeline_preserves_customers_and_audits_orphans(self) -> None:
        result, audit = customer_features(self.customers(), self.transactions())
        self.assertEqual(result["customer_id"].tolist(), [1, 2])
        self.assertEqual(result["transaction_count"].tolist(), [2, 0])
        self.assertEqual(result["total_amount"].tolist(), [10.0, 0.0])
        self.assertEqual(audit["counts"]["orphan_transactions"], 1)
        self.assertEqual(audit["counts"]["customers_without_transactions"], 1)

    def test_duplicate_customer_fails(self) -> None:
        customers = pd.concat([self.customers(), self.customers().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            customer_features(customers, self.transactions())

    def test_negative_amount_fails(self) -> None:
        transactions = self.transactions()
        transactions.loc[0, "amount"] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            customer_features(self.customers(), transactions)

    def test_missing_column_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            customer_features(self.customers().drop(columns="country"), self.transactions())


if __name__ == "__main__":
    unittest.main()
