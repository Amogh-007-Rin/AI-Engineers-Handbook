"""Audited Polars lazy customer aggregation."""

from __future__ import annotations
import polars as pl


def customer_features(customers: pl.DataFrame, transactions: pl.DataFrame) -> pl.DataFrame:
    required_customers = {"customer_id", "country"}
    required_transactions = {"transaction_id", "customer_id", "amount"}
    if not required_customers.issubset(customers.columns) or not required_transactions.issubset(transactions.columns):
        raise ValueError("required columns missing")
    if customers["customer_id"].n_unique() != customers.height:
        raise ValueError("customer_id must be unique")
    if transactions["transaction_id"].n_unique() != transactions.height:
        raise ValueError("transaction_id must be unique")
    if transactions.filter(pl.col("amount") < 0).height:
        raise ValueError("amount cannot be negative")
    aggregates = transactions.lazy().group_by("customer_id").agg(
        pl.len().alias("transaction_count"), pl.col("amount").sum().alias("total_amount")
    )
    return (
        customers.lazy()
        .join(aggregates, on="customer_id", how="left", validate="1:1")
        .with_columns(
            pl.col("transaction_count").fill_null(0),
            pl.col("total_amount").fill_null(0.0),
        )
        .sort("customer_id")
        .collect()
    )
