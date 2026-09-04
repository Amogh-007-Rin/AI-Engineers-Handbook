"""Auditable customer feature pipeline built with explicit Pandas contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


CUSTOMER_COLUMNS = {"customer_id", "country", "signup_at"}
TRANSACTION_COLUMNS = {"transaction_id", "customer_id", "amount", "occurred_at"}


@dataclass
class Audit:
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def record(self, name: str, value: int) -> None:
        self.counts[name] = int(value)


def require_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def prepare_customers(raw: pd.DataFrame, audit: Audit) -> pd.DataFrame:
    require_columns(raw, CUSTOMER_COLUMNS, "customers")
    customers = raw.loc[:, sorted(CUSTOMER_COLUMNS)].copy()
    if customers["customer_id"].isna().any():
        raise ValueError("customer_id cannot be missing")
    if customers["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique")
    customers["country"] = customers["country"].astype("string").str.strip().str.upper()
    customers["signup_at"] = pd.to_datetime(customers["signup_at"], utc=True, errors="raise")
    audit.record("customers_input", len(raw))
    audit.record("customers_valid", len(customers))
    return customers


def prepare_transactions(raw: pd.DataFrame, audit: Audit) -> pd.DataFrame:
    require_columns(raw, TRANSACTION_COLUMNS, "transactions")
    transactions = raw.loc[:, sorted(TRANSACTION_COLUMNS)].copy()
    if transactions[["transaction_id", "customer_id"]].isna().any().any():
        raise ValueError("transaction and customer IDs cannot be missing")
    if transactions["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique")
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="raise")
    if (~transactions["amount"].ge(0)).any():
        raise ValueError("amount cannot be negative")
    transactions["occurred_at"] = pd.to_datetime(transactions["occurred_at"], utc=True, errors="raise")
    audit.record("transactions_input", len(raw))
    audit.record("transactions_valid", len(transactions))
    return transactions


def customer_features(customers: pd.DataFrame, transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = Audit()
    customers = prepare_customers(customers, audit)
    transactions = prepare_transactions(transactions, audit)

    orphan_mask = ~transactions["customer_id"].isin(customers["customer_id"])
    audit.record("orphan_transactions", int(orphan_mask.sum()))
    valid_transactions = transactions.loc[~orphan_mask]

    aggregates = valid_transactions.groupby("customer_id", as_index=False).agg(
        transaction_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        median_amount=("amount", "median"),
        first_transaction_at=("occurred_at", "min"),
        last_transaction_at=("occurred_at", "max"),
    )
    result = customers.merge(aggregates, on="customer_id", how="left", validate="one_to_one", indicator=True)
    result["transaction_count"] = result["transaction_count"].fillna(0).astype("int64")
    result["total_amount"] = result["total_amount"].fillna(0.0)
    audit.record("customers_without_transactions", int((result["_merge"] == "left_only").sum()))
    audit.record("output_rows", len(result))
    result = result.drop(columns="_merge").sort_values("customer_id").reset_index(drop=True)
    return result, {"counts": audit.counts, "warnings": audit.warnings}
