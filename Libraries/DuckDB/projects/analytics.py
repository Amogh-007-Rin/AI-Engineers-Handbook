"""Audited DuckDB analytical fixture and customer aggregation."""

from __future__ import annotations
import duckdb


def build_connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY, country VARCHAR NOT NULL)")
    connection.execute("CREATE TABLE transactions(id INTEGER PRIMARY KEY, customer_id INTEGER, amount DOUBLE CHECK(amount >= 0))")
    connection.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "GB"), (2, "US")])
    connection.executemany("INSERT INTO transactions VALUES (?, ?, ?)", [(10, 1, 2.5), (11, 1, 7.5), (12, 99, 4.0)])
    return connection


def customer_summary(connection: duckdb.DuckDBPyConnection) -> list[tuple]:
    return connection.execute("""
        SELECT c.id, c.country, COUNT(t.id) AS transaction_count,
               COALESCE(SUM(t.amount), 0.0) AS total_amount
        FROM customers c LEFT JOIN transactions t ON c.id = t.customer_id
        GROUP BY c.id, c.country ORDER BY c.id
    """).fetchall()


def orphan_count(connection: duckdb.DuckDBPyConnection) -> int:
    return connection.execute("""
        SELECT COUNT(*) FROM transactions t
        WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.id = t.customer_id)
    """).fetchone()[0]
