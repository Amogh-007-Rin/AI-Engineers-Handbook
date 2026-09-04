---
title: Pandas cleaning joining grouping and validation
slug: pandas-workflows
level: practitioner
stage: data
estimated_hours: 7
prerequisites:
  - pandas-foundations
learning_objectives:
  - Build explicit cleaning and missing-data policies
  - Validate join cardinality and aggregation grain
  - Reshape and aggregate tables without losing entity meaning
formats:
  - lesson
  - exercise
compute: cpu
status: published
last_verified: 2026-09-03
library: Pandas
supported_versions: 2.x
---

# Cleaning, joining, grouping, and validation

Before every transformation, state the grain: what does one row represent? After it, verify row count, keys, schema, and missingness. Joins are a major source of silent data multiplication.

```python
result = orders.merge(
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)
assert not (result["_merge"] == "left_only").any()
```

Use `validate` to encode expected cardinality and `indicator` to audit unmatched keys. For grouping, use named aggregations and decide deliberately whether keys become an index.

Missing data is information about collection, not merely an obstacle. Distinguish unknown, not applicable, not yet observed, and corrupted values before dropping or imputing them.

## Exercise

Given customers and transactions tables, produce one row per customer with transaction count, total, median, first/last timestamps, and days since last transaction. Validate key uniqueness and join cardinality. Retain customers with no transactions and define every resulting missing value. Test duplicate customer keys, orphan transactions, empty input, and timezone inconsistency.

## Completion criteria

- [ ] Input and output grain are documented and tested.
- [ ] Every join declares expected cardinality.
- [ ] Missingness policy distinguishes meanings.
- [ ] Corrupt fixtures cause specific test failures.
