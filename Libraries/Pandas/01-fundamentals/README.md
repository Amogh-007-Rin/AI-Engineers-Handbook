---
title: Pandas Series DataFrames indexes and selection
slug: pandas-foundations
level: foundation
stage: data
estimated_hours: 5
prerequisites:
  - numpy-foundations
  - data-quality-foundations
learning_objectives:
  - Construct labeled tables with intentional indexes and dtypes
  - Select rows and columns without ambiguous chained assignment
  - Explain label alignment and test its consequences
formats:
  - lesson
  - exercise
compute: cpu
status: published
last_verified: 2026-09-03
library: Pandas
supported_versions: 2.x
---

# Series, DataFrames, indexes, and selection

A `Series` maps index labels to values. A `DataFrame` aligns multiple Series by index. Operations align labels before calculating, which is powerful and can create missing values when labels differ.

```python
import pandas as pd

df = pd.DataFrame({"city": ["Leeds", "London"], "value": [3.0, 5.0]})
df = df.set_index("city")
assert df.loc["Leeds", "value"] == 3.0   # labels
assert df.iloc[0, 0] == 3.0               # positions
df.loc[df["value"] < 4, "band"] = "low"
```

Use `.loc` for label selection and assignment, `.iloc` for positions, and explicit Boolean masks. Avoid chained assignment because it obscures whether a temporary object or original data is modified.

## Exercise

Create two Series with partially overlapping entity indexes. Predict their sum, then run it and explain every missing value. Build a DataFrame with explicit string, nullable integer, Boolean, category, and datetime dtypes. Write assertions for schema and unique entity keys.

## Completion criteria

- [ ] You can predict alignment before running it.
- [ ] Selection uses intentional label or position semantics.
- [ ] Schema and key assumptions are executable tests.
- [ ] No chained assignment warnings are ignored.
