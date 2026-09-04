---
title: Seaborn statistical visualization foundations
slug: seaborn-foundations
level: foundation
stage: data
estimated_hours: 8
prerequisites:
  - matplotlib-foundations
  - pandas-foundations
learning_objectives:
  - Map tidy variables to semantic visual encodings
  - Distinguish observations distributions estimates and uncertainty
  - Control aggregation ordering facets and accessibility explicitly
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Seaborn
supported_versions: 0.x
---

# Seaborn statistical visualization foundations

Seaborn maps columns in tidy data to position, hue, style, size, row, and column. Axes-level functions compose inside existing Matplotlib figures; figure-level functions own a grid and are useful for faceting. Know which interface owns layout before combining plots.

Some functions aggregate and calculate uncertainty automatically. Never publish them without knowing the estimator, sampling unit, interval method, and repeated-measure structure. A bar of means can conceal sample size, skew, multimodality, and outliers; show observations or distributions when they answer the question.

Semantic mappings need stable order and accessible redundancy. Explicitly set category order and palettes; use style or facets when color alone is insufficient. Facets must share or intentionally vary scales, and empty combinations are evidence about the data.

## Exercise

Create a tidy repeated-measures dataset. Show raw observations, within-subject change, group distributions, and an estimate with an interval. Compare naive row-level uncertainty with subject-level resampling and explain the unit-of-analysis difference.

## Completion criteria

- [ ] Every mark has a defined data and statistical unit.
- [ ] Aggregation and uncertainty behavior are explicit.
- [ ] Category order and missing combinations are tested.
- [ ] The plot remains interpretable without color.
