---
title: Plotly figures encodings interactivity and portable artifacts
slug: plotly-foundations
level: foundation
stage: visualization
estimated_hours: 10
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Build figures with explicit traces, encodings, and units
  - Validate aggregation and hover metadata before interaction
  - Export JSON/HTML artifacts reproducibly
  - Select static or interactive output from a user and deployment contract
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Plotly
supported_versions: 6.x
---

# Plotly foundations

Plotly figures are structured specifications: data traces, layout, axes, and
interaction configuration. Inspect the figure JSON instead of trusting the
rendered browser view. Check that aggregations preserve denominators, labels
state units, and hover fields expose the row or query provenance.

Choose HTML when interaction is part of the task contract and a static export
when archival or accessibility constraints require it. Pin template, locale,
Plotly version, and data ordering; avoid callbacks that fetch untrusted URLs.
Test empty, null, duplicate, and extreme values before embedding a figure in a
dashboard.

## Completion criteria

- [ ] Trace schema and units are explicit.
- [ ] Aggregation and missing-data choices are tested.
- [ ] JSON/HTML output is reproducible.
- [ ] Deployment and accessibility trade-offs are documented.
