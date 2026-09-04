---
title: Bokeh figures glyphs accessibility and deterministic HTML exports
slug: bokeh-foundations
level: foundation
stage: visualization
estimated_hours: 10
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Map tidy data to glyphs, scales, and interaction state
  - Validate labels, units, ranges, and missing values
  - Produce deterministic standalone HTML artifacts
  - Test accessibility and browser-independent data contracts
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Bokeh
supported_versions: 3.x
---

# Bokeh foundations

Bokeh separates data sources, glyphs, ranges, tools, and layouts. Build the
analytical mapping first: identify the measured variable, uncertainty, units,
aggregation denominator, and intended comparison. Tooltips and hover callbacks
must expose provenance rather than hide transformations.

Validate finite values, categorical ordering, axis ranges, and missing-data
policy before rendering. Prefer `ColumnDataSource` for explicit column schemas,
and keep callbacks bounded and auditable. A standalone HTML export is an
artifact: capture Bokeh/Python versions, input hash, theme, and whether external
resources are embedded or loaded from a CDN.

## Completion criteria

- [ ] Data columns, units, and aggregation are specified.
- [ ] Missing, infinite, and empty inputs have explicit behavior.
- [ ] Rendered output is deterministic and accessible.
- [ ] Export metadata permits clean reproduction.
