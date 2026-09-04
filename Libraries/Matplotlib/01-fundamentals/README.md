---
title: Matplotlib figures axes and truthful visual encoding
slug: matplotlib-foundations
level: foundation
stage: data
estimated_hours: 8
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Build plots with explicit figure and axes ownership
  - Match marks scales and summaries to analytical questions
  - Produce deterministic labeled and accessible visual artifacts
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Matplotlib
supported_versions: 3.x
---

# Matplotlib figures, axes, and truthful visual encoding

A `Figure` owns the full canvas; each `Axes` owns a coordinate system and plotted artists. Prefer the object-oriented interface (`fig, ax = plt.subplots()`) so functions receive an axes explicitly and remain composable and testable.

Choose encodings from the question: position along a common scale is effective for comparison; lines imply ordered continuity; bars compare magnitudes from a meaningful baseline; scatter plots expose relationships and density; distributions need more than an average. Never use a second axis, truncated scale, area, or color in a way that exaggerates differences.

Labels must state variable and unit. Legends identify encodings, not decorate. Color cannot be the only carrier of meaning; combine it with markers, line styles, direct labels, or facets. Provide alt text stating purpose, axes, pattern, uncertainty, and important exceptions—not every pixel.

## Exercise

Create a reusable function that accepts an axes and a validated time series, draws observations and an uncertainty band, labels units, marks missing intervals rather than connecting across them, and returns artists for testing. Export a deterministic PNG with metadata stripped or controlled.

## Completion criteria

- [ ] Figure and axes ownership are explicit.
- [ ] Scale, aggregation, and uncertainty choices are justified.
- [ ] Visual meaning survives grayscale and common color-vision deficiencies.
- [ ] Tests inspect artists and labels, not fragile pixel equality alone.
