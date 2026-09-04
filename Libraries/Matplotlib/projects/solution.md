# Matplotlib project solution guidance

The reference accepts an `Axes`, validates aligned nonempty inputs, returns artists, labels units, uses uncertainty as a secondary band, isolates style with `rc_context`, selects a headless backend, and closes figures after saving. Tests assert semantic structure instead of comparing renderer-dependent pixels. Production reports should additionally lock data/aggregation versions and review axis limits, missingness, color, alt text, and export metadata.
