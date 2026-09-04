---
title: Image representation geometry and evaluation
slug: vision-image-foundations
level: foundation
stage: computer-vision
estimated_hours: 10
prerequisites:
  - neural-architecture-families
learning_objectives:
  - Explain image tensors coordinate systems and geometric transforms
  - Prevent label corruption while augmenting images and annotations
  - Compute and test overlap metrics for detection and segmentation
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Image representation, geometry, and evaluation

An image array has axes whose meaning depends on a library: height–width–channels or channels–height–width. Dtype and range matter as much as shape. `uint8` values in `[0,255]`, floating values in `[0,1]`, and standardized tensors are not interchangeable.

Coordinates may describe pixels, normalized positions, box corners, centers, or widths. Define whether upper bounds are inclusive and whether boxes use `(x,y)` or `(row,column)`. Resize, crop, rotate, and flip annotations with the image; a visually plausible image with stale labels silently poisons training.

Intersection over Union is intersection area divided by union area. IoU is undefined for two empty regions unless the task defines a convention. Detection evaluation also depends on confidence ranking, matching policy, IoU threshold, class, and duplicate predictions; one overlap number is not mean average precision.

## Exercise

Implement box-area and IoU functions with tests for identical, disjoint, partially overlapping, zero-area, reversed, and edge-touching boxes. Document coordinate convention. Then design an augmentation test that applies a horizontal flip twice and proves both image and boxes return to their original state.

## Completion criteria

- [ ] Shape, range, dtype, channels, and coordinates are explicit.
- [ ] Invalid boxes fail rather than producing misleading scores.
- [ ] Geometric transforms update every spatial annotation.
- [ ] Evaluation limitations are stated alongside the metric.
