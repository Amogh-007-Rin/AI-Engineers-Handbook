---
title: MMDetection configs datasets registries training evaluation and deployment
slug: mmdetection-foundations
level: advanced
stage: computer-vision
estimated_hours: 16
prerequisites:
  - vision-task-design
learning_objectives:
  - Compose configuration and registry components without hidden overrides
  - Convert and validate detection datasets and class metadata
  - Train evaluate and diagnose detectors with reproducible hooks and schedules
  - Export and deploy through the OpenMMLab compatibility stack
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: MMDetection
supported_versions: 3.x
---

# MMDetection foundations

MMDetection is a configuration-driven detection framework built on MMEngine and
MMCV. A config inherits and overrides models, datasets, pipelines, schedules,
hooks, evaluators, and runtime settings. Print the fully resolved config and
record every package/weight revision; a small override can silently change class
count, normalization, augmentation, or evaluation.

Dataset adapters must preserve image identity, dimensions, category mapping,
box mode, masks, ignored/crowd annotations, and empty images. Validate data
before registry construction and split by scene/video/subject. Pipelines apply
geometry to both pixels and targets; test them on visual and numeric fixtures.

Report AP by IoU, class, and size along with localization, classification,
duplicate, and background errors. Checkpoint model/optimizer/scheduler/config
state. Export through the supported deployment stack and compare source/runtime
predictions, preprocessing, NMS, dynamic shapes, latency, and memory.

## Completion criteria

- [ ] Resolved config and compatibility matrix are captured.
- [ ] Dataset/category/geometry contracts are validated.
- [ ] Training and evaluation evidence includes error slices.
- [ ] Export parity, licensing, monitoring, and rollback are tested.
