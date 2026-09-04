---
title: Detectron2 datasets structures training evaluation and deployment
slug: detectron2-foundations
level: advanced
stage: computer-vision
estimated_hours: 16
prerequisites:
  - vision-task-design
learning_objectives:
  - Register datasets with valid boxes masks classes and metadata
  - Configure models trainers hooks and augmentations explicitly
  - Evaluate detection and segmentation by class size and failure type
  - Export and deploy with dependency licensing and performance controls
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: Detectron2
supported_versions: 0.6.x
---

# Detectron2 foundations

Detectron2 represents each sample as a dictionary containing image identity,
dimensions, and annotations. Define whether boxes are XYXY or XYWH and whether
coordinates are absolute or relative. Class IDs must be contiguous for training
even when source dataset IDs are not; preserve the mapping in metadata. Validate
boxes, polygon/RLE masks, keypoints, crowd flags, and image dimensions before
registration.

Configuration selects architecture, backbone, weights, transforms, solver,
batching, thresholds, and device. Treat pretrained weights and model-zoo configs
as versioned licensed dependencies. Custom trainers and hooks can change data,
optimization, evaluation, and checkpoint state, so test them independently.

COCO AP is a family of metrics, not one accuracy value. Report AP by IoU, class,
and object size; analyze localization, classification, duplicate, background,
and missed-detection errors. Split by scene/video/subject to avoid near-frame
leakage. Export to the real runtime and compare predictions, latency, memory,
and unsupported operations before rollout.

## Completion criteria

- [ ] Dataset dictionaries and category mappings are validated.
- [ ] Config, weights, transforms, and hooks are versioned.
- [ ] Evaluation includes class/size and detection-error analysis.
- [ ] Export parity, licensing, monitoring, and rollback are documented.
