---
title: Ultralytics YOLO datasets training detection evaluation and export
slug: ultralytics-yolo-foundations
level: practitioner
stage: computer-vision
estimated_hours: 16
prerequisites:
  - vision-task-design
learning_objectives:
  - Define YOLO dataset labels classes normalized coordinates and split integrity
  - Train and validate detection models against task baselines
  - Interpret confidence IoU NMS precision recall and mAP
  - Export and benchmark models with licensing and deployment controls
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: Ultralytics YOLO
supported_versions: 8.x
---

# Ultralytics YOLO foundations

Detection data couples each image to class IDs and normalized center-width-height
boxes. Validate class range, finite coordinates, positive area, boxes inside the
image, missing labels, duplicates, and train/validation identity before training.
Split by scene, video, subject, or time to prevent near-frame leakage.

Confidence thresholds and non-maximum suppression trade precision, recall, and
duplicate detections. Report AP per class and size, PR curves, localization and
classification errors, empty-image behavior, latency, memory, and throughput.
Compare with a simple or pretrained baseline and inspect visual failures.

Pin weights/config/data/code revisions and review dataset/model licenses. Export
to the actual runtime, compare source and target predictions within tolerances,
test dynamic shapes and malformed images, and establish monitoring/rollback.

## Completion criteria

- [ ] Dataset coordinates, classes, and split identity are validated.
- [ ] Evaluation includes per-class/size errors and threshold analysis.
- [ ] Source/export parity and performance are measured.
- [ ] Licenses, provenance, monitoring, and rollback are documented.
