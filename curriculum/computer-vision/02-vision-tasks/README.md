---
title: Vision task design and failure analysis
slug: vision-task-design
level: practitioner
stage: computer-vision
estimated_hours: 12
prerequisites:
  - vision-image-foundations
learning_objectives:
  - Distinguish classification detection segmentation and tracking targets
  - Design leakage-safe dataset splits and task metrics
  - Analyze vision errors across environmental and demographic slices
formats:
  - lesson
  - project
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Vision task design and failure analysis

Classification assigns labels to an image or region; detection localizes instances; semantic segmentation labels pixels by class; instance segmentation separates objects; tracking preserves identity over time. Choose the least complex target that supports the decision.

Split by source entity, scene, camera, time, patient, or video—not random frames—when neighboring images share content. Audit duplicates and near-duplicates. Evaluate per class and relevant slices such as illumination, scale, occlusion, device, geography, and population. Inspect false positives, false negatives, localization errors, and annotation ambiguity separately.

Pretrained models inherit dataset assumptions and licenses. Record preprocessing, class mapping, input size, threshold, non-maximum-suppression policy, precision, latency, memory, and export behavior. Robustness and human review may matter more than a small aggregate-score gain.

## Completion criteria

- [ ] Task granularity follows the real decision.
- [ ] Related images cannot cross splits.
- [ ] Error taxonomy separates recognition, localization, and data failures.
- [ ] Deployment metrics include quality, latency, resources, and review policy.
