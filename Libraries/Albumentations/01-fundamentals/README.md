---
title: Albumentations transforms targets replay and vision-data integrity
slug: albumentations-foundations
level: practitioner
stage: computer-vision
estimated_hours: 12
prerequisites:
  - vision-image-foundations
learning_objectives:
  - Apply spatial and pixel transforms consistently to images and annotations
  - Configure bounding-box, mask, and keypoint coordinate contracts
  - Reproduce stochastic augmentation with replay and seeded tests
  - Detect invalid or destructive transforms through visual and numeric audits
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: Albumentations
supported_versions: 2.x
---

# Albumentations foundations

Augmentation changes the training distribution. Each transform must preserve the
task label: a horizontal flip may be valid for animals and wrong for text,
medical laterality, or traffic signs. Separate spatial transforms—which must
update boxes, masks, and keypoints—from pixel transforms such as color and
noise. Declare image dtype/range and annotation coordinate format.

Use `Compose` with explicit target parameters and `ReplayCompose` when an exact
stochastic decision must be inspected or reproduced. Seed tests, but evaluate
the distribution across many samples: probabilities, crop survival, box area,
class retention, and visual plausibility. Split data before augmenting and never
apply stochastic training transforms to validation or test data.

## Completion criteria

- [ ] Image and every annotation target share the same geometry.
- [ ] Coordinate format, label mapping, dtype, and range are tested.
- [ ] Replay or seed evidence reproduces a transformed sample.
- [ ] Distribution and domain-risk audits justify each transform.
