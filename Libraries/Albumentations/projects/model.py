"""Deterministic Albumentations image/box transform contract."""

import os

# Keep training and CI imports deterministic, offline, and free of update
# telemetry. Applications can perform dependency checks in their own release
# process instead of at model-import time.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np


PIPELINE = A.ReplayCompose(
    [A.HorizontalFlip(p=1.0)],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)


def augment(image, boxes, labels):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or len(boxes) != len(labels):
        raise ValueError("expected HxWx3 image and aligned boxes/labels")
    result = PIPELINE(image=image, bboxes=boxes, labels=labels)
    return result
