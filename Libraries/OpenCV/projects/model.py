"""OpenCV image contract and deterministic normalization."""

import cv2
import numpy as np


def normalize(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("expected HxWx3 uint8 BGR image")
    if array.size == 0:
        raise ValueError("image must be non-empty")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def resize(image, width, height):
    if width < 1 or height < 1:
        raise ValueError("positive target dimensions required")
    return cv2.resize(normalize(image), (width, height), interpolation=cv2.INTER_AREA)
