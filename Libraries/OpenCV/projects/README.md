# OpenCV image contract

Validate BGR uint8 images, convert to RGB float tensors, and resize without a
model download. Run `python -W error -m unittest -v`; extend with codec round
trips, annotation transforms, malformed uploads, and performance bounds.
