# Albumentations target-integrity project

Flip a synthetic image and Pascal VOC box, verify geometry, and replay the exact
transform. Run `python -W error -m unittest -v`; extend with masks, keypoints,
crop-survival metrics, and a domain-specific augmentation audit. The project
disables Albumentations' import-time update request so offline and privacy-
sensitive execution is deterministic; handle dependency update checks in the
surrounding release process.
