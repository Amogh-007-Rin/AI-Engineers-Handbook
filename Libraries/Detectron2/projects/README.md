# Detectron2 dataset-contract project

Validate Detectron2-style image dictionaries, absolute XYXY boxes, and contiguous
class IDs without downloading weights. Run `python -W error -m unittest -v`;
extend with masks/keypoints, DatasetCatalog registration, training, error
analysis, and export parity.
