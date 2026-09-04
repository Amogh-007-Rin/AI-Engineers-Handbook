"""MMDetection-style resolved configuration validator."""


def validate_config(config):
    for key in ("model", "train_dataloader", "val_dataloader", "val_evaluator", "train_cfg", "default_hooks"):
        if not config.get(key):
            raise ValueError(f"missing resolved config section: {key}")
    classes = config.get("metainfo", {}).get("classes", [])
    head_classes = config["model"].get("bbox_head", {}).get("num_classes")
    if not classes or head_classes != len(classes) or len(set(classes)) != len(classes):
        raise ValueError("dataset classes and model head must match uniquely")
    if not config.get("load_from") or "@" not in config["load_from"]:
        raise ValueError("pretrained/checkpoint artifact must be versioned")
    return True
