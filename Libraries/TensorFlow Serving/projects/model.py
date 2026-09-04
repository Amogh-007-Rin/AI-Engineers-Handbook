"""TensorFlow Serving model-config contract validator."""


def validate_config(config):
    models = config.get("models", [])
    if not models:
        raise ValueError("at least one model is required")
    for model in models:
        if not model.get("name") or not model.get("base_path"):
            raise ValueError("model name and base path required")
        if not str(model.get("version_policy", "")).startswith("specific:"):
            raise ValueError("production policy must pin specific versions")
    return True
