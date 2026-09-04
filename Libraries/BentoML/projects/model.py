"""Dependency-free Bento service specification validator."""

import math


def validate_service(spec):
    for key in ("name", "model_ref", "input_schema", "output_schema", "timeout_ms", "max_batch"):
        if key not in spec:
            raise ValueError(f"missing service field: {key}")
    if "@" not in spec["model_ref"] or spec["timeout_ms"] <= 0 or spec["max_batch"] < 1:
        raise ValueError("model must be versioned and limits positive")
    if not spec.get("readiness") or not spec.get("metrics"):
        raise ValueError("readiness and metrics are required")
    return True


def predict_score(features):
    if (not isinstance(features, list) or not features or len(features) > 32
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in features)):
        raise ValueError("features must contain 1-32 finite numeric values")
    return sum(float(value) for value in features) / len(features)


def build_native_service():
    """Construct a real BentoML 1.x service without starting a network server."""
    import bentoml

    @bentoml.service(
        name="handbook_predict",
        traffic={"timeout": 5, "max_concurrency": 8},
    )
    class PredictionService:
        @bentoml.api(route="/predict")
        def predict(self, features: list[float]) -> dict[str, float]:
            return {"score": predict_score(features)}

    return PredictionService
