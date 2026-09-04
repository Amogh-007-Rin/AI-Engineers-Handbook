"""Small real PyCaret classification experiment for the Python 3.11 profile."""

from __future__ import annotations
import pandas as pd
from pycaret.classification import ClassificationExperiment


FEATURES = ["age", "income", "country"]


def dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20, 24, 31, 35, 42, 48, 56, 61, 67, 72] * 3,
        "income": [20, 25, 32, 39, 48, 60, 72, 81, 94, 105] * 3,
        "country": ["GB", "US"] * 15,
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 3,
    })


def train():
    experiment = ClassificationExperiment()
    experiment.setup(data=dataset(), target="target", session_id=7, fold=3, html=False, verbose=False, n_jobs=1)
    model = experiment.create_model("lr", verbose=False)
    return experiment, model


def validate_inference(frame: pd.DataFrame) -> pd.DataFrame:
    missing, extra = set(FEATURES) - set(frame.columns), set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return frame.loc[:, FEATURES]
