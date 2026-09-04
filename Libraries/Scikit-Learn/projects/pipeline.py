"""Leakage-safe heterogeneous classification pipeline."""

from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = ["age", "income", "country"]


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    transform = ColumnTransformer([("numeric", numeric, ["age", "income"]), ("category", categorical, ["country"])])
    return Pipeline([("features", transform), ("model", LogisticRegression(max_iter=500, random_state=0))])


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES).difference(frame.columns)
    extra = set(frame.columns).difference(FEATURES)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return frame.loc[:, FEATURES]
