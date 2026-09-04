"""Real Orbit DLT smoke model for the isolated Python 3.11 profile."""

import pandas as pd
from orbit.models import DLT


def fit_predict():
    train = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30, freq="D"), "value": range(30)})
    model = DLT(response_col="value", date_col="date", estimator="stan-map", seasonality=None)
    model.fit(train)
    future = pd.DataFrame({"date": pd.date_range("2024-01-31", periods=3, freq="D")})
    return model.predict(future)
