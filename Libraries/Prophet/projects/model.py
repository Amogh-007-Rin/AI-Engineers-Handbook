"""Small deterministic Prophet trend fixture."""

from __future__ import annotations
import pandas as pd
from prophet import Prophet


def training_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame({"ds": dates, "y": [10 + .5 * index for index in range(40)]})


def fit_and_forecast(periods: int = 5) -> pd.DataFrame:
    if periods <= 0:
        raise ValueError("periods must be positive")
    data = training_frame()
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False,
                    uncertainty_samples=0, changepoint_prior_scale=.01)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq="D", include_history=False)
    return model.predict(future).loc[:, ["ds", "yhat"]]
