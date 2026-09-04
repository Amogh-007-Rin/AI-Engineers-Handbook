"""Tiny fastai tabular classifier and artifact fixture."""

from pathlib import Path

import pandas as pd
from fastai.tabular.all import Categorify, FillMissing, Normalize, TabularDataLoaders, tabular_learner


def frame():
    return pd.DataFrame({
        "age": [18, 22, 31, 45, 52, 61, 27, 39, 48, 56, 34, 43],
        "region": ["n", "s", "n", "w", "s", "w"] * 2,
        "label": ["low", "low", "low", "high", "high", "high"] * 2,
    })


def train(epochs=1):
    dls = TabularDataLoaders.from_df(
        frame(), y_names="label", cat_names=["region"], cont_names=["age"],
        procs=[Categorify, FillMissing, Normalize], valid_idx=[8, 9, 10, 11], bs=4,
    )
    learner = tabular_learner(dls, layers=[8])
    learner.fit(epochs, lr=0.03)
    return learner


def export(learner, destination):
    destination = Path(destination)
    learner.export(destination)
    return destination
