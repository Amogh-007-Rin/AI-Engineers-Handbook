from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train(seed: int = 7):
    x, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=seed)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed)
    model = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=.1, subsample=.9,
                          colsample_bytree=.9, eval_metric="logloss", n_jobs=1, random_state=seed)
    model.fit(train_x, train_y)
    score = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    return model, float(score), test_x
