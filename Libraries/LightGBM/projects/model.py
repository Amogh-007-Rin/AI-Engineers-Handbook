from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


def train(seed=7):
    x, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=seed)
    a, b, c, d = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed)
    model = LGBMClassifier(n_estimators=40, num_leaves=15, max_depth=4, learning_rate=.1,
                           n_jobs=1, random_state=seed, verbosity=-1)
    model.fit(a, c)
    return model, float(roc_auc_score(d, model.predict_proba(b)[:, 1])), b
