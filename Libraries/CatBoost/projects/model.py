from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


def train(seed=7):
    x, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=seed)
    a, b, c, d = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed)
    model = CatBoostClassifier(iterations=40, depth=4, learning_rate=.1, random_seed=seed,
                               thread_count=1, verbose=False, allow_writing_files=False)
    model.fit(a, c)
    return model, float(roc_auc_score(d, model.predict_proba(b)[:, 1])), b
