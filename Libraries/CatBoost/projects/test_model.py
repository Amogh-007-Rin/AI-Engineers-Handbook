import tempfile, unittest
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from model import train


class CatBoostTests(unittest.TestCase):
    def test_quality_and_round_trip(self):
        model, score, x = train(); self.assertGreater(score, .8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"; model.save_model(path)
            restored = CatBoostClassifier(); restored.load_model(path)
            np.testing.assert_allclose(model.predict_proba(x), restored.predict_proba(x))

    def test_reproducible(self):
        first, _, x = train(); second, _, _ = train()
        np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))

if __name__ == "__main__": unittest.main()
