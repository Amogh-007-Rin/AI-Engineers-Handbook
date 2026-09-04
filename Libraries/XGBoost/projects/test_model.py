import tempfile, unittest
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from model import train


class XGBoostTests(unittest.TestCase):
    def test_training_quality_and_native_round_trip(self):
        model, score, test_x = train()
        self.assertGreater(score, .8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save_model(path)
            restored = XGBClassifier(); restored.load_model(path)
            np.testing.assert_allclose(model.predict_proba(test_x), restored.predict_proba(test_x))

    def test_seed_is_reproducible(self):
        first, _, x = train(); second, _, _ = train()
        np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))

if __name__ == "__main__": unittest.main()
