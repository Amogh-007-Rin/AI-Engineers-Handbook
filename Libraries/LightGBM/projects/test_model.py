import tempfile, unittest
from pathlib import Path
import numpy as np
import lightgbm as lgb
from model import train


class LightGBMTests(unittest.TestCase):
    def test_quality_and_round_trip(self):
        model, score, x = train(); self.assertGreater(score, .8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.txt"; model.booster_.save_model(path)
            restored = lgb.Booster(model_file=str(path))
            np.testing.assert_allclose(model.predict_proba(x)[:, 1], restored.predict(x))

    def test_reproducible(self):
        first, _, x = train(); second, _, _ = train()
        np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))

if __name__ == "__main__": unittest.main()
