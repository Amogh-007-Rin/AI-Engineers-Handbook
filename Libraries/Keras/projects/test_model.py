import os, tempfile, unittest, warnings
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-engineers-handbook-matplotlib")
import keras
from model import data, train


class KerasTests(unittest.TestCase):
    def compatibility_warnings(self):
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.filterwarnings("ignore", message="__array__ implementation doesn't accept a copy keyword.*", category=DeprecationWarning)
        self.addCleanup(context.__exit__, None, None, None)

    def test_training_and_native_round_trip(self):
        model, losses = train(); x, y = data()
        self.assertLess(losses[-1], 1e-5)
        self.compatibility_warnings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"; model.save(path)
            restored = keras.models.load_model(path)
            np.testing.assert_allclose(model.predict(x, verbose=0), restored.predict(x, verbose=0), atol=1e-6)

    def test_seeded_training(self):
        first, _ = train(10); second, _ = train(10)
        self.compatibility_warnings()
        for left, right in zip(first.get_weights(), second.get_weights(), strict=True):
            np.testing.assert_allclose(left, right)

if __name__ == "__main__": unittest.main()
