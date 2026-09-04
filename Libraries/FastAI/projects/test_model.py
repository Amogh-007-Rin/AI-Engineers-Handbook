import tempfile
import unittest
from pathlib import Path

from fastai.learner import load_learner

from model import export, frame, train


class FastAIProjectTests(unittest.TestCase):
    def test_split_is_disjoint(self):
        learner = train()
        self.assertFalse(set(learner.dls.train.items.index) & set(learner.dls.valid.items.index))

    def test_export_round_trip(self):
        learner, row = train(), frame().iloc[0]
        before = learner.predict(row)[0]
        with tempfile.TemporaryDirectory() as directory:
            after = load_learner(export(learner, Path(directory) / "export.pkl")).predict(row)[0]
        self.assertEqual(str(before), str(after))


if __name__ == "__main__":
    unittest.main()
