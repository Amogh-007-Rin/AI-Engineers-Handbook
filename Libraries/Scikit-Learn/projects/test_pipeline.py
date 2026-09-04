import unittest
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

from pipeline import build_pipeline, validate_features


class PipelineTests(unittest.TestCase):
    def data(self):
        x = pd.DataFrame({
            "age": [20, 22, 35, 37, 50, 52, 65, 67],
            "income": [20, None, 35, 38, 70, 72, 90, 95],
            "country": ["GB", "GB", "US", "US", "GB", "GB", "US", "US"],
        })
        return x, [0, 0, 0, 0, 1, 1, 1, 1], [1, 2, 3, 4, 1, 2, 3, 4]

    def test_grouped_pipeline_beats_constant_baseline(self) -> None:
        x, y, groups = self.data()
        splitter = GroupKFold(n_splits=4)
        pipeline_score = cross_val_score(build_pipeline(), x, y, groups=groups, cv=splitter, scoring="balanced_accuracy").mean()
        baseline_score = cross_val_score(DummyClassifier(strategy="most_frequent"), x, y, groups=groups, cv=splitter, scoring="balanced_accuracy").mean()
        self.assertGreater(pipeline_score, baseline_score)

    def test_unknown_category_and_schema_contract(self) -> None:
        x, y, _ = self.data()
        model = build_pipeline().fit(x, y)
        prediction = model.predict(pd.DataFrame({"age": [40], "income": [50], "country": ["FR"]}))
        self.assertEqual(len(prediction), 1)
        with self.assertRaises(ValueError):
            validate_features(x.assign(leak=y))


if __name__ == "__main__":
    unittest.main()
