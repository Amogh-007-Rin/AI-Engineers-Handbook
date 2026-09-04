import unittest

from baselines import MajorityClassifier, accuracy, grouped_split, precision_recall


class MajorityClassifierTests(unittest.TestCase):
    def test_fit_predict_and_metrics(self) -> None:
        model = MajorityClassifier().fit([0, 1, 1])
        predicted = model.predict(3)
        self.assertEqual(predicted, [1, 1, 1])
        self.assertAlmostEqual(accuracy([0, 1, 1], predicted), 2 / 3)
        self.assertEqual(precision_recall([0, 1, 1], predicted, 1), (2 / 3, 1.0))

    def test_requires_training_data(self) -> None:
        with self.assertRaises(ValueError):
            MajorityClassifier().fit([])
        with self.assertRaises(RuntimeError):
            MajorityClassifier().predict(1)

    def test_group_split_has_no_overlap_and_is_reproducible(self) -> None:
        groups = ["a", "a", "b", "b", "c", "c", "d"]
        first = grouped_split(groups, test_fraction=0.25, seed=7)
        second = grouped_split(groups, test_fraction=0.25, seed=7)
        self.assertEqual(first, second)
        train, test = first
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})
        self.assertEqual(sorted(train + test), list(range(len(groups))))


if __name__ == "__main__":
    unittest.main()
