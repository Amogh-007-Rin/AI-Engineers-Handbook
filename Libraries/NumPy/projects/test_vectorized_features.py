import unittest
import numpy as np

from vectorized_features import Standardizer, cosine_similarity, matrix, pairwise_interactions


class FeatureTests(unittest.TestCase):
    def test_standardizer_does_not_mutate_and_handles_constant(self) -> None:
        source = np.array([[1.0, 4.0], [3.0, 4.0]])
        original = source.copy()
        transformed = Standardizer().fit_transform(source)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_allclose(transformed.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_array_equal(transformed[:, 1], [0, 0])

    def test_fit_state_and_feature_contract(self) -> None:
        with self.assertRaises(RuntimeError):
            Standardizer().transform([[1]])
        with self.assertRaises(ValueError):
            Standardizer().fit([[1, 2]]).transform([[1]])

    def test_interactions_have_stable_order(self) -> None:
        np.testing.assert_array_equal(pairwise_interactions([[2, 3, 5]]), [[6, 10, 15]])

    def test_cosine_handles_zero_rows(self) -> None:
        result = cosine_similarity([[1, 0], [0, 1], [0, 0]])
        np.testing.assert_allclose(result, np.diag([1, 1, 0]), atol=1e-12)

    def test_invalid_input(self) -> None:
        for value in ([], [1, 2], [[float("nan")]]):
            with self.assertRaises(ValueError):
                matrix(value)


if __name__ == "__main__":
    unittest.main()
