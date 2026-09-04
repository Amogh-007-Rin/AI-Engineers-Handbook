import math
import unittest

from linear_algebra import (
    determinant_2x2, dot, matmul, matrix, matvec, norm, project,
    sensitivity_example, solve_2x2, transpose, vector,
)


class LinearAlgebraTests(unittest.TestCase):
    def test_vector_and_matrix_validation(self):
        self.assertEqual(vector([1, 2]), (1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            vector([])
        with self.assertRaisesRegex(ValueError, "finite"):
            vector([math.nan])
        with self.assertRaisesRegex(ValueError, "equal"):
            matrix([[1, 2], [3]])

    def test_dot_and_norm(self):
        self.assertEqual(dot((1, 2, 3), (4, -1, 2)), 8.0)
        self.assertEqual(norm((3, 4)), 5.0)
        with self.assertRaisesRegex(ValueError, "equal dimensions"):
            dot((1,), (1, 2))

    def test_transpose_swaps_axes(self):
        self.assertEqual(transpose(((1, 2, 3), (4, 5, 6))), ((1, 4), (2, 5), (3, 6)))

    def test_matvec_validates_the_shared_dimension(self):
        self.assertEqual(matvec(((1, 2), (3, 4)), (2, -1)), (0.0, 2.0))
        with self.assertRaisesRegex(ValueError, "columns"):
            matvec(((1, 2),), (1,))

    def test_matmul_composes_transformations(self):
        self.assertEqual(matmul(((1, 2), (3, 4)), ((2, 0), (1, 2))), ((4, 4), (10, 8)))
        with self.assertRaisesRegex(ValueError, "left columns"):
            matmul(((1, 2),), ((1, 2),))

    def test_projection_is_orthogonal_to_the_residual(self):
        value = (3, 4)
        projection = project(value, (1, 1))
        residual = tuple(x - p for x, p in zip(value, projection))
        self.assertAlmostEqual(dot(residual, (1, 1)), 0.0)
        with self.assertRaisesRegex(ValueError, "non-zero"):
            project(value, (0, 0))

    def test_determinant_and_solve(self):
        source = ((2, 1), (1, 3))
        self.assertEqual(determinant_2x2(source), 5.0)
        solution = solve_2x2(source, (5, 5))
        self.assertEqual(matvec(source, solution), (5.0, 5.0))
        with self.assertRaisesRegex(ValueError, "singular"):
            solve_2x2(((1, 2), (2, 4)), (1, 2))

    def test_near_dependence_amplifies_perturbation(self):
        self.assertGreater(sensitivity_example(), 100_000)


if __name__ == "__main__":
    unittest.main()
