---
title: Vectors matrices and linear transformations
slug: linear-algebra-foundations
level: foundation
stage: mathematics
estimated_hours: 8
prerequisites:
  - python-foundations
learning_objectives:
  - Interpret vectors and matrices as data and transformations
  - Compute and verify shapes dot products and matrix products
  - Explain how linear algebra represents an ML prediction
  - Diagnose dependence conditioning and floating point limitations
  - Connect basis projections norms and transformations to AI workflows
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Vectors, matrices, and linear transformations

Linear algebra describes quantities with several coordinates and operations
that preserve addition and scaling. In AI, it expresses observations, model
parameters, embeddings, image channels, attention projections, gradients, and
batched computation. Symbols become useful only after every axis has a meaning.

This eight-hour core path uses Python 3.10 or newer and no third-party package.
It runs offline on a CPU in seconds with negligible memory. Every diagram is
also described in text. Color is not used as the only carrier of meaning, so
the core material remains accessible in plain text and to screen readers. NumPy
appears only after you can explain the underlying operation.

## Outcomes and evidence

You will calculate a linear prediction by hand, implement shape-safe vector and
matrix operations, test malformed inputs, interpret transformations geometrically,
and demonstrate why near-dependent systems amplify error. Evidence consists of
the eight-test lab, written derivations, exercises, the independent
[numerical project](../project/README.md), and at least 80/100 on the
[mathematics assessment](../assessment.md).

## Scalars, vectors, matrices, and tensors

A scalar is one number. A vector is an ordered list of coordinates relative to
a chosen basis. A matrix is a rectangular array that may store a dataset, a
linear transformation, pairwise relationships, or several vectors. A tensor is
a multidimensional array; its axes still require domain meaning.

Suppose a data matrix `X` contains `n` observations and `d` features, and `w`
contains one weight per feature:

```text
X: (n observations, d features)
w: (d features)
Xw: (n predictions)
```

The shared feature dimension must agree. Shape is necessary but not sufficient:
height in metres and temperature in Celsius could share a length while remaining
semantically incompatible. Record axis names, units, ordering, and missing-value
meaning at interfaces.

## Vector operations and geometry

For vectors `x = (x₁, …, x_d)` and `y = (y₁, …, y_d)`:

- addition is coordinatewise: `(x + y)ᵢ = xᵢ + yᵢ`;
- scaling is coordinatewise: `(αx)ᵢ = αxᵢ`;
- the dot product is `x · y = Σᵢ xᵢyᵢ`; and
- the Euclidean norm is `||x||₂ = √(x · x)`.

The dot product measures aligned magnitude. If nonzero vectors have dot product
zero, they are orthogonal. The angle relation
`x · y = ||x||₂ ||y||₂ cos θ` explains cosine similarity, but a high cosine
does not prove two records have the same meaning. Preprocessing, embedding model,
population, and task define what geometry is useful.

Projection of `x` onto nonzero vector `u` is
`proj_u(x) = (x · u / u · u)u`. The scalar coefficient says how far along `u`
the projection lies. Recommendations, latent-factor models, and attention all
use related ideas, although their learned spaces require empirical validation.

### Worked example: score one observation

Let an observation contain normalized features `x = (2, -1, 3)` and weights
`w = (0.5, 2, -1)`. Then:

```text
x · w = 2(0.5) + (-1)(2) + 3(-1)
      = 1 - 2 - 3
      = -4
```

Increasing the second weight by `0.1` changes this prediction by
`x₂(0.1) = -0.1`. More generally, the sensitivity of `x · w` to weight `w_j`
is feature `x_j`. This is a local algebraic fact, not proof that the feature
causes the outcome.

## Matrices as transformations

An `m × n` matrix maps a vector in `n`-coordinate input space to an
`m`-coordinate output. Its columns are where the input basis vectors go. For:

```text
A = [[2, 0],
     [0, 1]]
```

the first coordinate doubles while the second stays fixed. A text-described
unit square with corners `(0,0), (1,0), (0,1), (1,1)` becomes a rectangle with
corners `(0,0), (2,0), (0,1), (2,1)`.

A transformation `T` is linear when `T(x + y) = T(x) + T(y)` and
`T(αx) = αT(x)`. Translation by a constant is affine rather than linear. ML
layers commonly compute `Wx + b`: a linear transformation plus bias.

## Matrix multiplication and composition

If `A` has shape `(m, n)` and `B` has shape `(n, p)`, their product `AB` has
shape `(m, p)` with:

```text
(AB)[i,j] = Σ_k A[i,k] B[k,j]
```

Entry `(i,j)` is row `i` of `A` dotted with column `j` of `B`. The operation
represents composition: `ABx` applies `B` first, then `A`. Order generally
matters; `AB` may differ from `BA`, or one order may not be defined.

Elementwise multiplication instead pairs entries at matching coordinates. It
does not compose transformations. Libraries use distinct operators because
confusing these operations can produce either an immediate shape error or,
worse, a plausible but wrong result.

The transpose `Aᵀ` swaps axes: `(Aᵀ)[i,j] = A[j,i]`. For a data matrix with
observations in rows, `XᵀX` contains feature cross-products. Its shape is
`(d,d)`, regardless of observation count. Forming it explicitly can worsen
conditioning and is often avoided by robust least-squares algorithms.

## Systems, rank, and independence

The equation `Ax = b` asks which input maps to output `b`. A unique solution
requires enough independent constraints. Rank counts independent directions in
the row or column space. If one feature is an exact combination of others, the
matrix is rank-deficient and parameters may not be uniquely identifiable.

For a `2 × 2` matrix `[[a,b],[c,d]]`, determinant `ad - bc` is the signed area
scale. A zero determinant means the transformation collapses the plane into a
lower-dimensional set and has no inverse. Determinants are useful intuition,
but production solvers do not decide general invertibility by testing a
floating-point determinant for exact zero.

## Numerical stability and conditioning

Mathematically valid operations can be numerically unreliable. Floating-point
numbers represent a finite subset of real values. Subtracting nearly equal
values can discard meaningful digits; very large intermediate products can
overflow; repeated summation accumulates rounding error.

Conditioning describes sensitivity of the mathematical problem to small input
changes. Stability describes how an algorithm behaves under rounding. A stable
algorithm cannot make a fundamentally ill-conditioned question informative.

Consider:

```text
x + y = 2
x + 1.000001y = 2.000001
```

The equations are nearly parallel. Tiny changes in their right-hand sides can
cause large relative changes in the recovered parameters. In ML this appears
with highly correlated features, poorly scaled optimization, and inverse-like
operations. Report tolerances, scaling, data precision, and perturbation tests.

Never compare general floating-point calculations only with exact equality.
Choose absolute and relative tolerances based on scale and downstream decision.
Also reject `NaN` and infinity explicitly: comparisons involving `NaN` can
silently be false.

## Decompositions and AI applications

A decomposition rewrites a matrix into structured factors. You should recognize:

- **LU** for solving suitable square systems;
- **QR** for least squares without directly forming `XᵀX`;
- **eigendecomposition** for invariant directions of some square matrices; and
- **singular value decomposition (SVD)** for rank, compression, pseudoinverses,
  and principal-component reasoning.

If `A = UΣVᵀ`, singular values in `Σ` describe how strongly orthogonal input
directions are scaled. Small singular values identify directions where inversion
amplifies noise. Truncating them yields a lower-rank approximation, trading
detail for compression or regularization. The choice must be evaluated against
task performance, subgroup behavior, and reconstruction error—not aesthetics.

## Runnable lab: transparent matrix operations

The dependency-free lab implements validation, dot products, transposition,
matrix-vector/matrix-matrix multiplication, norms, projections, determinants,
and a sensitivity demonstration. Run from the repository root:

```bash
python3 curriculum/mathematics/01-linear-algebra/lab/linear_algebra.py
python3 -m unittest discover -s curriculum/mathematics/01-linear-algebra/lab -v
```

Expected JSON includes predictions `[2.5, -3.0, 1.5]`, an orthogonal projection,
and a sensitivity ratio greater than 100,000. Eight tests should pass. Read the
[lab guide](lab/README.md) before editing the implementation.

## Failure practice and debugging

Trigger a ragged matrix, empty vector, mismatched inner dimension, non-finite
value, projection onto the zero vector, and singular solve. For each, identify
whether the failure is structural, semantic, or numerical. Then deliberately
swap matrix order and explain why a shape-compatible output can still represent
the wrong composition.

When a numerical result looks wrong, check axis meanings and shapes first, then
units, finiteness, scaling, condition sensitivity, and tolerance. Printing more
digits does not restore lost information.

## Exercises

1. **Recall:** define dot product, norm, transpose, rank, conditioning, and
   stability without notation, then restore the notation.
2. **Hand calculation:** multiply a `2 × 3` matrix by a length-three vector and
   annotate every axis and unit.
3. **Implementation:** add matrix addition with strict rectangular validation
   and tests for incompatible and empty inputs.
4. **Analysis:** demonstrate numerically that `(AB)C = A(BC)` within tolerance,
   then find an example showing `AB != BA`.
5. **Extension:** implement Gram–Schmidt for two independent vectors. Test the
   result's norms and dot product, then explore near dependence.
6. **AI application:** construct a tiny design matrix and explain how feature
   scaling changes gradient-based optimization without changing the underlying
   exact linear relationship.

## Common misconceptions

- **“A vector is always an arrow.”** It can represent parameters, tokens,
  samples, or directions; state the semantics.
- **“Matching lengths mean compatible data.”** Units and axis ordering can still
  disagree.
- **“Matrix multiplication is elementwise.”** It contracts a shared dimension
  and composes transformations.
- **“A small determinant proves singularity.”** Scale and conditioning matter;
  use suitable factorization and tolerance.
- **“More dimensions always preserve information.”** Features can be redundant,
  noisy, leaked, or unsupported by sample size.
- **“Numerical error is random noise.”** Algorithms can systematically amplify
  rounding through ill-conditioned operations.

## Knowledge check

1. Why must the inner dimensions of `AB` match?
2. What does column `j` of a transformation matrix represent?
3. Why is `Wx + b` affine rather than strictly linear when `b != 0`?
4. How do rank deficiency and ill-conditioning differ?
5. Why might QR be preferred to solving normal equations with `XᵀX`?
6. What does a small singular value imply about inversion?
7. What evidence would justify a floating-point tolerance?

Score one point per precise answer with an example. Below six requires a new
hand calculation and sensitivity experiment before the project.

## Completion criteria

- [ ] Hand calculations and lab outputs agree within justified tolerances.
- [ ] All eight tests pass in the documented environment.
- [ ] You triggered and classified six specified failure modes.
- [ ] You completed four exercises, including implementation and AI application.
- [ ] You can explain every matrix axis without relying on syntax.
- [ ] You distinguish conditioning of a problem from stability of an algorithm.
- [ ] You score at least 6/7 here and 80/100 on the mathematics gate.

## Summary and glossary additions

Vectors hold coordinates, matrices store data or linear maps, and matrix
multiplication composes maps by contracting a shared dimension. Rank describes
independent directions; decompositions reveal useful structure; conditioning
and algorithmic stability determine whether finite-precision answers are
trustworthy.

- **Basis:** independent vectors used as coordinates for a space.
- **Conditioning:** sensitivity of a problem's output to input perturbations.
- **Rank:** number of independent row or column directions.
- **Stability:** resistance of an algorithm to rounding and intermediate error.

## Authoritative further reading

- [Python floating-point tutorial](https://docs.python.org/3/tutorial/floatingpoint.html)
- [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [LAPACK user guide](https://www.netlib.org/lapack/lug/)
- [Matrix Computation course notes, MIT OpenCourseWare](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)

The algebraic definitions are stable. Solver behavior, tolerances, supported
dtypes, and acceleration are library- and version-sensitive; consult exact
documentation and test the deployed environment.

Continue with the [numerical verification project](../project/README.md) and
the [NumPy academy](../../../Libraries/NumPy/README.md).
