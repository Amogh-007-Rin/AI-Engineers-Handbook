---
title: Classical machine learning model families
slug: classical-model-families
level: practitioner
stage: machine-learning
estimated_hours: 14
prerequisites:
  - ml-preprocessing-pipelines
learning_objectives:
  - Explain inductive biases of major classical model families
  - Select baselines from data constraints rather than popularity
  - Diagnose underfitting overfitting calibration and threshold errors
  - Compare supervised unsupervised and anomaly objectives responsibly
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Classical machine-learning model families

A model family defines which patterns can be represented, how evidence changes
parameters, and what computation occurs during training and prediction. This
inductive bias is useful when it resembles the problem and harmful when it
encodes the wrong invariance. Selection begins with data, decision, sample size,
feature geometry, and operational constraints—not popularity.

This fourteen-hour core path uses Python 3.10 or newer. Its from-scratch lab is
offline, CPU-only, dependency-free, and completes in seconds. All comparisons
are labeled in text, so color is not the sole carrier of meaning; outputs are
plain JSON suitable for assistive technology.

## Outcomes and evidence

You will explain linear, neighbor, probabilistic, tree, ensemble, kernel,
clustering, reduction, and anomaly models; implement representative transparent
algorithms; identify failure modes; and write a justified model shortlist.
Completion requires eight lab tests, four exercises, the
[stage project](../project/README.md), and a model card explaining one rejected
candidate as carefully as the selected one.

## Bias, variance, capacity, and regularization

Prediction error can arise because a family cannot express the pattern
(underfitting), because fitting follows sample noise (overfitting), or because
data and targets do not represent deployment. More parameters do not map to
complexity identically across families. Measure train/validation learning curves,
seed variation, slice behavior, and residual patterns.

Regularization constrains a fitted solution. L2 penalties discourage large
weights smoothly; L1 can produce zeros; tree depth, neighbor count, kernel width,
and early stopping also regulate effective capacity. Regularization cannot fix
leakage, invalid targets, or selection on the test set.

## Linear and generalized linear models

Linear regression predicts an additive weighted sum and commonly minimizes
squared residuals. It is fast, extrapolates along a plane, and provides a strong
baseline. Correlated features make individual coefficients unstable; outliers
can dominate squared loss; a good predictive coefficient is not a causal effect.

Logistic regression models linear log-odds and outputs a score through the
logistic function. Its probability quality and action threshold remain separate
evaluation questions. Feature interactions and nonlinear bases expand what it
can represent while changing interpretability and overfitting risk.

### Worked example: one-dimensional least squares

For points `(1,2), (2,4), (3,6)`, slope is covariance divided by variance:

```text
mean x = 2, mean y = 4
slope = Σ(x-2)(y-4) / Σ(x-2)² = 4 / 2 = 2
intercept = 4 - 2(2) = 0
```

Prediction at `x=4` is `8`. With constant `x`, the denominator is zero and a
unique slope cannot be learned; an intentional error is more honest than an
arbitrary coefficient.

## Nearest neighbors and distance

Nearest-neighbor methods store training examples and defer work to prediction.
They can express irregular local boundaries without a parametric equation, but
prediction cost and storage grow with data. Results depend on distance, scaling,
irrelevant dimensions, neighbor count, and tie policy.

In high dimensions distances can become less discriminative. Standardization
does not prove every feature deserves equal importance. Validate the metric and
representation against the task, and prevent exact or near-duplicate leakage.

## Probabilistic and kernel models

Naive Bayes estimates class evidence under strong conditional-independence
assumptions. Those assumptions are often false, yet the model can remain an
excellent sparse-text baseline because estimation is efficient. Smooth unseen
events deliberately and inspect probability calibration.

Kernel methods represent similarity through pairwise evaluations. A linear
kernel retains a linear boundary; nonlinear kernels add flexible boundaries.
Kernel choice and width encode geometry, feature scaling matters, and training
or prediction may scale poorly with observation count.

## Trees, bagging, and boosting

A decision tree repeatedly splits feature space to reduce an impurity or loss.
It handles nonlinear interactions and mixed scaling, but a deep tree can memorize
noise and change dramatically after a small data perturbation.

A decision stump is one split. It is weak but inspectable: choose a feature,
threshold, predictions on each side, and loss. The lab enumerates midpoint
thresholds and uses deterministic tie-breaking, making the search auditable.

Bagging fits models on perturbed samples and averages them, reducing variance
when errors are not perfectly correlated. Random forests also vary candidate
features. Gradient boosting fits corrections sequentially and is often powerful
for tabular data, but needs validation of depth, learning rate, rounds, early
stopping, calibration, and missing-value behavior.

Feature importance from a tree describes model behavior under a method's
assumptions; it does not establish causal influence. High-cardinality features
can distort some importance measures.

## Ensembles

Voting, averaging, stacking, and boosting combine models. Diversity helps when
component errors differ. Stacking must generate meta-model training predictions
out of fold; fitting the meta-model on in-sample predictions leaks performance.
An ensemble also increases latency, artifact count, failure modes, and monitoring
burden. Require measured benefit over its strongest component.

## Clustering and dimensionality reduction

Clustering optimizes an objective under a representation and similarity. K-means
minimizes squared distance to centroids, favoring roughly compact Euclidean
clusters and requiring a cluster count. Its numeric labels have no intrinsic
meaning or stable identity across runs.

Do not present a cluster as a discovered human category without external evidence.
Evaluate initialization sensitivity, stability, compactness, meaningful external
outcomes, and downstream utility. Scaling and outliers can dominate centroids.

PCA finds orthogonal directions of decreasing variance. It supports compression,
noise analysis, and visualization, but high variance is not automatically high
task value. Fit it within training folds and report explained variance plus
downstream and reconstruction effects. Two-dimensional visual separation is not
proof of natural classes.

## Anomaly detection and recommendation

Anomaly methods score deviation from a reference distribution. Rare does not
mean harmful, and harmful cases may resemble common ones. Evaluate with scarce
labels, review capacity, temporal drift, false-alert cost, and adversarial behavior.
Thresholds require operational ownership.

Recommendation methods estimate relevance from users, items, and interactions.
Popularity is a necessary baseline. Logged feedback reflects previous ranking,
exposure, position, and selection, so offline metrics can reinforce feedback
loops. Evaluate novelty, coverage, latency, long-term effects, and user control.

## Selecting a family

Build a shortlist using:

- sample count, feature types, sparsity, missingness, and dimensionality;
- expected smoothness, locality, interactions, monotonicity, and extrapolation;
- label quality, imbalance, shift, and repeated entities;
- training budget, prediction latency, memory, update frequency, and portability;
- explanation, uncertainty, privacy, security, and governance needs; and
- evidence that would reject the candidate.

Begin with the simplest credible family. Compare identical splits, preprocessing,
metrics, uncertainty, and hardware. Tune only after the baseline and error
taxonomy are trustworthy.

## Runnable lab: representative model biases

The lab implements one-dimensional least squares, deterministic k-nearest-neighbor
classification, a binary decision stump, and seeded one-dimensional k-means:

```bash
python3 curriculum/machine-learning/03-model-families/lab/model_families.py
python3 -m unittest discover -s curriculum/machine-learning/03-model-families/lab -v
```

Expected output fits slope `2`, classifies a nearby point as `warm`, finds a
stump with zero training error, and returns two ordered centroids. Eight tests
pass. Read the [lab guide](lab/README.md) before editing.

## Failure practice and debugging

Trigger empty training data, unequal feature/label lengths, constant-x linear
data, invalid neighbor count, non-finite input, a stump with one unique feature
value, more clusters than unique points, and an iteration limit below one.
Then scale one feature by 1,000 in a distance example and explain the changed
neighbor without calling the original result “wrong.”

When a candidate overfits, compare train and validation behavior, simplify its
capacity, inspect leakage, and collect learning curves before switching families.
When seeds disagree, report the distribution rather than selecting the best run.

## Exercises

1. **Recall:** state the primary bias, scaling behavior, prediction cost, and
   failure mode of five families.
2. **Implementation:** extend least squares with MAE reporting and a residual
   table; identify why fitting absolute error requires a different optimizer.
3. **Analysis:** compare `k=1` and larger odd `k` on a noisy boundary and explain
   the bias/variance change.
4. **Trees:** add a maximum-error criterion to the stump and compare its chosen
   threshold with classification error.
5. **Clustering:** rerun k-means under transformed units and several seeds;
   quantify assignment stability without interpreting cluster names.
6. **Selection:** write a model shortlist for sparse text, regulated small data,
   nonlinear tabular data, and million-item similarity search, including rejection
   evidence and operational cost.

## Common misconceptions

- **“Complex models always win.”** They must beat credible baselines under the
  same protocol and operational constraints.
- **“Linear means simplistic.”** Suitable features can yield strong, stable
  systems; coefficients still need careful interpretation.
- **“Trees need no preprocessing.”** Schema, missingness, categories, leakage,
  and data validity remain essential.
- **“Clusters are discovered classes.”** They are outcomes of representation,
  metric, objective, initialization, and chosen cluster count.
- **“Feature importance is causal.”** It describes model behavior, often under
  dependence-sensitive assumptions.
- **“The best seed is the model.”** Selecting lucky randomness biases evidence.

## Knowledge check

1. What is an inductive bias, and why is it unavoidable?
2. Why is linear-regression slope undefined for constant input?
3. How do scaling and dimension affect nearest neighbors?
4. Why can bagging reduce tree variance?
5. How must stacking predictions be generated for the meta-model?
6. What does k-means optimize, and what does it not discover automatically?
7. Which result would make you reject a more complex candidate?

Score one point per precise answer with an example. Below six requires a changed
dataset, new comparison, and written correction before advancing.

## Completion criteria

- [ ] All eight lab tests pass and expected failures are explained.
- [ ] Every candidate identifies inductive bias and operational cost.
- [ ] Baselines are simpler than candidates and use the same evaluation protocol.
- [ ] Scaling, randomness, capacity, and leakage are tested where relevant.
- [ ] Predictive explanation is not confused with causality.
- [ ] You complete four exercises and score at least 6/7 here.
- [ ] The stage project documents one rejected family and why.

## Summary and glossary additions

Model families encode different assumptions about additivity, locality,
similarity, partitioning, variance, and latent structure. Selection is a tested
engineering decision spanning data, evidence, compute, risk, and operations.

- **Inductive bias:** assumptions that allow generalization beyond observed data.
- **Capacity:** range of patterns a fitted model can express.
- **Bagging:** averaging models fit on perturbed samples to reduce variance.
- **Centroid:** representative mean optimized by k-means for one cluster.

## Authoritative further reading

- [scikit-learn supervised learning guide](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html)
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

The family-level principles are stable. Solver defaults, category handling,
parallelism, probability behavior, and persistence are version-sensitive.

Continue with [tuning and interpretability](../04-tuning-interpretability/README.md).
