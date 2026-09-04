---
title: Leakage-safe preprocessing and feature pipelines
slug: ml-preprocessing-pipelines
level: practitioner
stage: machine-learning
estimated_hours: 8
prerequisites:
  - ml-framing-evaluation
learning_objectives:
  - Fit preprocessing only on training observations
  - Build composable pipelines for numeric categorical and missing data
  - Test feature contracts and prevent train-serving skew
  - Serialize fitted state with provenance and compatibility checks
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Leakage-safe preprocessing and feature pipelines

Preprocessing is part of the learned model whenever it estimates state from
data. Means, medians, category vocabularies, selected features, scaling ranges,
and dimensionality projections influence predictions and must follow the same
split and version discipline as model parameters.

This eight-hour lesson uses Python 3.10 or newer and an offline, dependency-free
lab that runs in seconds on a CPU. Every feature is labeled in text; color is not
the only carrier of meaning, and JSON artifacts remain screen-reader accessible.

## Outcomes and evidence

You will specify a feature contract, fit state only on training observations,
transform validation and serving records without mutation, handle unseen values,
serialize state, and expose train-serving skew through tests. Completion requires
the eight-test lab, exercises, a failure analysis, and application of the
pipeline inside the [stage project](../project/README.md).

## Feature contracts precede transformation

For every input record name the prediction unit and time, then record each
feature's source, type, units, valid range, null meaning, availability, sensitivity,
and owner. Define exact required fields, ordering of emitted columns, category
policy, and transformation version. A schema match does not prove a feature is
available at prediction time.

### Worked example: two features

Assume training records contain numeric `age` and categorical `plan`. The
preprocessor learns median age `30`, center `30`, scale `10`, and categories
`["basic", "pro"]`. Record `{"age": 40, "plan": "pro"}` becomes:

```text
age_scaled = (40 - 30) / 10 = 1
plan=basic = 0
plan=pro   = 1
plan=other = 0
```

An unseen `enterprise` plan maps to an explicit `other` column, not silently to
`basic`. A missing age uses the training median and emits `age_missing = 1`.
The missing indicator is justified only if absence has legitimate meaning and
will exist under the same process at serving time.

## Fit and transform are different contracts

`fit` may inspect permitted training observations and returns learned state.
`transform` applies that frozen state to any compatible record. It must not add
new categories, recompute a mean, change output order, or consult labels.

A safe evaluation sequence is:

1. split raw observations according to deployment;
2. fit preprocessing on the training partition;
3. transform training and validation with the same fitted state;
4. fit the estimator on transformed training data; and
5. evaluate the entire pipeline on transformed validation data.

Cross-validation repeats every learned step independently in every fold. Fitting
an imputer or vocabulary on all rows before cross-validation leaks distribution
information even without explicitly passing targets.

## Numeric transformations

Standardization subtracts a training center and divides by a training scale.
It often helps gradient- and distance-based methods but is not universally
beneficial for tree models. Robust centers and scales can reduce outlier influence,
but do not make corrupted values trustworthy.

Zero-variance features require an explicit policy: reject, drop with recorded
lineage, or use a safe scale while marking the feature constant. Never divide by
zero or infer a tiny scale from validation data. Units must be checked before
scaling; mixing dollars and cents yields valid types and invalid meaning.

Missing-value handling follows the measurement process. Fit imputation values
on training data only, distinguish structural absence from collection failure,
and test all-missing columns. Dropping rows can shift the target population and
must reconcile counts.

## Categorical transformations

One-hot encoding creates one column per fitted category. Define ordering and
unknown behavior. Rejecting unknown values can be correct for a closed code set;
an explicit other bucket can be correct for evolving categories; hashing can
bound dimension while introducing collisions.

Target and frequency encodings are learned state. Target encoding must be
computed out-of-fold for training rows and then frozen, or each row can leak its
own label. High-cardinality identifiers may allow memorization and should not be
treated as ordinary categories without a defensible deployment reason.

## Feature composition and selection

Transformers for numeric, categorical, text, and timestamps should compose into
one versioned pipeline with deterministic output names. Preserve sparse
representations when dense expansion would exceed memory.

Feature selection, PCA, and learned embeddings are fitted steps. Put them inside
the validation loop. A manual choice made after viewing test performance is also
test leakage even if the code contains no `fit` call.

Time-derived features require a clock contract and timezone. Aggregates such as
“transactions in the previous 30 days” must use only records strictly available
at the prediction cutoff and apply the same late-arrival policy in training and
serving.

## Train-serving parity

Reimplementing transformations separately in a service creates skew through
different defaults, versions, units, field order, or category updates. Prefer
one tested artifact or generated contract consumed by both paths. Validate a
golden set through training and serving boundaries and compare exact feature
names plus values within justified numeric tolerances.

Serialize state with training-data version, code commit, schema fingerprint,
library/runtime versions, fitted timestamp, and compatibility range. Treat
deserialization as a security boundary: do not load untrusted pickle-like
artifacts, and verify provenance and integrity.

## Monitoring and evolution

Monitor missingness, unknown-category rate, ranges, schema changes, transform
failures, and feature-distribution shifts. A rising unknown rate may signal a
new product tier rather than mere drift. Assign an owner and response threshold.

Schema evolution can add optional fields compatibly, but reordering or changing
units can silently alter predictions. Version breaking changes and support
rollback. Preserve the old transformer while models depending on it remain live.

## Runnable lab: fitted feature pipeline

The dependency-free lab fits median/scale/category state, transforms records to
named tuples, serializes JSON, and rejects schema and compatibility errors:

```bash
python3 curriculum/machine-learning/02-preprocessing/lab/preprocessing.py
python3 -m unittest discover -s curriculum/machine-learning/02-preprocessing/lab -v
```

Expected output shows training median `30`, categories `basic` and `pro`, and an
unseen category activating `plan=other`. Eight tests pass. Read the
[lab guide](lab/README.md) before extending the pipeline.

## Failure practice and debugging

Trigger fit on an empty set, all-missing numeric input, extra or absent fields,
non-finite values, Boolean-as-number input, category collision with the reserved
token, and incompatible artifact version. Then demonstrate leakage by fitting a
median after appending an extreme validation record; explain why the program is
correct but the evaluation protocol is wrong.

When serving values differ, compare raw schema, units, feature names/order,
artifact checksum, runtime version, and one golden record before retraining.

## Exercises

1. **Recall:** list every kind of learned preprocessing state in this lesson.
2. **Implementation:** add a bounded numeric range and a configurable reject or
   clip policy. Test both and document why clipping can hide upstream defects.
3. **Leakage:** write a two-fold example showing why global median imputation
   changes validation features.
4. **Categories:** compare reject, other-bucket, and hashing policies for a
   regulated code set and a public search query.
5. **Parity:** create a golden feature fixture, serialize state, reload it, and
   assert feature names and values match.
6. **Operations:** design alerts for missingness and unknown rate with thresholds,
   windows, minimum samples, owner, and rollback action.

## Common misconceptions

- **“Preprocessing has no labels, so it cannot leak.”** Validation distribution
  and row identity can still influence fitted state.
- **“Scaling helps every model.”** Its value depends on the model and data.
- **“Unknown categories are bad data.”** They may be legitimate evolution; the
  contract determines the response.
- **“A pipeline prevents every leak.”** It cannot fix post-outcome fields or a
  split that violates deployment.
- **“Serialization guarantees parity.”** Provenance, schema, version, and golden
  behavior must also match.
- **“Monitoring shift means automatic retraining.”** Diagnose collection and
  product change before acting.

## Knowledge check

1. Which preprocessing states must be fitted inside each validation fold?
2. Why can an all-missing training feature not yield a defensible median?
3. How can target encoding leak each row's target?
4. What does a golden train-serving fixture prove and not prove?
5. Why must feature names and order be versioned?
6. Which security risk applies to untrusted serialized artifacts?

Score one point per precise answer with an example. Below five requires a new
leakage fixture and corrected test before advancing.

## Completion criteria

- [ ] No learned state is fitted before a split or outside a fold.
- [ ] All eight lab tests pass and expected failures are explained.
- [ ] Unknown, missing, constant, and non-finite values have explicit policies.
- [ ] Feature names, ordering, units, and availability are tested.
- [ ] Training and serving use the same verified artifact and golden fixture.
- [ ] Serialization records provenance and rejects incompatible state.
- [ ] You complete four exercises and score at least 5/6 here.

## Summary and glossary additions

Preprocessing is learned model state. Split first, fit only on training data,
freeze transformation behavior, preserve feature semantics, reuse one artifact,
and monitor its boundary. Correct syntax cannot rescue leaked state or skewed
serving logic.

- **Fit:** estimate transformer state from permitted observations.
- **Transform:** apply frozen state without learning from the input batch.
- **Train-serving skew:** difference between development and deployed feature
  computation.
- **Golden fixture:** versioned input/output case used to compare boundaries.

## Authoritative further reading

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn pipeline documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)

The fit/transform and leakage principles are stable; estimator defaults, feature-
name behavior, sparse output, and persistence compatibility are version-sensitive.

Continue with [model families](../03-model-families/README.md).
