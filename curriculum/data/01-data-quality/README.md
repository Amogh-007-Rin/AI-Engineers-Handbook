---
title: Data quality and reproducible analysis
slug: data-quality-foundations
level: foundation
stage: data
estimated_hours: 8
prerequisites:
  - python-foundations
learning_objectives:
  - Define and test a dataset contract before analysis
  - Identify missingness duplication leakage and invalid values
  - Produce a reproducible data quality report
  - Verify split isolation join cardinality and temporal availability
  - Document provenance privacy licensing and monitoring decisions
formats:
  - lesson
  - project
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Data quality and reproducible analysis

A model learns patterns in recorded measurements, not reality itself. Every row
is produced by collection, selection, measurement, annotation, transformation,
and storage decisions. Data quality is therefore fitness for a declared decision,
not a universal cleanliness score.

This eight-hour lesson uses Python 3.10 or newer, no network, and generated
fixtures. The lab runs on a CPU in seconds with negligible memory. Tables and
issues are represented in text and structured JSON; color is not the only
carrier of meaning, and issue messages remain readable by screen readers.

## Outcomes and evidence

You will write an executable contract, distinguish absence from invalidity,
detect duplicate identities and split leakage, defend join cardinality, and
produce a quality report whose counts reconcile to the source. Completion needs
the eight-test lab, failure analysis, exercises, the independent
[dataset-quality project](../project/README.md), a
[dataset card](../../../templates/dataset-card-template.md), and 80/100 on the
[data assessment](../assessment.md).

## Start with the decision and unit of observation

Before loading a table, state:

- who or what makes a decision from the output;
- what one row represents and whether multiple rows may describe one entity;
- the prediction or analysis time and which information exists then;
- population, period, geography, sampling, exclusions, and intended uses; and
- the cost of false, missing, delayed, or duplicated records.

“Customer data” is not a unit. “One completed support interaction, keyed by
interaction identifier, recorded at closure in UTC” is testable. A dataset may
be excellent for billing reconciliation and invalid for predicting escalation
at ticket creation because closure fields arrive too late.

## Dataset contracts

A contract translates meaning into executable expectations. For each field,
record name, type, nullability, units, allowed values or range, temporal meaning,
sensitivity, and source. Also define primary or composite identity, row grain,
ordering, cross-field invariants, and evolution policy.

Do not silently coerce malformed values until you know why they are malformed.
Mapping the strings `"unknown"`, `"N/A"`, an empty string, and a parsing failure
to the same null destroys distinct evidence. Normalize known sentinels explicitly,
count them, quarantine unexpected forms, and retain lineage.

### Worked example: event contract

Suppose one row represents a scored event:

```text
event_id: non-empty unique string
entity_id: non-empty string
observed_at: integer Unix second, not after prediction_time
score: finite number in [0, 1], nullable only when status = "pending"
status: one of pending, complete
```

The cross-field rule matters: a missing score for a pending event can be expected,
while the same absence for a complete event violates the contract. A generic
“95% complete” metric cannot express this difference.

## Six practical quality dimensions

- **Validity:** values follow type, range, category, format, and cross-field rules.
- **Completeness:** required information exists for the applicable population.
- **Uniqueness:** keys and duplicate semantics match the declared grain.
- **Consistency:** units, encodings, timestamps, relationships, and replicas agree.
- **Timeliness:** values arrive before the decision that consumes them.
- **Representativeness:** sampled records support the intended population and use.

Add accuracy only when there is a trustworthy reference or measurement method.
A syntactically valid age can still be wrong. Quality reports should show
denominators, affected populations, and examples safe to disclose—not only a
single percentage.

## Missingness and measurement

Missing values may be structurally inapplicable, not yet observed, refused,
lost, redacted, or failed during parsing. These mechanisms affect whether
dropping or imputing is defensible. Missingness correlated with group membership,
service access, device behavior, or the outcome can create systematic error.

Measure missingness by field and meaningful slices over time. Fit imputers only
on training data, preserve a missingness indicator when it has legitimate
predictive meaning, and test how conclusions change under plausible alternatives.
Never infer that missing at random is true merely because a library defaults to
an imputation strategy.

## Duplicates, entities, and joins

Byte-identical rows are only one duplicate form. The same entity may appear
under several identifiers; repeated events may be valid; retried ingestion may
replay the same event. Define identity and acceptable multiplicity before using
`drop_duplicates`.

For a join, state expected cardinality: one-to-one, one-to-many, many-to-one, or
deliberate many-to-many. Compare row counts and key coverage before and after.
An accidental many-to-many join multiplies observations and can bias aggregates
while producing no exception.

If a left table has two rows for key `A` and the right also has two, an inner
join produces four `A` rows. The correct response is not automatically to remove
duplicates: determine which table violates its declared grain.

## Leakage and split integrity

Leakage occurs when training or evaluation uses information unavailable or
illegitimate at the intended prediction time. Common forms include:

- target-derived fields, post-outcome timestamps, and manually resolved status;
- fitting preprocessing on all data before splitting;
- the same person, device, document, or near-duplicate across splits;
- future rows influencing an earlier forecast; and
- feature selection or threshold tuning on the final test set.

Build the split around the deployment unit. Use group splits when identities
repeat, temporal splits for future-facing decisions, and geographic or source
holdouts when transfer matters. Hashes can detect exact overlap but not semantic
near-duplicates or shared provenance.

## Drift and data-centric iteration

Input-distribution change, relationship change, and label-policy change are
different problems. Monitor features whose meaning and collection are stable,
with reference windows, thresholds, sample sizes, and response ownership.
Statistical significance alone can flag harmless changes at large scale; small
samples can hide operationally important ones.

When a quality issue appears, preserve raw data, quarantine affected records,
identify upstream cause, quantify downstream impact, and backfill only through
versioned transformations. Do not mutate history invisibly to make a dashboard
green.

## Provenance, privacy, and licensing

Record source, owner, version, retrieval time, checksum, collection and annotation
method, transformations, and consumers. Verify that the license and consent
support the intended use and redistribution. Public accessibility is not the
same as permission or ethical suitability.

Minimize personal and sensitive data, restrict access, define retention and
deletion, and avoid placing raw values in logs or test fixtures. Synthetic data
can test pipeline behavior but does not prove representativeness, fairness, or
privacy safety. Follow the repository's [dataset practice](../../../datasets/README.md).

## Reproducible pipelines

Keep immutable raw input, version transformation code and configuration, pin the
environment, seed stochastic operations, and write outputs to versioned locations.
Each stage should report received, accepted, rejected, quarantined, and emitted
counts that reconcile. Make reruns idempotent or explicitly document replacement
semantics.

Tests should include normal, boundary, malformed, temporal, duplicate, and
cardinality cases. A test fixture is an intentional miniature, not evidence that
production data share its distribution.

## Runnable lab: event-quality contract

The dependency-free lab validates event records, aggregates issue counts,
detects identity overlap between splits, and verifies join cardinality:

```bash
python3 curriculum/data/01-data-quality/lab/data_quality.py
python3 -m unittest discover -s curriculum/data/01-data-quality/lab -v
```

Expected JSON reports four rows, two accepted, two quarantined, and issue kinds
for range, temporal availability, and cross-field completeness. Eight tests
pass. Read the [lab guide](lab/README.md) before extending the contract.

## Failure practice and debugging

Trigger missing and extra fields, duplicate event identity, non-finite score,
out-of-range score, post-prediction timestamp, a complete event without score,
cross-split entity overlap, and a cardinality violation. For each, record whether
the pipeline should reject the whole batch, quarantine a row, warn, or continue.
That policy is a product and risk decision, not merely an exception choice.

When counts disagree, trace them stage by stage. When a model metric is
implausibly high, inspect leakage before celebrating. When a join expands rows,
group keys on both sides and compare multiplicities before changing code.

## Exercises

1. **Recall:** define grain, validity, completeness, uniqueness, timeliness,
   representativeness, lineage, and leakage with one example each.
2. **Implementation:** add an allowed `source` category and an applicable-null
   rule. Write malformed and boundary tests first.
3. **Analysis:** design train/validation/test splits for repeated patients,
   monthly transactions, and randomly assigned experiments; explain differences.
4. **Debugging:** create a many-to-many join fixture whose total doubles while
   key coverage appears unchanged. Add a prevention assertion.
5. **Extension:** compare field distributions between two periods with counts
   and effect sizes; do not reduce the result to a p-value.
6. **Governance:** fill a dataset card including an unsupported use, deletion
   process, unresolved representation risk, and accountable owner.

## Common misconceptions

- **“No nulls means high quality.”** Values can be wrong, leaked, duplicated,
  stale, or unrepresentative.
- **“Remove every duplicate.”** Repeated entities or events may be the true grain.
- **“Random splitting is neutral.”** It can leak identities, time, and context.
- **“Schema validation proves semantic validity.”** Types and ranges cannot prove
  that a measurement represents the intended construct.
- **“Synthetic data solves privacy and bias.”** It can inherit patterns and only
  demonstrates the behavior it was designed to contain.
- **“Drift means retrain.”** First determine what changed, whether performance
  or impact changed, and whether the collection system failed.

## Knowledge check

1. Why must row grain be declared before duplicate handling?
2. How can missingness be valid for one row and invalid for another?
3. Why can a one-to-one join assertion prevent biased evaluation?
4. Name three leakage paths unrelated to directly copying the target.
5. What evidence distinguishes reproducibility from rerunning changed data?
6. Why is a valid schema insufficient for representative data?
7. Which counts should reconcile in a quarantine pipeline?

Score one point per precise answer with an example. Below six requires a new
corrupted fixture, regression test, and written correction before the project.

## Completion criteria

- [ ] All eight lab tests pass from a clean checkout.
- [ ] You trigger and classify the documented quality and leakage failures.
- [ ] Input, accepted, rejected, quarantined, and output counts reconcile.
- [ ] You complete four exercises including join and split analysis.
- [ ] A dataset card records provenance, license, exclusions, privacy, and risks.
- [ ] A reviewer can reproduce the report without modifying raw input.
- [ ] You score at least 6/7 here and 80/100 on the data assessment.

## Summary and glossary additions

Data quality begins with a decision, unit of observation, and executable
contract. Missingness, duplication, joins, temporal availability, splits,
provenance, and representation affect validity. A trustworthy pipeline preserves
raw evidence, reconciles counts, isolates splits, and makes limitations visible.

- **Grain:** what exactly one row represents.
- **Lineage:** traceable path from source through transformations to consumers.
- **Quarantine:** isolated invalid or uncertain records retained for diagnosis.
- **Target leakage:** illegitimate target-related information available during
  development but not the intended prediction.

## Authoritative further reading

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [W3C Data on the Web Best Practices](https://www.w3.org/TR/dwbp/)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Pandas missing-data guide](https://pandas.pydata.org/docs/user_guide/missing_data.html)

The quality principles are stable. Schema APIs, missing-value behavior, storage
formats, and library defaults are version-sensitive; test the pinned environment
and cite exact upstream documentation.

Continue with the [dataset-quality project](../project/README.md) and use the
[Pandas academy](../../../Libraries/Pandas/README.md) for tool-specific workflows.
