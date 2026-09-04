---
title: Machine learning problem framing and evaluation
slug: ml-framing-evaluation
level: foundation
stage: machine-learning
estimated_hours: 10
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Translate a product question into an evaluable ML task
  - Design leakage-safe splits baselines and metrics
  - Perform slice-based error analysis and communicate uncertainty
  - Select probability thresholds from explicit costs and constraints
  - Separate model evidence from product and causal claims
formats:
  - lesson
  - project
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Machine-learning problem framing and evaluation

Evaluation asks whether an intervention supported by a model improves a
specified decision under realistic conditions. It is not a ritual performed
after training. A perfectly computed metric can answer the wrong question,
reward leakage, hide harmful slices, or ignore operational constraints.

This ten-hour lesson runs offline with Python 3.10 or newer and no third-party
packages. The lab completes in seconds on a CPU with negligible memory. Tables
use labels as well as values, and color is not the only carrier of meaning;
structured output is readable as plain text and by screen readers.

## Outcomes and evidence

You will write a decision brief, build a baseline ladder, choose a deployment-
faithful split, compute and interpret classification and probability metrics,
select thresholds from costs, quantify sampling uncertainty, and inspect slices.
Completion requires the eight-test lab, exercises, the
[baseline evaluation project](../project/README.md), and 80/100 on the
[classical-ML assessment](../assessment.md).

## Frame the decision before the model

Write these fields before choosing an algorithm:

- **Decision and actor:** what action changes, who takes it, and what happens
  without a model?
- **Prediction unit:** one person, transaction, document, session, or event.
- **Observation and prediction time:** latest legitimate information and when
  output becomes available.
- **Target and horizon:** exact outcome, measurement process, label delay, and
  time window.
- **Actionability:** how different outputs alter behavior or resource allocation.
- **Errors and abstention:** consequences of false positives, false negatives,
  delayed decisions, and no decision.
- **Constraints:** capacity, latency, privacy, fairness, security, cost, and
  human-review limits.
- **Success and stop criteria:** minimum improvement and unacceptable outcomes.

If labels do not exist at the correct time, the action cannot use predictions,
or a simple rule already meets the need, more modeling is not the answer.

### Worked example: review delayed payments

A team can manually review 100 invoices per day to prevent delayed payment.
The prediction unit is an invoice at issue time; the target is payment more than
30 days late; features must exist when issued. The action is prioritized review.
A false positive consumes review capacity, while a false negative misses a
preventable delay. A constant prevalence estimate, oldest-account-first rule,
and simple learned score form a useful baseline ladder.

Accuracy is a poor primary metric if only 5% are delayed: predicting “on time”
always yields 95%. At capacity 100, precision at the selected cutoff estimates
review yield, recall estimates recovered delayed cases, and a cost table compares
operational consequences. Evaluation also needs temporal splits because future
payment behavior should not influence earlier model selection.

## Splits simulate deployment

A split is a claim about the future population. Choose it from the data-generating
and deployment process:

- random or stratified splits may suit independent, exchangeable observations;
- grouped splits keep every person, device, organization, or source together;
- temporal splits train on the past and evaluate later periods;
- geographic or source holdouts test transfer to a new context; and
- nested cross-validation separates tuning from performance estimation when
  data are limited.

Near-duplicates, repeated entities, post-outcome features, target encodings,
global preprocessing, and test-guided feature selection can leak information.
Freeze the final test set before model decisions. Repeatedly checking it turns
it into another validation set, even if no gradient is computed on it.

Stratification preserves observed class proportion but does not repair group or
time leakage. Report label prevalence, entity overlap, periods, exclusions, and
sample counts for every split.

## Baselines make complexity accountable

Use a ladder rather than one weak comparator:

1. current non-model process;
2. constant majority, mean, or prevalence predictor;
3. transparent domain rule;
4. simple learned model; and
5. more complex candidate only if evidence justifies it.

A baseline catches label inversions, broken metrics, implausible joins, and weak
problem formulation. Compare every candidate on the same eligible observations,
splits, metrics, and uncertainty procedure. Added complexity must earn its
training cost, latency, operational burden, security surface, and maintenance.

## Confusion counts and classification metrics

For a declared positive class:

```text
                  predicted +   predicted -
actual +               TP             FN
actual -               FP             TN
```

- precision = `TP / (TP + FP)`: yield among positive predictions;
- recall = `TP / (TP + FN)`: coverage of actual positives;
- specificity = `TN / (TN + FP)`: rejection of actual negatives;
- accuracy = `(TP + TN) / N`; and
- F1 is the harmonic mean of precision and recall.

Undefined denominators need an explicit policy. Returning zero can be practical
for a report, but it must not imply observed evidence. Always publish counts,
positive class, threshold, prevalence, and denominator with ratios.

Ranking metrics such as ROC AUC or average precision summarize many thresholds.
They do not choose an operating point, incorporate capacity automatically, or
guarantee calibrated probabilities. With rare positives, inspect precision-
recall behavior and the actual decision region.

## Probabilities, calibration, and thresholds

A calibrated prediction of `0.7` means that among comparable predictions near
`0.7`, approximately 70% become positive under the measured conditions. Brier
score averages squared probability error; log loss penalizes confident errors
more strongly. Calibration bins are diagnostic but depend on binning and sample
size.

A score threshold converts ranking or probability into an action. Select it on
validation evidence using declared costs, capacity, or utility—not the test set.
For costs `C_FP` and `C_FN`, empirical decision cost is
`C_FP × FP + C_FN × FN`, optionally including review and abstention. Costs can
vary across people and time, and ethical constraints may rule out the nominal
minimum-cost choice.

Prevalence shift can alter precision and calibration even when conditional model
behavior is unchanged. Revalidate thresholds in the intended population.

## Uncertainty and repeated experiments

A point estimate changes with the sampled observations, split, seed, and label
noise. Report uncertainty appropriate to the sampling unit. If a person has many
rows, resampling rows independently understates dependence; bootstrap groups or
use a method matching the design.

A percentile bootstrap repeatedly samples evaluation units with replacement,
recomputes a metric, and reports empirical quantiles. It describes variability
under its resampling assumptions; it is not a guarantee about deployment shift.
State repetitions, seed, unit, interval method, and sample size.

For stochastic training, run multiple seeds and distinguish training variability
from evaluation-sample uncertainty. Do not select the luckiest seed.

## Slice and error analysis

Aggregate performance can hide a failed region. Choose slices from product
risks and data generation—time, source, device, language, geography, missingness,
or relevant groups—not from whichever post-hoc cut looks dramatic. For every
slice publish support, prevalence, confusion counts, metric, and uncertainty.

Small slices may not support a stable conclusion; report that limitation rather
than suppressing them. Sensitive-attribute analysis needs lawful, ethical data
handling and stakeholder context. Equal metric values do not alone establish
fairness.

Inspect representative false positives, false negatives, high-confidence errors,
and abstentions without exposing personal data. Build a failure taxonomy, count
categories, and connect each category to a possible data, model, product, or
policy change. Qualitative examples complement metrics; they do not replace them.

## Regression and other task families

Metric choice follows the decision. MAE expresses typical absolute error in
target units; squared error emphasizes larger misses; quantile loss supports
asymmetric decisions; interval coverage requires interval width and calibration.
Ranking, retrieval, forecasting, generation, and reinforcement learning require
task-specific protocols. Never transfer a classification metric because it is
familiar.

## Runnable lab: threshold and evaluation report

The dependency-free lab validates binary labels and probabilities, computes
confusion metrics and Brier score, compares thresholds by cost, performs a seeded
bootstrap, and reports named slices:

```bash
python3 curriculum/machine-learning/01-evaluation/lab/evaluation.py
python3 -m unittest discover -s curriculum/machine-learning/01-evaluation/lab -v
```

Expected JSON selects threshold `0.6` for the example cost table and includes
overall and `new`/`returning` slice support. Eight tests pass. Read the
[lab guide](lab/README.md) and predict the threshold before running it.

## Failure practice and debugging

Trigger empty data, unequal lengths, a label outside `{0,1}`, probability below
zero or above one, `NaN`, an invalid threshold, a one-class slice, and a slice-
label length mismatch. Then intentionally choose the threshold on test labels
and explain why correct code produces invalid evidence.

When metrics disagree, return to the confusion counts and decision cost. When a
score seems unexpectedly strong, compare the baseline, inspect splits and feature
availability, and search for duplicates or leakage before tuning.

## Exercises

1. **Recall:** define unit, target, horizon, baseline, validation set, test set,
   calibration, threshold, and slice with the invoice example.
2. **Implementation:** add negative predictive value and balanced accuracy to
   the lab, including undefined-denominator tests.
3. **Threshold analysis:** choose three different false-negative costs and show
   how the selected threshold and workload change.
4. **Split design:** design protocols for repeated patients, future demand, new
   stores, and independent manufactured parts. Identify leakage tests.
5. **Uncertainty:** modify the bootstrap to resample entity groups and compare
   interval width with row-level resampling.
6. **Error analysis:** create a five-category failure taxonomy and define one
   falsifiable follow-up experiment for the largest category.

## Common misconceptions

- **“Accuracy summarizes model quality.”** It ignores error asymmetry and can be
  dominated by the majority class.
- **“Stratification prevents leakage.”** It preserves label ratios, not entity,
  time, preprocessing, or target isolation.
- **“AUC chooses the threshold.”** It summarizes ranking across thresholds.
- **“Calibration means correctness.”** Calibrated predictions can have poor
  discrimination or encode an invalid target.
- **“A confidence interval covers every future shift.”** It reflects a stated
  sampling model, not arbitrary deployment change.
- **“Equal aggregate metrics prove fairness.”** Impact, access, uncertainty, and
  context remain necessary.

## Knowledge check

1. Why is the prediction time part of the feature contract?
2. When is a grouped split preferable to stratification?
3. What do precision and recall condition on?
4. Why can a good ranking model have poor probabilities?
5. What information must accompany a slice metric?
6. Why does repeated test-set inspection invalidate its role?
7. What assumptions does a row-level bootstrap make?

Score one point per precise answer with an example. Below six requires a new
decision brief, changed split, and rerun before the project.

## Completion criteria

- [ ] The target and inputs exist at their claimed times.
- [ ] The split mirrors deployment and blocks documented leakage paths.
- [ ] All eight lab tests pass and expected failures are explained.
- [ ] Metrics follow costs and include counts, uncertainty, and slices.
- [ ] Threshold selection uses validation data, not the final test set.
- [ ] Complexity is compared with non-ML, constant, rule, and learned baselines.
- [ ] You complete four exercises and score at least 6/7 here.
- [ ] The project earns at least 80/100 without an automatic failure.

## Summary and glossary additions

Evaluation begins with a decision and temporal contract. Deployment-faithful
splits, credible baselines, confusion counts, calibrated probabilities, explicit
threshold costs, uncertainty, and risk-driven slices turn scores into evidence.
No metric repairs an invalid target or leaked protocol.

- **Operating point:** threshold and resulting tradeoff used for action.
- **Calibration:** agreement between predicted probability and observed frequency.
- **Slice:** declared subset evaluated with its support and uncertainty.
- **Test set:** held-out sample used after model choices are fixed.

## Authoritative further reading

- [scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [CONSORT-AI reporting extension](https://www.nature.com/articles/s41591-020-1034-x)

Metric definitions are stable; library averaging defaults, calibration tools,
splitters, and threshold APIs are version-sensitive. Record exact versions and
verify the installed behavior.

Continue with [leakage-safe preprocessing](../02-preprocessing/README.md) and
the [Scikit-Learn academy](../../../Libraries/Scikit-Learn/README.md).
