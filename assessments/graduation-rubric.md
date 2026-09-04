# Graduation and capstone rubric

Score each category from 0 to its maximum. Passing requires **80/100**, at least
60% in every category, and no critical failure. Record the exact artifact or
test supporting every non-zero score.

| Category | Points | Full-credit evidence |
|---|---:|---|
| Problem and product reasoning | 10 | Real stakeholder, decision, constraints, non-ML baseline, measurable success and stop criteria |
| Data practice | 15 | Provenance, license, validation, splits, leakage controls, representation analysis, reproducible pipeline |
| Modeling and evaluation | 15 | Justified baselines, controlled comparisons, uncertainty, slices, failure taxonomy, limitations |
| Software engineering | 15 | Modular typed code, tests, documentation, dependency control, CI, maintainable interfaces |
| Operations and reliability | 15 | Deployment evidence, observability, SLOs, load behavior, rollback, runbook, incident exercise |
| Security and responsible practice | 15 | Threat/impact models, privacy, abuse controls, human oversight, accessibility, residual risk |
| Research and reproducibility | 10 | Traceable claims, seeds, configurations, immutable results, negative findings, clean reproduction |
| Communication | 5 | Clear report, decision records, honest demo, audience-appropriate explanation and defense |

## Critical failures

Any one of these prevents a pass until remediated:

- fabricated, selectively altered, or untraceable evidence;
- data leakage that invalidates the central result;
- use of data, code, models, or media without compatible permission;
- committed secrets, personal data exposure, or uncontrolled high-impact action;
- inability to reproduce the main result from the frozen commit;
- no tested rollback or containment path for a deployed system; or
- failure to disclose a known material limitation or conflict.

## Defense prompts

Reviewers select at least one prompt from each group:

- **Choice:** What simpler alternative did you reject, and what evidence would
  make you reverse that decision?
- **Validity:** Which assumption is most likely to invalidate the result? Show
  the test or analysis that probes it.
- **Failure:** Trigger a malformed input, dependency failure, or degraded-model
  condition. Demonstrate containment and recovery.
- **Tradeoff:** Improve latency, cost, fairness, or recall while accepting a
  documented loss elsewhere. Explain the decision boundary.
- **Reproduction:** Recreate one reported result from a clean checkout and
  reconcile any deviation.

## Decision record

Record candidate name, repository and commit, environment, reviewer roles,
category scores, critical-failure status, commands executed, findings,
remediation deadline, and final decision. Reviewers must declare conflicts and
must not approve work they substantially authored.
