# Project ladder

Projects are the handbook's evidence of applied competence. They become less
guided as the learner advances: early projects isolate one engineering habit;
later projects combine data, modeling, evaluation, operations, and risk. A
project is complete only when its tests, written reasoning, and reproducibility
evidence pass the linked rubric.

## Sequence

| Step | Portfolio outcome | Canonical brief |
|---:|---|---|
| 1 | Tested Python data explorer | [Software foundations](../curriculum/foundations/project/README.md) |
| 2 | Reproducible numerical investigation | [Mathematics](../curriculum/mathematics/project/README.md) |
| 3 | Validated data-quality pipeline | [Data foundations](../curriculum/data/project/README.md) |
| 4 | Baseline-driven tabular ML system | [Machine learning](../curriculum/machine-learning/project/README.md) |
| 5 | Neural model and gradient implementation | [Deep learning](../curriculum/deep-learning/project/README.md) |
| 6 | Framework-parity benchmark | [Framework parity lab](../curriculum/deep-learning/04-framework-parity/README.md) |
| 7 | Vision or language application | [Vision](../curriculum/computer-vision/project/README.md) or [NLP](../curriculum/nlp-and-speech/project/README.md) |
| 8 | Structured domain system | [Forecasting](../curriculum/time-series/project/README.md), [ranking](../curriculum/recommender-systems/project/README.md), [graphs](../curriculum/graph-learning/project/README.md), or [RL](../curriculum/reinforcement-learning/project/README.md) |
| 9 | Evaluated grounded generation system | [Generative AI](../curriculum/generative-ai/project/README.md) |
| 10 | Bounded tool-using agent | [Agents](../curriculum/agents/project/README.md) |
| 11 | Reliable model service | [ML systems](../curriculum/ml-systems/project/README.md) |
| 12 | Controlled research reproduction | [Research](../curriculum/research/project/README.md) |
| 13 | Role-specific system | [Specializations](../specializations/README.md) |
| 14 | Production-grade defended system | [Capstone](capstone/README.md) |

The [responsible-AI project](../curriculum/responsible-ai/project/README.md) is
not optional side work. Apply its impact assessment and threat model to steps
4–14. The [career portfolio project](../curriculum/career/project/README.md)
turns the resulting evidence into a reviewable portfolio.

## Evidence package

Keep each project in its own repository or clearly isolated directory. Submit:

- a concise problem statement, stakeholder, non-ML baseline, and explicit
  success and stop criteria;
- an environment lock or pinned requirements plus a one-command clean run;
- provenance, license, checksum, schema, split logic, and limitations for data;
- tests for normal, boundary, malformed, and failure-recovery behavior;
- versioned experiments, baseline comparisons, uncertainty, slice analysis,
  and negative findings—not only the best score;
- a dataset card, model/system card, threat model, cost estimate, and decision
  record proportional to risk;
- a demonstration plus logs or reports that another person can inspect; and
- a retrospective naming what the evidence does not prove.

Do not include secrets, restricted data, large model weights, generated build
artifacts, or claims whose source cannot be traced.

## Review protocol

1. Freeze a candidate commit and record its identifier.
2. A reviewer follows the README in a clean environment without private help.
3. Run the published tests, then add at least one adversarial or boundary test.
4. Score the linked stage assessment and the
   [graduation rubric](../assessments/graduation-rubric.md) where applicable.
5. Record findings, remediation, rerun evidence, and reviewer role with the
   [review template](../templates/review-record-template.md).

Passing a unit test proves only the behaviors asserted by that test. Design
quality, data validity, communication, accessibility, and responsible use need
human review.
