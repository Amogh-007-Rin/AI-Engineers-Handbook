---
title: AI engineering professional practice
slug: ai-professional-practice
level: advanced
stage: career
estimated_hours: 12
prerequisites:
  - paper-reproduction-project
learning_objectives:
  - Convert ambiguous needs into testable AI system requirements
  - Record architecture tradeoffs and communicate uncertainty
  - Review code experiments incidents and product outcomes constructively
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# AI engineering professional practice

Begin discovery with the decision, user, workflow, current baseline, harm of errors, feedback delay, constraints, and non-ML alternatives. Write measurable acceptance criteria before proposing models. Estimate data, compute, latency, cost, maintenance, and organizational dependencies.

Architecture decisions record context, options, decision, evidence, consequences, and revisit triggers. Communicate estimates as ranges with assumptions. Separate observation, inference, recommendation, and uncertainty. Escalate material safety or integrity concerns with concrete evidence.

Code review protects correctness and shared ownership; experiment review protects causal claims; incident review improves systems without hiding accountability. Strong portfolios show problem framing, decisions, failed approaches, tests, results, operation, risks, and reflection—not only polished demos.

## Exercise

Write a one-page proposal and architecture decision for replacing a manual triage workflow. Include non-ML baseline, stakeholders, error costs, data availability, evaluation, human override, rollout, monitoring, rollback, cost range, and conditions that would cancel the project. Conduct a peer review using evidence-based comments.

## Completion criteria

- [ ] Requirements are observable and trace to stakeholder needs.
- [ ] Alternatives include a non-ML option.
- [ ] Uncertainty and cancellation conditions are explicit.
- [ ] Review feedback produces recorded changes or reasoned rejection.
