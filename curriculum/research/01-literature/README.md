---
title: Reading evaluating and synthesizing AI research
slug: research-literature
level: advanced
stage: research
estimated_hours: 12
prerequisites:
  - responsible-ai-governance
learning_objectives:
  - Trace research claims to methods results and assumptions
  - Compare papers using a structured evidence table
  - Identify unsupported generalization confounding and missing baselines
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Reading, evaluating, and synthesizing AI research

Read a paper in passes. First identify the question, claimed contribution, setting, and evidence. Then inspect data, splits, baselines, compute, metrics, uncertainty, ablations, and limitations. Finally trace critical equations and implementation details. A result supports only the population, task, metric, and experimental conditions actually tested.

Create an evidence table for related work: claim, method, dataset, comparison, effect, uncertainty, compute, artifacts, and threats to validity. Citation count and benchmark rank are not evidence quality. Look for data contamination, tuning-budget asymmetry, weak baselines, multiple comparisons, selective reporting, unavailable artifacts, and conclusions broader than experiments.

## Exercise

Select three primary papers addressing one question. Write the question before reading results. Extract their claims and evidence into a common table, reproduce one key calculation, list conflicting assumptions, and propose an experiment that distinguishes the strongest competing explanations.

## Completion criteria

- [ ] Every summary separates author claim from your inference.
- [ ] Comparisons account for data, compute, tuning, and metric differences.
- [ ] Missing evidence and negative results remain visible.
- [ ] The proposed experiment could falsify a preferred explanation.
