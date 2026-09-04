---
title: Offline recommendation ranking project
slug: recommender-ranking-project
level: practitioner
stage: recommender-systems
estimated_hours: 12
prerequisites:
  - recommender-systems-foundations
learning_objectives:
  - Compute ranking metrics at explicit cutoffs with deterministic tie handling
  - Compare a candidate ranker with popularity and non-personalized baselines
  - Audit user coverage cold start and recommendation concentration
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Offline recommendation ranking project

Implement ranking metrics for held-out relevant items and compare a candidate
ranking with a popularity baseline. Run `python3 -m unittest -v`. Extend the
fixture with temporal events, cold-start users/items, coverage, diversity,
popularity concentration, and a product decision with limitations.
