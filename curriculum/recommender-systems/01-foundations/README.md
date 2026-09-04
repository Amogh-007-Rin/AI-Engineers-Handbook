---
title: Recommender systems retrieval ranking evaluation and feedback loops
slug: recommender-systems-foundations
level: practitioner
stage: recommender-systems
estimated_hours: 18
prerequisites:
  - ml-framing-evaluation
learning_objectives:
  - Frame recommendation as candidate generation ranking and policy decisions
  - Build temporal user-aware splits and popularity content and collaborative baselines
  - Evaluate relevance coverage diversity novelty calibration and system cost
  - Govern feedback loops cold start privacy exposure bias and exploration
formats:
  - lesson
  - exercise
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Recommender systems foundations

A recommendation system chooses which items to expose, in what order, under
inventory, latency, policy, and business constraints. Separate candidate
generation from ranking and post-ranking. Define the user/session context,
eligible catalog, prediction timestamp, relevance event, cutoff, and harm of a
bad or missing recommendation.

Split interactions by time and user according to deployment. Random event splits
leak later preferences and repeated items. Fit user/item statistics only on the
training interval. Begin with popular, recent, and content-similarity baselines;
then consider collaborative filtering, two-tower retrieval, learning-to-rank,
or sequential models. Define cold-start behavior for users and items.

Offline metrics include precision/recall/nDCG/MRR at a cutoff, but logged data is
biased by earlier exposure. Also report catalog/user coverage, popularity bias,
diversity, novelty, calibration, latency, and relevant user/item slices. Online
experiments need guardrails, power, duration, interference awareness, and an
ethical exploration policy.

Monitor input freshness, candidate availability, feature drift, score/rank
distributions, coverage, latency, user outcomes, and feedback-loop concentration.
Minimize retained behavioral data, provide controls and deletion, filter unsafe
items, and keep a non-personalized fallback.

## Exercises

1. Define a recommendation event and enumerate exposure/label biases.
2. Implement popularity and recent-item baselines with a temporal cutoff.
3. Compute precision, recall, MRR, and nDCG at two cutoffs by user slice.
4. Measure coverage, diversity, and popularity concentration.
5. Design an online test with guardrails, fallback, and deletion behavior.

## Assessment

Pass at 80/100: 20 framing/split integrity, 20 baselines and ranking metrics, 20
coverage/diversity/slices, 20 feedback/privacy/experimentation, 20 reproducibility
and operations. Exposure leakage or missing fallback is an automatic failure.
