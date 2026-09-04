---
title: How to learn with the handbook
slug: handbook-orientation
level: foundation
stage: foundations
estimated_hours: 2
prerequisites:
learning_objectives:
  - Create a realistic weekly learning and review routine
  - Verify work with tests evidence and written reflection
formats:
  - lesson
  - exercise
compute: cpu
status: published
last_verified: 2026-09-03
---

# How to learn with the handbook

AI engineering is learned by making predictions, building things, observing failures, and correcting your mental model. Reading without retrieval and practice creates familiarity, not reliable skill.

## The learning loop

1. Write what you expect to learn and already believe.
2. Study the explanation, then reconstruct it from memory.
3. Implement the smallest working example.
4. Change one assumption and predict the result before running it.
5. Complete the exercise without looking at a solution.
6. Explain the result, failure, and tradeoff in plain language.
7. Revisit difficult material after one day and one week.

## Evidence over activity

Keep a learning log with the date, objective, artifact, test result, mistake, correction, and next question. “Studied NumPy” is activity. “Predicted three shape failures and passed the exercise tests” is evidence.

## Using AI assistants

Use assistants to challenge explanations, generate test cases, or review work. Do not submit code you cannot explain. State expected behavior, test suggestions, inspect failures, and summarize why the final version works.

## Practice

Create a personal `learning-log.md` and record your 12–18 month objective, four-week checkpoint, weekly study blocks, expected evidence, recovery plan for missed work, and rule for seeking help. Then run:

```bash
python3 scripts/validate_content.py
python3 -m unittest discover -s tests
```

## Completion criteria

- [ ] Your schedule can be placed on a calendar.
- [ ] The first checkpoint produces an inspectable artifact.
- [ ] You ran both checks and can explain their output.
- [ ] You recorded one uncertainty rather than hiding it.

Continue to [Python foundations](../01-python-foundations/README.md).
