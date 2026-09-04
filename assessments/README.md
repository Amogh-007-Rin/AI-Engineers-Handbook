# Assessment system

Assessment is evidence of mastery, not a reading-completion counter. Learners
may revise and resubmit; the standard stays fixed while the path to it remains
supportive.

## Assessment layers

1. **Lesson checks** test recall, interpretation, and misconceptions without
   pretending that short answers prove implementation skill.
2. **Executable exercises** test deterministic behavior, edge cases, and
   debugging. Learners must explain what the tests omit.
3. **Stage projects** test integration and judgment against public rubrics.
4. **Portfolio reviews** test reproducibility and communication across projects.
5. **Capstone defense** tests an independently built production system under
   failure, risk, and operational questioning.

## Stage gates

| Competency | Assessment | Evidence project |
|---|---|---|
| Software foundations | [Gate](../curriculum/foundations/assessment.md) | [Data explorer](../curriculum/foundations/project/README.md) |
| Mathematics | [Gate](../curriculum/mathematics/assessment.md) | [Numerical investigation](../curriculum/mathematics/project/README.md) |
| Data | [Gate](../curriculum/data/assessment.md) | [Quality pipeline](../curriculum/data/project/README.md) |
| Classical ML | [Gate](../curriculum/machine-learning/assessment.md) | [Baseline system](../curriculum/machine-learning/project/README.md) |
| Deep learning | [Gate](../curriculum/deep-learning/assessment.md) | [Gradient implementation](../curriculum/deep-learning/project/README.md) |
| Computer vision | [Gate](../curriculum/computer-vision/assessment.md) | [Vision evaluation](../curriculum/computer-vision/project/README.md) |
| NLP and speech | [Gate](../curriculum/nlp-and-speech/assessment.md) | [Language evaluation](../curriculum/nlp-and-speech/project/README.md) |
| Time series | [Gate](../curriculum/time-series/assessment.md) | [Backtest](../curriculum/time-series/project/README.md) |
| Recommenders | [Gate](../curriculum/recommender-systems/assessment.md) | [Ranking evaluation](../curriculum/recommender-systems/project/README.md) |
| Graph learning | [Gate](../curriculum/graph-learning/assessment.md) | [Leakage-safe split](../curriculum/graph-learning/project/README.md) |
| Reinforcement learning | [Gate](../curriculum/reinforcement-learning/assessment.md) | [Seeded bandit study](../curriculum/reinforcement-learning/project/README.md) |
| Generative AI | [Gate](../curriculum/generative-ai/assessment.md) | [Grounding evaluation](../curriculum/generative-ai/project/README.md) |
| Agents | [Gate](../curriculum/agents/assessment.md) | [Capability boundary](../curriculum/agents/project/README.md) |
| ML systems | [Gate](../curriculum/ml-systems/assessment.md) | [Reliability design](../curriculum/ml-systems/project/README.md) |
| Responsible AI | [Gate](../curriculum/responsible-ai/assessment.md) | [Risk register](../curriculum/responsible-ai/project/README.md) |
| Research | [Gate](../curriculum/research/assessment.md) | [Controlled comparison](../curriculum/research/project/README.md) |
| Professional practice | [Gate](../curriculum/career/assessment.md) | [Portfolio](../curriculum/career/project/README.md) |

## Rules for a valid pass

- Use the stated threshold; where none is stricter, require at least 80/100.
- Require every automatic-fail condition to be absent. A high average cannot
  erase leakage, fabricated evidence, unsafe behavior, or irreproducibility.
- Attach the exact commit, commands, environment, outputs, scorer, date, and
  remediation history.
- Sample oral explanation or a fresh modification to distinguish understanding
  from copied output.
- Permit a new attempt after targeted remediation, using a changed case or seed
  where memorization would invalidate the evidence.
- Accommodations may change presentation, time, or interaction method without
  changing the measured competency.

Use the [graduation rubric](graduation-rubric.md) for portfolio and capstone
decisions. Reviewers should record evidence using the
[review record](../templates/review-record-template.md); self-scores are useful
for learning but are not independent approval.
