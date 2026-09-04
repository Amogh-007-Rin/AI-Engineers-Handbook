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
  - Diagnose a local environment without changing it blindly
  - Use AI assistance while retaining authorship and understanding
formats:
  - lesson
  - exercise
compute: cpu
status: published
last_verified: 2026-09-03
---

# How to learn with the handbook

AI engineering is learned by making predictions, building things, observing failures, and correcting your mental model. Reading without retrieval and practice creates familiarity, not reliable skill.

This lesson takes about two hours on a CPU-only computer. It needs Python 3.10
or newer, a terminal, Git, and this repository; it needs no network, account,
API key, GPU, or paid service. Commands are written as text as well as code, so
color is never the only carrier of meaning.

## Outcomes and proof

By the end, you will have a calendar-ready weekly plan, a machine-readable
environment diagnostic, a passing lab test run, and a learning-log entry that
records one failed prediction and correction. These artifacts are stronger
evidence than elapsed study time because another person can inspect them.

## The learning loop

1. Write what you expect to learn and already believe.
2. Study the explanation, then reconstruct it from memory.
3. Implement the smallest working example.
4. Change one assumption and predict the result before running it.
5. Complete the exercise without looking at a solution.
6. Explain the result, failure, and tradeoff in plain language.
7. Revisit difficult material after one day and one week.

The loop separates *recognition* (“that explanation looks familiar”) from
*retrieval* (“I can reconstruct and use it without the page”). It also inserts
prediction before execution. If the result surprises you, the difference
between prediction and observation points directly at a weak mental model.

### Worked example: turn a vague goal into evidence

“Learn Python this week” has no observable finish line. Rewrite it as:

> By Sunday, implement a function that profiles two numeric columns, write one
> normal and two failure tests, run them from a clean checkout, and explain one
> traceback in my learning log.

The revised goal names an artifact, boundary behavior, command, and explanation.
It can still be too large. Time-box a first slice—one column, one test, thirty
minutes—and use the evidence to re-estimate instead of interpreting a poor
estimate as personal failure.

## Deliberate practice and review

Mix four modes during a week:

- **Learn:** form a mental model from an explanation and primary source.
- **Recall:** close the material and reconstruct definitions, steps, or code.
- **Build:** produce a working artifact under realistic constraints.
- **Review:** diagnose mistakes, compare alternatives, and schedule remediation.

A useful 10-hour week might contain two 60-minute learning blocks, three
90-minute implementation blocks, a two-hour project block, and a 90-minute
review. The exact schedule is personal; protecting recall, building, and review
is not. Shorter schedules should narrow scope rather than eliminate feedback.

Spaced review is not rereading on a timer. At roughly one day and one week,
attempt a fresh explanation or changed problem before consulting notes. Record
which cue restored the knowledge. Interleave adjacent ideas only after each has
a basic mental model; otherwise switching creates noise rather than useful
discrimination.

## Evidence over activity

Keep a learning log with the date, objective, artifact, test result, mistake, correction, and next question. “Studied NumPy” is activity. “Predicted three shape failures and passed the exercise tests” is evidence.

Use this compact record:

```text
Date / commit:
Question and prediction:
Artifact and exact command:
Observed result:
Mistake or uncertainty:
Correction and supporting source:
Next retrieval date:
```

Never place credentials, private data, proprietary code, or copied restricted
material in a learning log. Store only the names of required secret variables.

## Using AI assistants

Use assistants to challenge explanations, generate test cases, or review work. Do not submit code you cannot explain. State expected behavior, test suggestions, inspect failures, and summarize why the final version works.

Treat generated output as an untrusted proposal. Verify technical claims against
the installed behavior, tests, versioned official documentation, or primary
research. Do not paste confidential data or credentials. Before retaining a
suggestion, answer three questions: What assumption does it make? What test
could falsify it? Can I explain and modify every line? Record material assistance
according to the rules of the project or course where you submit the work.

## Debugging and asking for help

First preserve the failure. Record the exact command, complete error, versions,
expected behavior, observed behavior, and smallest reproducible input. Read the
last traceback line, then locate the earliest frame in code you control. Change
one variable and rerun the smallest relevant check.

A useful help request contains context, attempted diagnosis, and a focused
question. “It does not work” forces another person to reconstruct your state.
“On Python 3.12, this five-line input raises `ValueError`; I expected an empty
result because the contract says X. The boundary test passes for one row but
fails for zero. Which assumption should I inspect?” supports collaboration.

## Runnable lab: plan and environment diagnostic

From the repository root, run:

```bash
python3 curriculum/foundations/00-orientation/lab/study_plan.py \
  --repository . --target-minutes 180 \
  --block Tuesday 60 learn \
  --block Thursday 60 practice \
  --block Saturday 60 review
python3 -m unittest discover -s curriculum/foundations/00-orientation/lab -v
```

Expected JSON includes `"ready": true`, `"scheduled_minutes": 180`, and
`"target_met": true`; four tests should pass. The diagnostic only observes the
runtime and required files. It intentionally does not install packages or alter
your machine.

Now predict, then trigger, these failures:

1. Set a block to 5 minutes. Why does the validator reject it?
2. Set the target above scheduled minutes. Which exit status is returned, and
   why is an unmet plan different from malformed input?
3. Point `--repository` at another directory. Which evidence makes `ready`
   false?

On shells where backslash continuation differs, enter the first command on one
line. A hosted notebook can call the pure functions directly, but terminal
practice is part of the lesson outcome.

## Practice

Create a personal `learning-log.md` outside commits if it contains personal
information. Record your 12–18 month objective, four-week checkpoint, weekly
study blocks, expected evidence, recovery plan for missed work, and rule for
seeking help. Then complete these exercises:

1. **Recall:** explain recognition, retrieval, and evidence without reopening
   this page.
2. **Implementation:** add an optional `location` field to `StudyBlock`; update
   serialization and tests without weakening validation.
3. **Analysis:** compare a single ten-hour block with five two-hour blocks. Name
   workload or accessibility circumstances where each may be reasonable.
4. **Extension:** design a reminder that respects privacy and supports missed-
   block recovery without punishing streak loss.

Finally run the repository checks:

```bash
python3 scripts/validate_content.py
python3 -m unittest discover -s tests
```

## Common misconceptions and recovery

- **“More hours always means more learning.”** Evidence quality, feedback, rest,
  and retrieval matter. Reduce scope and improve the loop before adding hours.
- **“A passing test proves correctness.”** It proves only asserted behavior in
  that environment. Inspect untested assumptions and meaningful failure modes.
- **“Experts do not need notes or documentation.”** External records reduce
  memory load and make decisions reviewable; expertise improves what is recorded.
- **“Asking for help means I failed.”** An evidence-rich question is an
  engineering skill. Escalate after a bounded diagnostic attempt, not after
  hours of random changes.
- **“The assistant sounded certain.”** Confidence is not provenance. Reproduce
  the behavior and consult an authoritative source.

## Knowledge check

Answer without notes, then verify against the lesson:

1. Why is prediction-before-execution useful?
2. What makes a learning goal independently inspectable?
3. Which facts belong in a reproducible help request?
4. What can a fixed seed or passing unit test *not* establish?
5. Name two privacy or authorship boundaries for AI-assisted work.

If you cannot answer four of five with a concrete example, repeat the lab with
a changed target and write the missing explanation in your log.

## Completion criteria

- [ ] Your schedule can be placed on a calendar.
- [ ] The first checkpoint produces an inspectable artifact.
- [ ] The diagnostic reports a ready environment, and all four lab tests pass.
- [ ] You triggered and explained at least two expected failures.
- [ ] You ran both repository checks and can explain their output.
- [ ] You recorded one uncertainty rather than hiding it.
- [ ] You scored at least 4/5 on the knowledge check after closing the lesson.

## Summary and glossary additions

Effective study cycles through learning, retrieval, implementation, evaluation,
and reflection. Evidence names an artifact, command, result, and limitation.
Debugging preserves and reduces failures before changing them. Responsible AI
assistance accelerates feedback without transferring authorship or trust.

- **Deliberate practice:** focused work on a defined weakness with feedback.
- **Retrieval practice:** reconstructing knowledge before consulting the source.
- **Reproducibility:** recreating a result from declared code, inputs,
  environment, configuration, and procedure within a stated tolerance.

## Further reading

- [Python virtual environments](https://docs.python.org/3/tutorial/venv.html)
- [Python `unittest` documentation](https://docs.python.org/3/library/unittest.html)
- [Git documentation](https://git-scm.com/doc)

These are primary tool references, not proof of a universal study method. The
learning recommendations here are practical defaults; adapt them based on
measured progress, accessibility needs, and qualified support.

Continue to [Python foundations](../01-python-foundations/README.md).
