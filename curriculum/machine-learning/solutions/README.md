# Classical machine-learning solution guidance

Attempt the exercises and project before using this guidance. These are review criteria and diagnostic hints, not a single prescribed notebook.

## Evaluation protocol

A strong response defines an entity-level prediction at a specific time, lists only features available by that time, and chooses a grouped or temporal split matching deployment. It reports a constant and rule baseline before a learned candidate. The final test set is evaluated once after choices are fixed.

Check leakage with executable assertions: entity groups are disjoint; feature timestamps do not exceed prediction time; preprocessing is fitted within each training fold; join cardinality is declared; target-derived fields are absent.

## Debugging assessment

The corrected order is: split → fit preprocessing on training → transform training/validation → train → select using validation only → lock decisions → evaluate the untouched test set. Put preprocessing and estimation in one pipeline so cross-validation refits both. Validate joins before feature construction and add metrics aligned with error costs plus slice results.

## Project review

Run:

```bash
cd curriculum/machine-learning/project
python3 -m unittest -v test_baselines.py
```

The supplied majority classifier is deliberately simple. A valid submission preserves this baseline, adds a transparent rule and a learned model, and evaluates all three under identical splits. Reviewers should reject claims based only on the best aggregate score, unexplained threshold choice, or a confidence interval that ignores repeated entities.

## Remediation prompts

- If performance collapses under a group split, identify what the random split allowed the model to memorize.
- If accuracy is high but recall is unacceptable, quantify threshold tradeoffs rather than switching metrics after the fact.
- If importance changes across correlated features, test grouped permutation or remove redundancy and state the limitation.
- If reruns differ, isolate randomness in splitting, training, parallelism, and environment versions.
