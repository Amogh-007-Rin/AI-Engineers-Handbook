---
title: Python foundations for AI engineering
slug: python-foundations
level: foundation
stage: foundations
estimated_hours: 12
prerequisites:
  - handbook-orientation
learning_objectives:
  - Write small typed Python programs from a problem statement
  - Diagnose failures using tracebacks tests and controlled experiments
  - Separate pure transformations from input and output operations
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Python foundations for AI engineering

Python is the coordination language of most AI systems. Fluency means reasoning about values, state, interfaces, failure, and verification—not merely recalling syntax.

## Core mental model

A program transforms inputs into outputs. Names refer to objects, types constrain meaningful operations, and control flow determines which transformations run. Prefer small functions with explicit inputs and outputs.

```python
from collections.abc import Iterable


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("mean requires at least one value")
    return sum(items) / len(items)
```

This interface handles a boundary condition, avoids hidden state, and is testable. Type hints communicate intent but do not replace validation or tests.

## Required practice

Work through values, collections, branching, iteration, functions, modules, files, exceptions, classes, iterators, comprehensions, typing, virtual environments, debugging, and testing. For every topic, write a transformation and a failure case.

## Debugging protocol

1. Reproduce the smallest failure.
2. Read the final traceback line, then walk upward to your code.
3. State expected and observed behavior.
4. Inspect the boundary where values diverge.
5. Change one thing, rerun the smallest test, then the full suite.

## Project: dataset summary CLI

Build a command-line program that reads a CSV numeric column and emits count, missing count, minimum, maximum, and mean as JSON. Accept path and column arguments; reject missing files, columns, empty valid data, and nonnumeric values clearly. Separate I/O, parsing, calculation, and presentation. Test normal and failure cases. Do not use Pandas yet.

## Completion criteria

- [ ] The program satisfies the brief and passes tests.
- [ ] Functions have types and focused responsibilities.
- [ ] You can explain every exception boundary.
- [ ] Another person can run it from your README.

Next, learn [linear algebra](../../mathematics/01-linear-algebra/README.md) and use the [NumPy academy](../../../Libraries/NumPy/README.md).
