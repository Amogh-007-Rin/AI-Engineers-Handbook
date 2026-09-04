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
  - Use collections iterators exceptions and data classes deliberately
  - Measure before making performance or concurrency changes
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Python foundations for AI engineering

Python is the coordination language of many AI systems. Fluency means reasoning
about values, state, interfaces, failure, and evidence—not memorizing every
built-in function. This lesson needs Python 3.10 or newer, a terminal, and this
repository. The core path is dependency-free, offline, CPU-only, under 50 MB of
RAM, and normally completes in twelve hours including exercises and the project.

## Outcomes and evidence

You will transform untrusted records into typed observations, summarize them,
explain aliasing and iterator behavior, preserve useful exception context, and
write boundary tests. Completion evidence is a passing lab run, the independent
[data explorer project](../project/README.md), an explanation of a deliberately
triggered failure, and at least 80/100 on the
[foundation assessment](../assessment.md).

## Names, objects, and types

A Python name refers to an object. Assignment binds a name; it does not
necessarily copy the object. This matters whenever an object is mutable:

```python
training = {"features": ["age", "income"]}
alias = training
alias["features"].append("region")
assert training["features"] == ["age", "income", "region"]
```

Both names reach the same dictionary and nested list. Use an explicit shallow
or deep copy only after deciding which levels should be independent. Blindly
copying large arrays can create a performance problem; accidentally sharing
them can create a correctness problem.

Integers, floats, strings, and tuples are immutable. Lists, dictionaries, and
sets are mutable. Mutability is not bad; hidden or uncontrolled mutation is
difficult to test. Prefer functions whose returned value makes change visible.

Types define supported operations, not business validity. A probability can be
a valid `float` object while still being invalid at `1.4`. Type hints help tools
and readers understand interfaces, but runtime validation protects boundaries.

## Collections and control flow

Choose a collection by the question it answers:

- a `list` preserves order and duplicates;
- a `tuple` represents a fixed ordered record or immutable sequence;
- a `dict` maps unique hashable keys to values; and
- a `set` provides membership and uniqueness without semantic position.

Prefer direct iteration to manual indices:

```python
scores = [0.72, 0.91, 0.44]
accepted = [score for score in scores if score >= 0.70]
for rank, score in enumerate(sorted(accepted, reverse=True), start=1):
    print(rank, score)
```

Comprehensions suit one readable transformation. Use a normal loop when
validation, several branches, logging, or multiple outputs are involved.
Clever density is not an engineering goal.

Truthiness lets empty collections, zero, and `None` participate in conditions,
but those states can mean different things. `if not value` collapses them. Use
`value is None` when absence differs from zero or an empty result.

## Functions and contracts

A focused function has explicit inputs, one coherent responsibility, a useful
return value, and documented failure behavior:

```python
from collections.abc import Iterable


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("mean requires at least one value")
    return sum(items) / len(items)
```

Materializing the iterable allows an emptiness check and repeated use, but costs
memory proportional to input size. That tradeoff belongs in the contract. A
streaming mean could keep a running sum and count instead.

Avoid mutable defaults such as `def collect(item, result=[])`. The default
object is created once and shared across calls. Use `None` and create a new list
inside, or accept the collection explicitly.

Separate pure domain transformations from files, network, environment variables,
clocks, and random generators. Pure logic accepts relevant state as input and
returns output, making tests fast and precise. Thin boundary functions adapt
real I/O to that logic.

## Exceptions are part of the interface

Raise an exception when a function cannot honor its contract. `ValueError`
fits invalid content of the right general type; `TypeError` fits an unsupported
type; a domain exception can identify a failure callers may recover from.

This destroys evidence:

```python
try:
    load_records()
except Exception:
    return []
```

An empty dataset is now indistinguishable from a missing file, permission
failure, corrupt content, or programming bug. Catch at a boundary, add context,
and preserve the cause:

```python
try:
    value = float(raw_value)
except (TypeError, ValueError) as error:
    raise ValueError(f"score must be numeric, received {raw_value!r}") from error
```

The traceback then shows both the domain message and original failure. Catch
only errors that the current layer can handle meaningfully.

## Data classes, protocols, and composition

A data class suits a small record with named fields and generated comparison or
representation behavior. `frozen=True` prevents ordinary field rebinding and
communicates value semantics, but it does not recursively freeze nested objects.

Classes combine state and behavior when that model makes lifecycle clearer. Do
not introduce inheritance merely to avoid repeated lines. Small functions,
composition, and structural interfaces such as `Protocol` often produce clearer
AI pipelines. A predictor can be any object with a compatible `predict` method;
it need not inherit from your base class.

## Iterators, generators, and resources

An iterable can produce an iterator; an iterator yields one item at a time and
is usually consumed once. A generator suspends at `yield`, so it can stream
records without retaining all of them:

```python
def positive(values):
    for value in values:
        if value > 0:
            yield value
```

Calling `positive(values)` does not run its body. Work begins as a consumer asks
for items. Laziness affects when exceptions occur and when files or streams are
accessed. Test partial consumption and cleanup, not only `list(generator)`.

Context managers (`with ...`) express resource lifetime. Use them for files,
locks, database transactions, and temporary resources so cleanup occurs during
success and exceptions.

## Modules, environments, and entry points

A module is an importable `.py` file; a package groups modules behind an
interface. Imports execute top-level statements once per interpreter process,
so avoid training models, reading large data, or making network calls at import.
Put command behavior behind `main()` and:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Use a virtual environment per project and record dependency constraints. Run
commands from a documented directory instead of relying on an accidental import
path. Never name a local module `json.py`, `typing.py`, or another standard
library name because it can shadow the intended import.

## Debugging, tests, and logging

Debugging is controlled hypothesis testing:

1. Preserve the input, command, environment, traceback, and observed state.
2. Reduce to the smallest reproducible case.
3. State expected behavior and one hypothesis.
4. Inspect the earliest boundary where expected and observed values diverge.
5. Change one condition and run the narrowest test.
6. Add a regression test, then run the complete relevant suite.

Unit tests isolate contracts; integration tests verify boundaries cooperate.
Test normal, boundary, malformed, and recovery cases. Avoid tests that merely
repeat the implementation. For floating-point behavior, compare against an
explicit tolerance and explain its scale.

Use `logging` for operational events and attach context such as record count or
request identifier without exposing credentials or personal data. Use returned
values for program behavior; do not make tests parse decorative log text.

## Concurrency and performance awareness

Concurrency overlaps work; parallelism performs work simultaneously. Threads
can help blocking I/O, while processes or native numerical libraries may help
CPU work. Async code helps many cooperative I/O operations but adds cancellation
and lifecycle complexity. None fixes an inefficient algorithm or excess data
transfer.

First establish correctness, a representative workload, and a metric. Use
`timeit`, profiling, and memory measurements rather than intuition. Optimize the
measured bottleneck, then rerun correctness tests. In AI systems, vectorized or
compiled library operations often dominate Python micro-optimizations.

## Runnable lab: typed record transformations

The lab validates untrusted mappings, yields immutable observations, and
computes stable group summaries. From the repository root:

```bash
python3 curriculum/foundations/01-python-foundations/lab/transformations.py
python3 -m unittest discover -s curriculum/foundations/01-python-foundations/lab -v
```

Expected output contains groups `control` and `treatment`; six tests pass. Read
the [lab guide](lab/README.md), predict each result, and trigger its documented
failures before modifying the implementation.

## Exercises

1. **Recall:** explain binding, mutation, `None`, iterable, iterator, pure
   function, and exception chaining without reopening this page.
2. **Implementation:** add a validated integer `attempt` field to the lab and
   summarize only each subject's latest attempt. Write tests first.
3. **Debugging:** remove `from error` from one failure. Compare tracebacks and
   explain which evidence was lost.
4. **Analysis:** return a list instead of a generator. Measure peak memory for
   10,000 and 1,000,000 records; state when the simpler list is still preferable.
5. **Extension:** define a `Protocol` for a record source and implement an
   in-memory source without coupling domain logic to file I/O.
6. **Production:** identify input values unsafe or misleading to log and design
   a redaction test.

Do not open the [project solution](../project/solution.md) until preserving a
real attempt, failing test, and written hypothesis.

## Common misconceptions and recovery

- **“Type hints validate input.”** Standard annotations are not runtime guards.
- **“A tuple makes everything immutable.”** Nested mutable objects stay mutable.
- **“Catching every exception is robust.”** It usually hides failure meaning.
- **“Generators are always faster.”** They alter memory, overhead, and timing;
  measure the actual workload.
- **“Async makes code parallel.”** Async normally coordinates cooperative work
  on an event loop; CPU work can still block it.
- **“Passing tests means correctness.”** Tests cover selected cases under stated
  assumptions. Review the contract and untested risks.

## Knowledge check

1. Why does assigning a list to a second name not copy it?
2. When must `is None` replace a truthiness check?
3. What is gained by exception chaining?
4. When does a generator body begin execution?
5. Why should imports avoid expensive or external side effects?
6. What evidence is required before optimizing a function?

Score one point per precise answer with an example. Below five requires a fresh
example and another lab failure analysis before the project.

## Completion criteria

- [ ] Six lab tests pass from a clean checkout.
- [ ] You triggered invalid identifier, group, numeric, finite-value, duplicate,
      and empty-input failures and explained each detection boundary.
- [ ] You completed four exercises, including implementation and debugging.
- [ ] Your independent data explorer satisfies its brief and tests.
- [ ] Functions have types and focused responsibilities; effects are isolated.
- [ ] You can explain every caught exception and untested assumption.
- [ ] Another person can run the artifact using only its README.
- [ ] You score at least 5/6 here and 80/100 on the stage gate.

## Summary and glossary additions

Names bind to objects; mutation and ownership determine how state travels.
Functions and types communicate contracts, validation protects boundaries,
exceptions preserve failure meaning, and iterators control evaluation and
memory. Tests and measurements replace confidence with inspectable evidence.

- **Aliasing:** multiple references reach the same object.
- **Boundary:** where data or control crosses trust domains or components.
- **Iterator:** stateful object yielding a sequence one item at a time.
- **Pure function:** output depends on explicit inputs without externally visible
  side effects.

## Authoritative further reading

- [Python tutorial](https://docs.python.org/3/tutorial/)
- [Python data model](https://docs.python.org/3/reference/datamodel.html)
- [`typing` documentation](https://docs.python.org/3/library/typing.html)
- [`unittest` documentation](https://docs.python.org/3/library/unittest.html)
- [`venv` documentation](https://docs.python.org/3/library/venv.html)

Stable concepts include binding, iteration, contracts, and boundary validation.
Syntax and typing capabilities are version-sensitive; use documentation matching
the installed Python version.

Next, learn [linear algebra](../../mathematics/01-linear-algebra/README.md) and
use the [NumPy academy](../../../Libraries/NumPy/README.md).
