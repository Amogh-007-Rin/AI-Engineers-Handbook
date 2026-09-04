# Typed record transformation lab

Predict each output before execution. The lab isolates three contracts:

1. `parse_observation` validates one untrusted mapping and returns an immutable
   value object;
2. `iter_observations` validates lazily and detects duplicate identifiers; and
3. `summarize` consumes typed observations and computes deterministic output.

Run from the repository root:

```bash
python3 curriculum/foundations/01-python-foundations/lab/transformations.py
python3 -m unittest discover -s curriculum/foundations/01-python-foundations/lab -v
```

The first command prints three normalized observations and means of `0.5` for
control and `0.85` for treatment. Six tests pass. No network or third-party
package is required.

## Failure practice

In a temporary copy or interactive session, try a blank `subject_id`, unknown
group, missing score, the string `"high"`, `float("nan")`, and a duplicated
identifier. Record when the error occurs. The duplicate is intentionally not
raised until the generator reaches its second record.

Do not weaken validation to make malformed data pass. Extend the test first,
make the smallest implementation change, and run the whole lab suite afterward.

## Completion evidence

Save the Python version, exact command, passing output, one failing traceback,
your prediction, corrected mental model, and the commit containing the extension.
Explain what these six tests do not establish about real-world data.
