# Executable notebooks

These output-clean notebooks provide short, dependency-free practice loops for
Python contracts, model evaluation, and responsible-AI slice analysis. Each
contains executable assertions and a written reflection prompt.

Run all notebooks from the repository root:

```bash
python3 scripts/execute_notebooks.py
```

The runner rejects committed outputs and executes code cells sequentially in a
shared namespace. Library-specific executable projects live under each
`Libraries/<academy>/projects/` directory and use their isolated environment.

Notebook contributions must declare a kernel/runtime, avoid secrets and large
embedded data, keep outputs uncommitted, contain deterministic assertions, and
include failure cases plus reflection. Use scripts or projects when a workflow
needs packaging, services, concurrency, or multi-file tests.
