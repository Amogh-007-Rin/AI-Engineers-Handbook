# Solution guidance

Keep `profile` pure, validate schema before summarizing, and convert numeric
columns deliberately rather than silently coercing arbitrary text. The complete
solution uses `csv.DictReader`, catches file/parse errors at the CLI boundary,
emits stable JSON, returns non-zero exit codes, and tests stdout/stderr separately.
Record the Python version, commands, fixture license, assumptions, and one bug
investigated with a minimal reproducer.
