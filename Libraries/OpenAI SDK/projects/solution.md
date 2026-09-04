# Solution notes

The injected transport keeps provider calls outside unit tests while still
testing the actual SDK request and response types. Validation rejects non-finite
budgets, unbounded output, and unsafe tool calls before client construction.
Production code should use typed schemas, idempotency keys, policy checks,
secret managers, redacted telemetry, and separate live-evaluation budgets.
