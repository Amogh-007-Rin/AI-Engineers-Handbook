# Solution notes

The project keeps provider calls outside unit tests and rejects unbounded
requests or tool side effects. Production code should use typed schemas,
idempotency keys, policy checks, secret managers, and separate live-evaluation
budgets.
