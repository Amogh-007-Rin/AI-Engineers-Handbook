# Ray job contract

Validate resource reservations, retry budgets, and idempotent side effects
without starting a Ray cluster. Run `python -W error -m unittest -v`; extend
with remote tasks, actor checkpoints, cancellation, and a failure-injection run.
