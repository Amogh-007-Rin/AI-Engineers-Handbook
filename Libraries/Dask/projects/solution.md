# Solution notes

The project returns a lazy scalar so execution is explicit and testable. A
production pipeline should inspect graph size, partition memory, scheduler,
serialization, retries, and output idempotence before scaling out.
