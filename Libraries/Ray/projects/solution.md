# Solution notes

The validator makes retries safe by requiring an idempotency key for side
effects. A production run also captures serialized code/environment, object
store pressure, task metrics, and actor checkpoint lineage.
