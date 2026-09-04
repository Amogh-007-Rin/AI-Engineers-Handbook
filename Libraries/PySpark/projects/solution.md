# Solution notes

The validator catches unsafe distributed assumptions before execution. Production
evidence includes `explain` plans, partition/skew metrics, data-quality counts,
checkpointed input snapshots, and an idempotent output commit protocol.
