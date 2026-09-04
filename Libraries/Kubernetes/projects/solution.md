# Solution notes

The contract rejects mutable image tags and probe/resource omissions before a
cluster can accept them. Production evidence includes admission policy output,
rollout telemetry, and a tested previous revision rollback.
