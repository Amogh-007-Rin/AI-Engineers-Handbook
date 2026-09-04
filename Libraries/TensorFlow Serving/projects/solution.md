# Solution notes

Specific version policies make rollback deterministic and prevent an unnoticed
model swap. A production deployment also validates SavedModel signatures and
measures queueing and tail latency under realistic concurrency.
