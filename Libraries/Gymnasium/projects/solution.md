# Solution notes

The environment uses `super().reset(seed=seed)` so Gymnasium owns RNG setup and
returns a five-field step result. A production wrapper must transform its space
and test whether reward/termination semantics remain valid.
