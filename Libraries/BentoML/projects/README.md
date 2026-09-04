# BentoML service contract

Validate model version, typed schemas, limits, readiness, and metrics without a
server. In the declared environment, the native test constructs a real BentoML
1.x service, verifies its traffic limits and `/predict` API schema, then runs
the method in-process without opening a port. Run `python -W error -m unittest
-v`; extend with an ASGI request test, signed model lookup, a real Bento build,
load test, canary, and rollback bundle.
