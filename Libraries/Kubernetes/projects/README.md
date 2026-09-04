# Kubernetes deployment contract

Validate a deployment’s immutable image, resources, and three probe semantics
without a cluster. Run `python -W error -m unittest -v`; extend it with a kind
smoke test, service/network policy, canary metrics, and rollback execution.
