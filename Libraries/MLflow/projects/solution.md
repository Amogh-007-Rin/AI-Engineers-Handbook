# Solution notes

The manifest makes hidden provenance explicit and refuses promotion without a
quality score and model signature. A production implementation logs immutable
artifacts to MLflow and proves clean-environment reproduction. The native local
store test exercises the real client without requiring a server; production
must additionally authenticate the tracking endpoint, authorize artifact
access, and define retention and rollback ownership.
