# Solution notes

The contract rejects unversioned models and unbounded requests before deployment.
Keeping prediction validation independent of BentoML makes boundary behavior
cheap to test; the native layer proves decorator and schema integration without
confusing in-process execution with a deployed health check. Production evidence
should include Bento/model hashes, dependency lock, authenticated HTTP tests,
load testing, telemetry, and a tested schema-compatible rollback.
