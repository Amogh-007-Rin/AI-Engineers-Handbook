# ML systems stage assessment

Pass at 80/100: 20 API/artifact contracts, 20 reliability/idempotency/resources,
20 telemetry and SLO evidence, 20 security/privacy/supply chain, and 20 canary/
rollback/cost. Public admin access, mutable artifacts, missing timeouts, or no
rollback automatically fail. Remediation requires an injected-failure trace and
repeated successful recovery.
