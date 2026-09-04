# Security policy

## Supported content

Security fixes target the current `main` branch. The project has not yet issued
a stable release; older snapshots may not receive patches.

## Report a vulnerability

Use GitHub's private vulnerability reporting feature for this repository. If it
is unavailable, contact the repository owner privately through their GitHub
profile. Do not open a public issue for an unmitigated vulnerability.

Include the affected path and commit, realistic impact, a minimal reproduction
using synthetic data, and a suggested mitigation when known. Never send real
credentials, private data, proprietary artifacts, or harmful payloads.

## What to expect

The maintainer will assess scope and severity, coordinate a fix and regression
test, and agree on disclosure timing when appropriate. Response times are
best-effort for a volunteer-maintained project. Upstream dependency issues may
be redirected to the relevant security process.

In scope are repository code and CI, examples that teach unsafe defaults,
secret leakage, unsafe deserialization, unbounded agent actions, artifact trust
failures, and privacy-sensitive data handling.
