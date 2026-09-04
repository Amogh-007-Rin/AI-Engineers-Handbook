# Maintenance and review policy

Repository owner: `@Amogh-007-Rin`. Ownership means triage and assignment; it
does not imply one person must possess every domain expertise or approve their
own substantive changes.

## Verification cadence

| Content | Routine cadence | Required reviewers |
|---|---:|---|
| Foundations, mathematics, data, classical ML | Quarterly | Technical + pedagogy |
| Deep learning, domain AI, GenAI, agents | Every 8 weeks | Domain + pedagogy + responsible AI |
| Production, serving, security, external SDKs | Monthly | Domain + security/operations |
| Responsible AI, licensing, accessibility | Quarterly and after policy change | Relevant independent reviewer |
| Links, metadata, dependency compatibility | Every pull request and weekly scheduled CI | Maintainer triage |

Every review records commit, date, runtime/environment, reviewer identity/role,
scope, commands, findings, remediation owner, and next due date in a review
record copied from `templates/review-record-template.md`. Never mark `published`
from an unrecorded verbal or self-review.

## Change and incident workflow

1. Triage correctness, security/privacy, broken execution, accessibility, and
   license issues before enhancements.
2. Reproduce the issue on the declared environment and add a failing regression
   test or minimal evidence fixture.
3. Correct content and code, update `last_verified`, and run both structural
   audits plus relevant environment suites.
4. Request independent review appropriate to the table above.
5. For harmful or materially incorrect guidance, mark content `maintenance`,
   add a visible warning, and publish a corrected release note.

Dependency upgrades must retain an older compatibility lane until migration is
verified. Model/dataset replacements require license, provenance, checksum,
quality, safety, cost, and deletion review. Security findings should avoid
public exploit details until mitigation is available.

## Release evidence

A stable tag requires the two strict structural audits, clean notebook and CPU
project runs, scheduled framework results, link/license/secret/accessibility
checks, resolved high-severity findings, learner-journey records, and named
review approvals. Attach commit hashes and workflow URLs to
`reports/verification-status.md`; a workflow file alone is not execution proof.
