# Contributor handbook

Start with the repository-wide [contribution guide](../CONTRIBUTING.md). This
section routes content work to the contracts that make a contribution teachable,
executable, reviewable, and maintainable.

## Authoring route

1. Define the learner, prerequisite evidence, measurable outcomes, and the smallest complete learn–practice–assess slice.
2. Select the relevant [template](../templates/README.md) and cite primary sources close to version-sensitive or empirical claims.
3. Include runnable work, expected behavior, realistic failures, exercises, separated solutions, pass criteria, remediation, and compute requirements.
4. Validate locally with the commands in [CONTRIBUTING](../CONTRIBUTING.md).
5. Request technical, pedagogical, execution, accessibility, and applicable security/responsible-use review.
6. Record the reviewed commit and evidence using the [review template](../templates/review-record-template.md).

## Review questions

- Can the target learner explain the concept before invoking an API?
- Does every objective have practice and assessment evidence?
- Does the clean path run within the declared CPU/free-tier budget?
- Are expected failures taught, and do tests include boundaries and malformed inputs rather than only a happy path?
- Are data/model provenance, licenses, privacy, accessibility, security, cost, and maintenance handled in proportion to risk?
- Are claims traceable and limitations honest?
- Are solutions separated while still useful for remediation?

Never mark your own unreviewed work as independently approved. Use `draft` or
`review` until the publication gates in [project.md](../project.md) have actual
evidence.
