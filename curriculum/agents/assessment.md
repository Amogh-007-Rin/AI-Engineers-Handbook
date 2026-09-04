# AI agents stage assessment

Build a bounded tool-using agent. Pass at 80/100: 20 task/state design, 25 tool
schema/authorization/idempotency, 20 injection evaluation, 20 budget/failure/
recovery, and 15 reproducibility. Unknown tools, unconfirmed consequential
actions, secret exposure, or an unbounded loop automatically fail. Remediation
requires narrower permissions and new failing regression tests.
