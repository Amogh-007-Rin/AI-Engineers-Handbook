# OpenAI SDK safety contract

Validate finite budgets, bounded outputs, and allowlisted tool calls without
making network requests. In the declared environment, an optional native test
uses the real OpenAI Python SDK and an injected HTTP transport to serialize a
Responses API request and parse its typed response without a key, network call,
or billable model invocation. Run `python -W error -m unittest -v`; extend with
structured outputs, explicit retry state, redacted traces, and a separately
authorized budgeted live smoke test.
