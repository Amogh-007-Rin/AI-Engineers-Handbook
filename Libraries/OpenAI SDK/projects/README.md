# OpenAI SDK safety contract

Validate bounded requests and allowlisted tool calls without making network
requests. Run `python -W error -m unittest -v`; extend with mocked structured
outputs, timeout/retry state, redacted traces, and a budgeted live smoke test.
