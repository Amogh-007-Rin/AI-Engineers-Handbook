# Solution notes

The pure `summarize` function is testable without Dash, while the app test checks
the integration graph. Production dashboards should bound inputs, cache by user
and query identity, expose loading/error states, and use browser tests.
