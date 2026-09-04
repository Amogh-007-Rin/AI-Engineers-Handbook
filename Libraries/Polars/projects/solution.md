# Polars project solution guidance

The reference validates keys before lazy execution, expresses aggregation as a plan, uses a left `1:1` join to preserve customers, assigns meaningful zero identities, and collects only at the output boundary. A production solution should preserve orphan audits, specify input schemas during scans, inspect the optimized plan, and verify streaming support rather than assuming every lazy query streams.
