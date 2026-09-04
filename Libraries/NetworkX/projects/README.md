# NetworkX graph contract project

Construct a validated weighted graph, compute stable node features, and verify
JSON schema round-trip. Run `python -W error -m unittest -v`; extend it with
directed and disconnected fixtures plus a graph-aware train/test split. Both
serialization directions explicitly select the `links` field so a NetworkX
default change cannot silently alter the persisted wire schema.
