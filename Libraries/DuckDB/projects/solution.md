# DuckDB project solution guidance

The reference left join preserves customers with no transactions, `COUNT(t.id)` returns zero for their null-extended row, and `COALESCE(SUM(...),0)` encodes the additive identity intentionally. Orphans are audited separately. Primary/check constraints turn assumptions into failures. Production extensions should stage files, validate before replacement, use transactions for publication, and distinguish single-process analytical use from concurrent service storage.
