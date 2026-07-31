<!-- fnlp-ledger-schema: perf-ledger/v1 -->

## PERF-FIXTURE-MISSING-FAIRNESS-001

- Claim ID: none
- Evidence: benchmark receipt
- Fixture hashes: sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
- CPU feature string: avx2
- Command + environment: `fnlp bench`; threads=1; allocator=system; profile=diagnostic-f32
- Disposition: rejected
- Regime: R1-latency-generate
- Host fingerprint: fixture-host
- Artifact recipe + packing + kernel table + load mode: recipe=r1; packing=none; table=scalar; load=buffered
- p50/p95/p99: 1ms/2ms/3ms
- Bandwidth denominator: logical-tensor-bytes
