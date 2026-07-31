<!-- fnlp-ledger-schema: perf-ledger/v1 -->

# Performance ledger

This ledger keeps every performance campaign, including slower candidates.
Its percentiles and bandwidth numbers are meaningful only with the named host,
artifact, command, and fairness controls. STREAM peak and vendor TOPS are
context, never attainable denominators.

## Entry schema

Every `PERF-...` entry contains these fields. `Disposition` is one of `won`,
`rejected`, `prior`, or `deferred`. A sibling-project result is `prior`, never
a FrankenNLP measurement.

- `Claim ID`
- `Evidence`
- `Fixture hashes`
- `CPU feature string`
- `Command + environment`
- `Disposition`
- `Regime` — one of `R0-cold-warm-startup`, `R1-latency-generate`,
  `R2-corpus-scoring`, `R3-corpus-generation`, or `R4-long-context`.
- `Host fingerprint`
- `Artifact recipe + packing + kernel table + load mode`
- `p50/p95/p99`
- `Fairness controls` — thread count, allocator, precision, warmup, and any
  additional control required to compare candidates.
- `Bandwidth denominator` — explicitly `logical-tensor-bytes`,
  `packed-payload-bytes`, or `measured-dram-bytes`.

`Fixture hashes` uses immutable `sha256:<64-lowercase-hex>` values. A deferred
campaign may use `pending:` only while no measurement has been made.

## Current entries

No local performance campaign has been measured at scaffold time.
