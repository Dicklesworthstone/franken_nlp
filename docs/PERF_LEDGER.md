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

### PERF-6WT-Rope-projection-epilogue

- Claim ID: `PERF-6WT-Rope-projection-epilogue`
- Evidence: no measurement yet; the fused projection-epilogue implementation is
  retained as a bit-equality candidate only.
- Fixture hashes: `pending: no benchmark fixture has been accepted`
- CPU feature string: `pending: no host measurement`
- Command + environment: `pending: benchmark command is not yet authorized`
- Disposition: `deferred`
- Regime: `R1-latency-generate`
- Host fingerprint: `pending: no host measurement`
- Artifact recipe + packing + kernel table + load mode: `pending: no artifact`
- p50/p95/p99: `pending: no measurement`
- Fairness controls: `pending: same artifact, numerics profile, thread cap, and
  thermal state are required before promotion`
- Bandwidth denominator: `measured-dram-bytes`

The fused variant remains **default off** until this row is replaced by a
measured, profile-scoped winner for its exact `(ISA, shape, regime)` key. A
slower or non-promoted result remains in this ledger as `rejected`; it is never
removed merely because the unfused path remains selected.

## PERF-G9MI-R4-ADMISSION-PREFLIGHT

- Claim ID: `none`
- Evidence: no R4 long-context measurement has been made. The current tree has
  an admitted-cap RoPE table candidate, but no `fnlp robot plan --ctx --batch
  --quant` surface, active artifact/packing path, or tested official llama.cpp
  baseline from which to select an admitted context point and account for the
  complete process commitment. This row is a fail-closed preflight record, not
  a measurement of model acceptance, KV practicality, latency, or RSS.
- Fixture hashes: `pending: R4 fixtures require an exact admitted-memory certificate`
- CPU feature string: `pending: no host measurement`
- Command + environment: `n/a: R4 execution waits for the admission, artifact, and official-baseline surfaces`
- Disposition: `deferred`
- Regime: `R4-long-context`
- Host fingerprint: `pending: no host measurement`
- Artifact recipe + packing + kernel table + load mode: `pending: no runnable artifact or packing set`
- p50/p95/p99: `pending: no measurement`
- Fairness controls: `blocked: select every context point from the future admission certificate, then match oracle and official-baseline prompt, precision, thread, load-mode, warmup, and thermal controls`
- Bandwidth denominator: `measured-dram-bytes`

This row must be replaced—not promoted—only after the retained oracle and
official-baseline runs provide host-qualified context points, prefill/decode
distributions, peak RSS, exact KV overhead, admission boundary outcomes, and
cancellation-observation latency. Until then, the model's observed 262,144
position limit remains a non-practicality fact and does not authorize a >8K
claim.
