# Finite CPU kernel selftest

`fnlp robot selftest` executes a bounded synthetic integer suite on the current
binary and CPU, without model weights, input documents, inference network access,
or an asupersync runtime. It does not install a process kernel preference.

```sh
fnlp robot selftest
fnlp robot selftest --tier s_scalar_i32
fnlp robot selftest --tier x3b_avx2_widened_i16 --details
fnlp robot selftest --schema
```

The CLI names are the same exact kernel identifiers accepted by candidate
inference. `--tier` is local to this command, not an ambient default. Unknown
names and model/input arguments are usage errors. `--schema` is an execution-free
schema request and conflicts with `--tier` and `--details`.

## What it actually checks

The fixed `cpu-int8-execution-v1` suite has 180 cases per tested tier: eighteen
reduction lengths, five operation/shape combinations, and two input patterns.
It covers K=0, vector boundaries 7/8/9, 31/32/33, 63/64/65, 127/128/129, and
model reduction lengths 3072, 6144, 10751 and 10752. It exercises forced native
GEMV, single-row GEMM, four-row GEMM, a five-row matrix tail, and the separately
used four-row shared-weight cohort API. Patterns include full signed-byte
extremes, nonuniform weights and zero activation rows.

Every emitted expected value comes from a deliberately simple i64 reference,
not the scalar implementation under test. Every result is checked for size,
exact integer equality and the identity of the body reported by dispatch or
its actual detected handle. Equal numbers from the wrong kernel fail. Output
checksums are SHA-256 over a fixed domain, element count and little-endian i32
bytes; they describe only repository-authored synthetic data, never private text.

The largest input pair is 53,760 activation bytes plus 32,256 weight bytes;
report storage is bounded by nine times 180 case records. Those are logical
buffer lengths, not measured peak RSS or a full production admission receipt.
There are no user-supplied matrix dimensions, repetitions or input bytes.

## Results and automation

Stdout contains canonical NDJSON records under protocol `fnlp.cpu-selftest`,
schema version 1. This protocol is separately namespaced from the existing
version-3 task/health records; their schemas and command behavior are unchanged.
`--schema` emits the exact schema for all five selftest record types.

A run starts with `selftest_start`, followed by `selftest_tier` summaries and a
terminal `selftest_complete`. With `--details`, `selftest_case` records precede
their tier summary. Failing cases are emitted even without `--details`, including
expected/actual checksums, first differing synthetic coordinates/integers, or a
fixed native-refusal category. Resource failures produce `selftest_error`, not
a successful terminal summary. An incomplete/broken output stream exits nonzero.

Exit 0 means every executed case passed and at least one tier was completely
tested. Default all-available mode reports unsupported tiers explicitly as
`skipped_unsupported_isa`; it does not execute instructions merely because a
registry entry exists. A detected but unimplemented tier blocks a pass.
An explicitly requested unsupported tier yields verdict `unsupported` and exit 2.
Integer/route failures and execution/output failures exit 1. Command-line usage
errors exit 2. A partial transcript or a missing terminal record is not a pass.

## Scope and implementation status

A selftest pass covers only the finite, versioned synthetic integer cases that
actually ran on that binary and CPU. It is not exhaustive ISA/memory safety,
full fixed-N model-shape coverage, floating-stage parity, full-model execution,
performance evidence, automatic dispatch qualification, or a release gate.
The existing scalar model default is unchanged. The controller-owned DSR and
bead-specific qualification requirements remain separate.

This implementation adds a callable library engine, a real CLI route and
regression definitions. Compilation, Rust tests and execution of the new
selftest were not run in the implementation session; source/API and diff review
are not passing validation receipts.
