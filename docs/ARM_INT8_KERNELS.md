# Explicit ARM64 INT8 candidates

The Generic-row native engine has three independent ARM candidates:

- `a1_smmla`: signed matrix-multiply-accumulate with real OS-aware `neon` and
  `i8mm` admission. Dense GEMM processes up to four activation rows and two
  output columns in eight-K steps. Operands are assembled in registers from
  borrowed rows, not repacked into a second weight matrix. Short rows/columns
  use zero register halves without reading nonexistent data; K tails are scalar.
- `a2_dotprod`: signed NEON SDOT with real OS-aware `neon` and `dotprod`
  admission. Complete 16-byte blocks share one weight load across one to four
  activation rows; scalar tails never over-read.
- `a3_autovec`: safe Rust i32 MACs with sixteen independent K accumulators per
  row. No intrinsics, unsafe code or scalar-oracle call-through. LLVM's actual
  vectorization and performance are build/host questions, not asserted here.

```sh
fnlp candidate --int8-kernel a2_dotprod generate prompt.txt \
  --model local.fnlpq --memory-mib 8192
fnlp candidate --int8-kernel a1_smmla batch --task ner \
  --model local.fnlpq --memory-mib 8192 < documents.ndjson
```

Use the existing `asupersync-runtime` build and explicit local candidate model.
All three options flow through the already-borrowed decoder/head matrices,
including selected vocabulary rows and grouped native projections. Existing
fixed scale order, BF16 casts, work charges, cancellation and output admission
are unchanged. The choice is explicit, never a hardware-derived preference.
Scalar remains the default; wrong-architecture or unsupported-ISA requests fail
before model/input access. No new worker/runtime or weight copy is introduced.

`Int8Kernel::gemm_into` uses maximum tile geometries (M,N,K) of (4,2,8) for
SMMLA and (4,1,16) for SDOT/autovec. M/N tails are supported without padding the
caller arrays. Grouped native projections use the existing one-weight-row
cohort API, including SMMLA with an inactive zero second column; this is not a
claim that the whole transformer has been converted into a two-column GEMM.
The standalone scalar/AVX2 convenience GEMM routes retain their existing tiles.

Intrinsic bodies are compiled only for little-endian AArch64. SMMLA uses the
nightly `stdarch_neon_i8mm` feature only on that target; other targets do not
receive the feature attribute. Unsafe operations remain confined to the two
enumerated ARM files, behind actual CPU checks and explicit bounds proofs.

## Dispatch and reporting

The existing `Dispatcher::int8_gemm` and `int8_gemv` entrypoints execute A1/A2/A3
when explicitly forced or selected by a supplied compatible measurement row.
Each result identifies the body actually used and its maximum tile geometry;
no scalar result is relabeled as an ARM execution. The execution boundary
independently detects the real host, so an injected feature report cannot
mint a foreign-architecture or unsupported-ISA handle. `dotprod` and `i8mm`
remain separate capabilities. Inapplicable compilation targets report no body.

`robot backends` reports ARM implementations as registered on their applicable
targets without changing the report schema. Registration means source exists,
not that qualification or a performance comparison passed. No measurement rows
are installed by this change: an empty table still selects scalar. INT8
registration does not authorize an ARM INT4 path; explicit INT4 requests fail,
and unavailable table entries fall back with conservative-default provenance.
Measurement rows naming a tile with no corresponding body are rejected.

## Evidence boundary

The safe A3 arithmetic has portable test definitions; ISA cases skip explicitly
on unsupported test hosts. Dense tests cover SMMLA operand/result layout,
row/column/K tails, unaligned views, batch permutation, shape-before-write
rejection and extreme independent accumulators. Dispatch tests cover actual
forced/table execution, hostile injected capabilities, operation isolation,
allocation failures, target-specific registry state and unchanged defaults.
Existing exhaustive scalar/i64, native projection and CLI matrix definitions
enumerate all new kinds; immutable process-policy tests include all candidates.
Compilation, Rust tests, model parity, code-generation inspection and target-host
benchmarks were not run in this implementation session. These are code-first
candidates, not closure of the full hpt/g2n offline-packed-panel/performance gates.

Intrinsic APIs:
<https://doc.rust-lang.org/core/arch/aarch64/fn.vdotq_s32.html>
<https://doc.rust-lang.org/core/arch/aarch64/fn.vmmlaq_s32.html>.
