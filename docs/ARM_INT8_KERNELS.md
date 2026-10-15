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
including selected vocabulary rows and grouped native projections. Fixed scale
order, BF16 casts, complete work reservations and output admission are retained.
The choice is explicit, never a hardware-derived preference. Scalar remains the
default; wrong-architecture or unsupported-ISA requests fail before model/input
access. No new worker/runtime or weight copy is introduced.

`Int8Kernel::gemm_into` uses maximum tile geometries (M,N,K) of (4,2,8) for
SMMLA and (4,1,16) for SDOT/autovec. M/N tails are supported without padding the
caller arrays. The standalone scalar/AVX2 convenience GEMM routes retain their
existing tiles.

## Paired columns in native inference

`Int8Kernel::dot_tile_into` accepts one to four independently borrowed activation
rows and one or two weight rows, with compact row-major output. It checks the
entire tile before publishing output, including the second weight row. Sparse,
noncontiguous, unaligned and read-only aliased rows do not need a packing copy.
SMMLA reaches its existing checked two-column body. Other handles retain their
own shared-row implementation per column; no result is relabeled as another ISA.

When the explicitly selected kernel is SMMLA, both single-input and grouped
`QuantizedLinear` projections now pair real output columns. Selected vocabulary
IDs are paired in selection order, not by physical adjacency: no intervening
unselected weight row is projected. An odd final column uses the existing
one-column cleanup. Each cell retains its own activation scale, selected row
scale, canonical epilogue and final BF16 cast. This reaches the decoder and
vocabulary-head users of those projection APIs without another engine or CLI.

The complete logical M*N*K work is charged once by the public projection
entrypoint before computation. Two-column tiles do not increase the reservation,
refund work, or double the cancellation gap: the paired loop polls before a tile
would exceed 32 independent dots between checkpoints, plus a final poll. The
exact polling/write order may differ from the old one-column loop; any execution,
cast or cancellation error still invalidates the complete output. There are no
hot heap allocations or full-matrix intermediates. Other kernel selections keep
their previous projection loops. No automatic dispatch preference changes.

Intrinsic bodies are compiled only for little-endian AArch64. SMMLA uses the
nightly `stdarch_neon_i8mm` feature only on that target; other targets do not
receive the feature attribute. Unsafe operations remain confined to the two
enumerated ARM files, behind actual CPU checks and explicit bounds proofs.
The paired-column integration changes no instruction body, lint or allowlist.

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
are installed by this change: an empty table still selects scalar. The standalone
INT4 dispatcher remains separate from the explicit packed-INT4 projection APIs;
INT8 registration alone does not grant an INT4 implementation.
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

Six added `borrowed_int8_tiles` regression definitions cover live tile shapes,
64 alignments, K tails, independent signed-byte pairs, compact N=1 extraction,
sparse/aliased rows and no-write rejection. Existing `native_int8_kernel` and
portable batch definitions now reach paired projection when SMMLA is available.
An independent Python index/accounting model checked 4,288 small shapes and 25
model-width shapes: every small-shape output was visited once, total logical
work was unchanged, and all checkpoint gaps were at most 32 dots. This does not
execute the Rust source or validate floating-point/ISA behavior.

Compilation, Rust tests, model parity, code-generation inspection and target-host
benchmarks were not run in the paired-column implementation session. These are
code-first candidates, not closure of the full hpt/g2n offline-packed-panel or
performance gates. No speedup or model-quality improvement is claimed.

Intrinsic APIs:
<https://doc.rust-lang.org/core/arch/aarch64/fn.vdotq_s32.html>
<https://doc.rust-lang.org/core/arch/aarch64/fn.vmmlaq_s32.html>.
