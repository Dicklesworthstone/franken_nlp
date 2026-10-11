# Explicit ARM64 INT8 candidates

The Generic-row native engine now has two independent ARM candidates:

- `a2_dotprod`: signed NEON SDOT with real OS-aware `neon` and `dotprod`
  admission. Complete 16-byte blocks share one weight load across one to four
  activation rows; scalar tails never over-read. It is compiled only for
  little-endian AArch64, matching the stdarch intrinsic surface.
- `a3_autovec`: safe Rust i32 MACs with sixteen independent K accumulators per
  row. No intrinsics, unsafe code or scalar-oracle call-through. LLVM's actual
  vectorization and performance are build/host questions, not asserted here.

```sh
fnlp candidate --int8-kernel a2_dotprod generate prompt.txt \
  --model local.fnlpq --memory-mib 8192
fnlp candidate --int8-kernel a3_autovec batch --task ner \
  --model local.fnlpq --memory-mib 8192 < documents.ndjson
```

Use the existing `asupersync-runtime` build and explicit local candidate model.
Both options flow through the already-borrowed decoder/head matrices, including
selected vocabulary rows and grouped native projections. Existing fixed scale
order, BF16 casts, work charges, cancellation and output admission are unchanged.
The choice is explicit, never a hardware-derived preference. Scalar remains the
default; wrong-architecture or unsupported-ISA requests fail before model/input
access. No weight copies or new worker/runtime are introduced.

The safe A3 arithmetic has portable test definitions; actual SDOT definitions
skip explicitly on unsupported test hosts. Existing exhaustive scalar/i64 and
native projection tests enumerate both new kinds through the shared registry.
Compilation, Rust tests, model parity, code-generation inspection and target-host
benchmarks were not run in this implementation session. These are code-first
candidates, not a closure of the full hpt/g2n packed-panel or performance gates.

Intrinsic API: <https://doc.rust-lang.org/core/arch/aarch64/fn.vdotq_s32.html>.
