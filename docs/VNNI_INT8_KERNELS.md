# Explicit VNNI INT8 candidates

`x2_avx_vnni` is a real VEX AVX-VNNI candidate over borrowed Generic INT8 rows.
Actual OS-aware `avx2` and `avxvnni` detection is required. AVX-512 is not.
It uses `_mm256_dpbusd_avx_epi32`, not the similarly named EVEX intrinsic.

```sh
fnlp candidate --int8-kernel x2_avx_vnni generate prompt.txt \
  --model local.fnlpq --memory-mib 8192
```

Use the existing `asupersync-runtime` candidate build and local artifact.
The shared implementation registry exposes X2 across candidate commands and
existing decoder/head projections, including grouped projections. Scalar is
still the default. Existing scale order, BF16 casts, work budgets, cancellation
and output admission remain unchanged.

The leaf shares complete weight loads and correction across one to four
activation rows, with independent i32 accumulators. Dense GEMM uses four-row
cohorts. Unsigned offsets exist only in registers: the signed result is exactly
`dot(u,w) - 128*sum(w)`. Correction is computed from the same weight blocks,
not supplied metadata. Complete 32-byte blocks need no padding/alignment;
short tails use signed scalar cleanup. K <= 10752 bounds every partial.
All operand geometry is checked before caller-visible output writes.

The implementation is contained in the existing enumerated x86 SIMD module
`simd/avx2.rs`; no unsafe-policy allowlist or crate-root lint is changed.
No dependency, scheduler, matrix expansion, packing format or network is added.

Six new regression definitions cover actual feature admission, zero-activation
correction, boundary impulses, 64 input alignments, independent cohorts,
dense tails and rejection without writes. Existing all-kind kernel, projection
and CLI definitions also enumerate X2. Compilation, Rust tests, native/model
execution and performance measurements were not run in this session.

This advances bead `franken_nlp-gzh` without claiming its full closure. X1a/X1b,
general dispatcher registration, offline packed/cache-blocked execution and
sustained target-host qualification remain separate work at this checkpoint.

API: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_avx_epi32.html>.
