# Explicit VNNI INT8 candidates

Three distinct implementations consume borrowed Generic INT8 rows:

| Candidate | ISA required on the real host | Instruction width |
|---|---|---|
| `x2_avx_vnni` | AVX2 + AVX-VNNI | 256-bit VEX |
| `x1b_avx512_vnni_ymm` | AVX2 + AVX-512F + VNNI + VL | 256-bit EVEX |
| `x1a_avx512_vnni_zmm` | AVX-512F + VNNI | 512-bit EVEX |

The VEX body uses `_mm256_dpbusd_avx_epi32`. The similarly named
`_mm256_dpbusd_epi32` is the EVEX intrinsic: it cannot implement the X2
capability contract. X1a uses dword splats, full-vector loads and XOR so it
does not additionally require AVX-512BW or VL. Each constructor independently
checks the actual OS-aware feature surface; CPU names are not authority.

```sh
fnlp candidate --int8-kernel x2_avx_vnni generate prompt.txt \
  --model local.fnlpq --memory-mib 8192
fnlp candidate --int8-kernel x1b_avx512_vnni_ymm generate prompt.txt \
  --model local.fnlpq --memory-mib 8192
fnlp candidate --int8-kernel x1a_avx512_vnni_zmm batch --task ner \
  --model local.fnlpq --memory-mib 8192 < documents.ndjson
```

Use the existing `asupersync-runtime` candidate build and local artifact.
The shared implementation registry exposes all three across candidate commands
and existing decoder/head projections, including selected vocabulary rows and
grouped projections. Scalar remains default. Existing scale order, BF16 casts,
work budgets, cancellation and output admission are unchanged. No wider tier
is preferred merely because the host advertises it.

Each leaf shares complete weight loads and correction across one to four
activation rows, retaining independent i32 accumulators. Dense GEMM uses the
same four-row cohorts with one output column. X2/X1b process 32 K bytes per
step; X1a processes 64. Missing M rows never enter the instruction body. Short
K tails use signed scalar cleanup rather than an oversized or masked load.
This does not implement the separate offline-packed/cache-blocked panel path.

Unsigned offsets exist only in registers: the signed result is exactly
`dot(u,w) - 128*sum(w)`, where `u = x XOR 0x80`. Correction comes from the same
borrowed weight blocks, not supplied metadata; scalar tail bytes are excluded
from the vector correction. K <= 10752 bounds every partial independently:
raw magnitude <= 350945280, correction <= 176160768, conservative combined
magnitude <= 527106048, and corrected signed result <= 176160768. These bounds
fit i32. The implementation uses non-saturating VPDPBUSD, not DPBUSDS or a raw
saturating pair instruction. Every operand is checked before output writes.

All intrinsic bodies stay in the existing enumerated x86 SIMD module
`simd/avx2.rs`; neither the unsafe allowlist nor the crate-root lint changes.
No dependency, scheduler, matrix expansion, packing format or network is added.

## Evidence boundary

The X2-specific and cross-width regression definitions cover real-host feature
admission, all signed byte pairs in actual 64-byte vector blocks, zero-offset
cancellation, full-model bounds, 64 input alignments, every vector tail,
independent cohorts, matrix tails, aliases/permutations and no-write refusal.
Existing all-kind kernel, projection and CLI definitions enumerate all tiers.
Compilation, Rust tests, native/model execution and sustained measurements were
not run in this session. Source/API review is not an instruction or parity run.

This advances bead `franken_nlp-gzh` without claiming its full closure. General
dispatcher registration, offline packed/cache-blocked execution, exact-pin
code-generation inspection and sustained target-host qualification remain
separate work at this checkpoint. No speedup or model-quality claim is made.

API references:
<https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_avx_epi32.html>
<https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_epi32.html>
<https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_dpbusd_epi32.html>.
