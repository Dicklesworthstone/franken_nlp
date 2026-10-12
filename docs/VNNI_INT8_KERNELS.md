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

## Dispatch and reporting

`Dispatcher::int8_gemm` and `int8_gemv` execute X1a/X1b/X2 when explicitly
forced or named in a compatible caller-supplied measurement row. The returned
kernel ID identifies the body actually used; its maximum (M,N,K) tile is
(4,1,64) for X1a or (4,1,32) for X1b/X2. Measurements naming other tile
geometries are rejected rather than attributed to a different implementation.

Selection can be tested with synthetic feature vectors, but the execution
boundary always constructs the sealed handle using the real OS-aware detector.
Forged VEX/EVEX capability metadata cannot authorize unsupported instructions.
The detection policy includes AVX2 where its actual body uses AVX2 operations;
X1a does not inherit a blanket AVX2, VL or BW requirement. VEX and EVEX exposure
are independent, not inferred from one another or from a CPU marketing name.

`robot backends` reports these implementations as registered on x86-64 without
changing its schema. Registered means code is present, not that qualification
or a performance comparison passed. No measurement rows ship with this change:
an empty table still selects scalar with conservative-default provenance.
Non-scalar INT4 remains unimplemented; a forced VNNI INT4 request fails, while
an inapplicable INT4 table row falls back to scalar without a measured claim.
Native inference continues to use its existing immutable explicit kernel policy;
this change does not make standalone dispatch tables an ambient model default.

## Evidence boundary

Twenty new regression definitions in `vnni_int8.rs`, `vnni_widths.rs` and
`dispatch_vnni.rs` cover real-host admission, all signed byte pairs in actual
64-byte vector blocks, offset cancellation, full-model bounds, 64 alignments,
vector tails, independent cohorts, matrix tails, aliases/permutations,
shape/allocation refusal, forced/table body identity, all 32 feature masks,
foreign architectures, INT4 isolation and honest default/registry reporting.
Existing all-kind kernel and CLI definitions enumerate all tiers. Native
projection definitions now also exercise 63/64/65 and 127/128/129 K boundaries
through selected/full F32 and BF16 projections and grouped decode/prefill.

Compilation, Rust tests, native/model execution and sustained measurements were
not run in this session under the controller-owned validation policy. Source/API
and diff review is not an instruction, parity, code-generation or benchmark run.
This advances bead `franken_nlp-gzh` without claiming its full closure. Offline
packed/cache-blocked execution, exact-pin code-generation inspection and
sustained target-host qualification remain outstanding. No speedup or
model-quality claim is made.

API references:
<https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_avx_epi32.html>
<https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_epi32.html>
<https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_dpbusd_epi32.html>.
