# P7 Metal prefill scope gate

**Status:** DEFERRED — no implementation authority

**Owner ruling:** 2026-07-31

## Decision

Apple-integrated-GPU acceleration is a future, opt-in `ft-kernel-metal` track
for Nanbeige4.2-3B. It may accelerate prefill first; it does not alter the
CPU's status as the correctness authority or portable product floor. Decode
stays CPU-only until a separately reviewed decision supplies its own evidence.
CUDA is out of scope.

This record intentionally creates no Cargo feature, Metal binding, dispatch
route, unsafe island, server surface, or translation task. `fnlp` must keep
its exact CPU behavior on every host while this record is deferred.

## Absolute entry gate

No Metal source may land until the CPU-parity certification records all of the
following for the same canonical artifact recipe and CPU profile:

1. The CPU L-ladder is green, including L2 comparisons for all 44 decoder
   executions (`layer + loop * 22`) and both post-loop RMSNorm states.
2. The named CPU baseline and its retained fixtures identify the model source
   revision `f56ec5a9650268aa098496734743c25ea778bd2d`, conversion receipt,
   packing selection, profile, and test environment.
3. Measured dispatch selection is available, so an eventual GPU route is a
   reversible measured choice rather than a hard-coded replacement for CPU.

Absence of any item is `BLOCKED_CPU_PARITY`; it is not a reason to add a
placeholder GPU feature or to relax the CPU gates.

## GPU profile contract

The only reserved profile name is `metal-prefill-v1`. It owns GPU fidelity
claims and inherits none from `hf-bf16-eager`, `diagnostic-f32`, or a CPU
quantized profile. Before dispatch can select it, its immutable profile record
must specify:

| Required field | Required decision/evidence |
| --- | --- |
| Artifact identity | source revision, conversion receipt, logical-model digest, packing set, tokenizer/template/config closure |
| Cast points | each host-to-device, kernel-input, accumulator, reduction, and device-to-host cast point; no implied bf16 behavior |
| Floating comparisons | per-operation named metric and tolerance vectors, with fixture identities and non-finite behavior |
| Integer comparisons | L0 exact comparisons where the stage is integer |
| State ladder | L2 vectors for 44 layer outputs plus two loop norms; L4 and L5 records under `metal-prefill-v1` |
| Fallback | CPU implementation/profile selected by the same dispatch identity when Metal is unavailable, declined, or fails admission |

The mandatory end-to-end gate is a frozen-prefix GPU-prefill/CPU-decode run.
It logs dispatch choice, profile, tolerance-vector identity, and the typed
reason for `SKIPPED_NO_MODEL`, `SKIPPED_NON_APPLE_HOST`, or GPU unavailability.
It never converts a skip into a pass or a GPU result into CPU evidence.

## Measurement promotion rule

Only a post-gate performance entry may promote the candidate from deferred.
That entry compares GPU prefill with the proven CPU path on the *same host*,
artifact, prefix population, batch shape, and thermal state. It retains
p50/p95/p99, repeated-trial distribution, thread cap, CPU kernel table,
allocator, precision/profile, warmup, and energy when the host exposes a
credible energy measurement. A faster median without the named parity result,
tail distribution, or fairness controls leaves the default CPU-only.

## Adjacent scope decisions

| Surface | Decision | Revisit condition |
| --- | --- | --- |
| `fnlp serve` or any remote/routable listener | DEFERRED and out of scope. A resident process or AA-R1 experiment does not authorize a server design. | A separately claimed decision with owner-approved local-IPC evidence, framed protocol, bounded admission, and an explicit routability review. |
| Translation | DEFERRED and not a task surface. | Frozen multilingual evaluation population, task-quality scorecards, calibration evidence, and a separate task-surface decision. |

Neither future discussion is a fallback for the Metal track, and neither may
be implemented while its revisit condition is unmet.

## Rejected shortcuts

| Shortcut | Rejection reason |
| --- | --- |
| Add Metal now behind a default-off feature | It would create implementation authority before CPU parity and could change the release closure without profile evidence. |
| Treat GPU output as `hf-bf16-eager` | GPU cast/reduction behavior needs its own declared profile and tolerance authority. |
| Replace CPU dispatch once GPU is available | CPU must remain a portable, correctness-authoritative fallback; availability is not a measured performance decision. |
| Add `serve` or `translate` as part of a GPU change | Both are independently gated product decisions with missing evidence. |

## Promotion checklist

Before this status can change from `DEFERRED`, retain the CPU entry certificate,
the complete `metal-prefill-v1` profile, L0/L2/L4/L5 results, the frozen-prefix
E2E receipt, and a fair prefill A/B entry in `docs/PERF_LEDGER.md`. The
implementation review must also prove that a CPU-only build/host retains
identical CPU behavior and that `ft-kernel-metal` is a pinned FrankenSuite
surface rather than an ad-hoc binding.
