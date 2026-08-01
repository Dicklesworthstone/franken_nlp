# Reality Check & Bridge Plan — 2026-07-31

Produced by the orchestrator's full reality-check pass (code audit by independent read-only
agent + bead-coverage cross-check + campaign evidence). Revised in place; do not fork this
document.

## 0. Proof-state correction — 2026-08-01

The implementation inventory and bridge ordering below remain useful, but the original
`WORKING`, `proven`, `green`, and `closed` labels are **not acceptance authority**. A fresh
literal-contract audit found that they merged code-first/static progress with DSR,
model-gated, host-gated, and release evidence that does not yet exist. Until the affected
rows are rewritten individually, interpret those labels as **CODE-FIRST IMPLEMENTED OR
STRUCTURALLY EXERCISED; ACCEPTANCE OPEN**.

Current load-bearing corrections:

- No retained DSR `scripts/check.sh` receipt exists for the current immutable source SHA.
  Local Cargo, direct RCH, and GitHub Actions results are non-authoritative for this campaign.
  The owner policy disables GitHub Actions, but the retained workflow still auto-triggers on
  push/pull request and must be made inert. One occasional isolated DSR checkpoint is run only
  after the moving shared tree reaches a deliberately selected stable SHA and the named
  `production` feature graph exists.
- A real CPU/eager smoke now verifies the complete ten-file source closure before model load.
  The retained five-process × two-thread-count nondeterminism campaign predates that
  verification, however, and its own authority record said the model closure was not yet
  verified. The current oracle record explicitly keeps that floor historical. Exact stable
  prefixes and parity gates therefore require a new full-source-bound floor campaign.
- The later source-bound trace capture is valuable: all nine trace indexes retain 268 prefill
  and 268 append records, including 44 post-layer states, two post-loop norm states, and 44
  K/V slots per phase; a fresh static audit matched all 4,824 retained sidecar digests. It is
  not yet a 44+2 parity award because it imports the historical floor, and the committed
  fixture-verification receipt predates the trace regeneration and no longer identifies the
  current manifest.
- `.fnlpq` reader/writer/converter/package components contain substantial real code, but the
  canonical envelope/digest authority is internally inconsistent, the production model-root
  opener intentionally refuses, the converter does not complete staged reload + strict
  receipt + reconstruction/selftest + atomic activation, and release packaging has not been
  proven on a validated real canonical Generic artifact.
- The fetchers implement the pinned ten-file/8,360,887,509-byte closure and retain useful
  fixture coverage, but both Unix and PowerShell `--check-only` paths currently return exit 0
  when all ten files are absent. That contradicts the verification contract (`0` means
  complete/verified) and can counterfeit success for an exit-code-only caller. Trusted custom
  catalog parity, pre-contact redirect refusal, real full-closure transcripts, and a full
  Windows receipt also remain open.
- `fnlp pull`, `install.sh`, and `install.ps1` are still absent. The required end-user flow
  remains: install the exact binary, delegate all model acquisition to that binary's
  release-bound `fnlp pull`, stream-verify every fixed 1,957,046,720-byte part (except the
  tail), reassemble, validate, derive native packing, and atomically activate.
- Several Beads remain marked closed despite newer retained `KEEP OPEN` or
  `KEEP OPEN/BLOCKED` audit comments and/or live open blockers. Their closure status and every
  readiness result derived from it are challenged. Only work that remains ready after honoring
  the literal dependency and proof contracts may be scheduled.

These corrections preserve all implemented code and ambition. They change only the evidence
grade: code-first progress is valuable, but it cannot mint DSR, model, host, platform, or
publisher proof.

## 1. The honest verdict

**franken_nlp today is a substantial code-first reference/artifact scaffold, not yet a usable
or proved product.** No clean-SHA DSR code receipt exists, the production feature graph is not
wired, and model, artifact, platform, performance, publisher, and human-release gates remain
independent and open.

**Valuable evidence-bearing work that must be preserved and requalified:**
- A hash-locked CPU/eager environment and a real full-ten-file source-bound smoke transcript.
- A complete source-bound HF trace corpus for eager bf16, diagnostic f32, and variance-only
  SDPA, with internally consistent 44-execution + two-norm/KV sidecars. Its imported
  nondeterminism floor and retained verification receipt are stale, so it is structural/model
  evidence rather than an L2 native-parity award.
- Substantial `.fnlpq` writer, reader, converter, packager, canonical-JSON, execution-identity,
  receipt, calibration, validation, grammar, tokenizer/template, resource-broker, and memory
  ledger code. The envelope authority is split, the owned reader is not a production-scale
  admitted reader, conversion stops before qualified reload/selftest/receipt/activation, and
  packaging accepts unqualified inputs; none is release authority yet.
- Pinned source-fetch implementations and fixture transcripts. Their empty-cache success exit,
  cross-shell policy/evidence gaps, and missing real full-closure receipts must be corrected
  before source acquisition closes.

**REAL but UNWIRED (library code with zero in-src consumers — ~7,900 LOC, ~22% of src):**
tokenizer (asset not embedded), template, grammar masks, validation, receipts, calibration,
storage, native-cache packing (CLI leg refuses), the entire forward pass (reachable only from
tests; loads raw safetensors, not `.fnlpq`).

**MISSING outright:** every NLP task (11 one-line stub files), autoregressive decode loop,
sampler, batch scheduler, batch daemon, durable jobs, `fnlp pull`, SIMD kernel bodies (all 8
tiers `_pending`; crate-wide `unsafe` denial means the enumerated-island policy work hasn't
begun), `fnlp tokens/generate/chat/doctor/eval/...` — no CLI command runs inference at all.

## 2. Vision checklist (README promise → status)

| # | Goal | Status | Bead coverage |
|---|------|--------|---------------|
| 1 | `fnlp convert` → canonical Generic `.fnlpq` | CONVERTED (per §0: first artifact produced 2026-07-31 20:49, 4,690,873,282 bytes, fnlpq-file-sha256 15c57d4d…, anchors ×5; acceptance OPEN — receipt/reload-verify (xmy m6), two-clean-dir determinism (m7), 9rs independent read-back, 7p1s qualification all pending) | xmy in flight (m6-7) |
| 2 | Reference-fidelity ladder L0/L1/L2 | WORKING | closed |
| 3 | L3/L4/L5 ladder rungs | PARTIAL | arc, zre, 0yz, wk5 |
| 4 | End-to-end inference from CLI | MISSING — **4 broken joints** (see §3) | partial coverage — **gaps** |
| 5 | Task portfolio (extract, ner, sentiment, classify, judge, redact, summarize, keyphrases, answer, resolve) | MISSING (stub files) | 4B/4C epics — full coverage |
| 6 | generate/chat + sampler + thinking | MISSING | k45, 9tz, w6b7 |
| 7 | Valid-by-construction constrained decoding wired into generation | PARTIAL (compiler+masks real, unwired) | bw1, 91j, k9e, 95v |
| 8 | Batch fabric (layer-major, COW KV, prefix cache) | MISSING | 4A epic (4zh, mki, bdn, 7eu, 17h) |
| 9 | `fnlp batch` daemon + durable jobs | MISSING | l47, lva, 040, eas, zzh, xof, zcr |
| 10 | `fnlp pull` verified streaming install | MISSING | g9y, s3c, kzu |
| 11 | SIMD kernel campaign (NEON/AVX2/VNNI) | MISSING (scalar only) | y4w epic (hpt, g2n, fms, 3am, gzh, 53o, 11p) + cwr6 host-gated |
| 12 | int8→int4 staged quantized artifacts with parity gates | PARTIAL (recipe v1 emitting) | 73p, n23, of3, wwg, j2h |
| 13 | Robot mode full contract | PARTIAL (honest skeletons) | j47, 8lx, q4g |
| 14 | eval/calibrate/qualify user evidence | PARTIAL (math real, unwired) | 81o, 1m4 |
| 15 | Release/dist/install/attestation | MISSING (staged P6) | w9e epic |
| 16 | NlpEngine library with task methods | PARTIAL (lease facade only) | 4A + ngg |
| 17 | tune/doctor/models/licenses | MISSING | qtg, 3fn, 8lx |
| 18 | KV int8 runtime mode | MISSING | fpy |
| 19 | Metal staging (affirmed track) | STAGED | piw in flight |
| 20 | Model-root txn + activation journal | PARTIAL (**production opener unconditionally refuses**) | n27 closed (simulation); **gap** |

**Would completing every open bead close the gap?** Almost — the 169-bead graph is unusually
complete; the 4A→4B→4C→P3→P5→P6 staging maps the entire product. The audit found **six narrow
NO_BEAD / under-specified joints**, now covered by the beads in §4.

## 3. The four broken joints (why no token can flow today)

1. **Artifact→engine bridge missing.** Nothing converts a checked `.fnlpq` tensor
   (int8 panels + scales + row-sums, or bf16-verbatim sections) into engine weight
   structures. `hf_bf16_eager` loads raw HF safetensors and is documented never-CLI-reachable.
2. **Tokenizer asset not embedded.** `EmbeddedTokenizer::from_bytes` awaits an
   `include_bytes!` that doesn't exist; no CLI text can become ids.
3. **Template/engine/task wiring absent.** Template, masks, validation, receipts, calibration
   all have zero in-src consumers — the 4A/4B beads integrate them, and joint order matters.
4. **No generation loop.** `decode()` runs one position; nothing feeds `greedy_token` back.
   `decode.rs`/`sampler.rs`/`batchsched.rs` are one-line files (beads w6b7/9tz/4zh own them).

## 4. Bridge beads (created this pass)

| Bead | Joint | One-line contract |
|------|-------|-------------------|
| ARTIFACT-ENGINE BRIDGE | §3.1 | `.fnlpq` → typed engine weights (bf16-verbatim AND int8+scales+row-sums), census-on-load, digest cross-check, no raw-safetensors production path |
| EMBEDDED TOKENIZER ASSET | §3.2 | `include_bytes!` pinned tokenizer + configs, digest vs truth-pack pin + artifact copy, pre-model `tokens`/schema paths live |
| FNLP TOKENS COMMAND | §3.2 | exact counts, `--json`, robot event, no model load |
| MODEL-ROOT PLATFORM SURFACE | §2 row 20 | real owner-only opener (handle-relative/no-follow) for macOS/Linux per PLATFORM_SURFACES registry; replaces the unconditional refusal; unlocks `models derive` + real activation |
| INT8 EXECUTION PROFILE | §2 row 12 | forward variant consuming int8 panels through proven scalar kernels + quant algebra under the quantized profile's preregistered gates |
| ROBOT CONVERT EVENTS | hygiene | honor the parsed-but-dead `--robot` flag with versioned stage events |

## 5. The spine (critical path to first token, then first task) — REVISED r1

**The spinal-cord test (campaign north star).** One end-to-end gate, run in the central
suite once S1 lands, that no later change may break:
`fnlp generate --greedy -n 64 <fixture prompt>` through the REAL CLI, loading the REAL
canonical `.fnlpq`, must emit byte-for-byte the oracle's greedy transcript on the
oracle-reproducible prefix set (the L4 contract exercised through the shipping binary, not
the test harness). Until S1, the gate is RED and that is the honest top-line status.

**Milestone S1 — "First canonical token."** xmy artifact ✓ → 6qz7 bridge → 0w9e embedded
tokenizer → w6b7 decode loop (greedy only) → minimal `fnlp generate` slice of k45.
*Memory shape constraint (binds 6qz7):* the checked reader is owned-buffer by design (mmap
is gated behind R0/ej1a), so the bridge must stream section bytes into weight structures
incrementally — peak RSS ≤ weights + one section in flight + bounded scratch, never
envelope-resident + weights-resident doubled (~9 GB would violate the admission doctrine on
16 GB hosts). Exit: spinal-cord test GREEN at n=8 first, then 64.

**Milestone S1.5 — "First shipped value, no generation loop needed."** The B3 bet makes
classification-shaped tasks reachable BEFORE autoregressive maturity: one prefill + sliced
lm_head rows is exactly what the engine already does (single-position `decode()` +
`export_logits_f32`). Sequence: S1 bridge+tokenizer → 95v prompt ABI (minimal slice) →
10a `fnlp classify` prefill-only mode and 5l6 `fnlp sentiment` distribution mode, each with
its locked scorecard from 0yz before any marketing wording. This ships a real, measurable
user capability weeks before extract's full constrained-decode stack and exercises the
template/tokenizer/engine joints under the simplest decode strategy. It also forces the
lm_head sliced-row personality (B3) into existence early, where it is cheapest to prove
(sparse=full exact-equality gate on tiny label sets).

**Milestone S2 — "First guaranteed-valid task."** S1 + 95v full ABI + bw1/91j execution
compiler + ngg TaskIR → bbq `fnlp extract`. Constrained-greedy over the real artifact with
schema-valid-by-construction output is the thesis (B4). The independent validator
(validation/, already real) must be the acceptance judge — never the grammar's own automaton
grading its own homework.

**Milestone S3 — "Corpus fabric."** 7eu sealed team + 17h admission + wlb budgets +
mki KV slabs + 4zh scheduler + bdn prefix cache → l47 batch daemon → 040/eas durable jobs.
KV arithmetic stays load-bearing here: 176 KiB/token/seq bf16 means batch 64 × 8K context
≈ 88 GiB — the admission certificate, not optimism, decides what runs.

**Milestone S4 — "Fast."** P3 SIMD campaign (unsafe-island policy work first: the crate-wide
`unsafe` denial must grow the enumerated-allow module list + CI policy test BEFORE any
intrinsic lands) behind PG-2 exactness and the dispatch registry that already exists.
G1>G2: scalar-first correctness ships; speed lands on proven semantics only.

**Swarm lane map (collision-free parallel dispatch):**
| Lane | Beads (order) | Panes |
|------|---------------|-------|
| Spine | 6qz7 → 0w9e → w6b7 → k45 slice → gdlc | 2 panes (bridge is the hot seat) |
| Prefill value | 95v slice → 9tz fused-greedy slice → ngg core → 10a → 5l6 (+0yz scorecards) | 2 panes after spine's bridge |
| Task compiler | 91j → bw1 → k9e → ngg → bbq | 2 panes |
| Fabric | 7eu → 17h/wlb → mki → 4zh → bdn | 2 panes |
| Substrate/release | 7u3d → 8lx; 7p1s after 73p; j47; 5wty | 2 panes |
| In flight | fae, xmy(close), g9mi(parked), dsr, i2r, piw, hff | as-is |

Sequencing rationale: every milestone converts an existing *unwired-real* asset into a
wired, gated, user-visible capability; nothing waits on research; the lanes touch disjoint
module sets (exact-path commit discipline holds).

## 6. Integration gates (each joint gets its own proof)

- Bridge gate: engine-from-artifact ≡ engine-from-source on the L2 44+2 fixture set
  (bf16-verbatim path), plus census-on-load refusal fixtures naming the stage.
- Tokenizer gate: embedded-bytes digest == truth-pack pin == artifact copy; L0 conformance
  re-run through the embedded path.
- Decode gate: N-token greedy transcript ≡ oracle greedy on reproducible prefixes (the L4
  contract, exercised through the CLI slice, not only the test harness).
- Task gate: extract output validates under the independent validator (never the grammar's
  own automaton) + verbatim fields byte-verified — the B4 bet, end to end.

## 6b. Process upgrades forced by tonight's evidence — r2

Conversion attempts 1→5 each failed at a NEW joint (hardcoded refusal wiring, half-committed
API, naming grammar at write time, non-atomic failure stub, silent 28-minute logs). The
generalization is a house rule for every integration bead in this plan:

1. **Real-data rehearsal before close.** Synthetic fixtures systematically miss real-scale
   failure modes (three-pass runtimes, staged-write cadence, name-grammar collisions). An
   integration bead's close evidence must include one run against the real artifact/model
   data path, not only its unit fixtures. The orchestrator runs these centrally (batch-verify
   discipline unchanged — agents still never build).
2. **Stage-line observability is a contract, not a favor.** The `CONVERT STAGE=… RESULT=…`
   convention (which turned attempt 5 from undiagnosable to self-narrating) extends to every
   long-running surface this plan creates: artifact load, prefill, batch epochs, job
   commit/resume, pull. Robot-parseable, identity-carrying, versioned under j47's contract.
   Validation moves to the EARLIEST stage that can rule (plan-time, not write-time — the
   28-minute lesson).
3. **Fail-closed must also fail-clean.** Attempt 4 left a 0-byte destination stub; the fix
   (hidden staging + rename) is now the mandatory shape for every artifact-producing surface
   (derive, pull, job materialization). A failed run leaves NOTHING at any destination path.
4. **Quantized honesty guard.** The first int8 artifact existing is NOT evidence the int8
   MODEL works: until 7p1s lands and the quantized profile's preregistered L3/L4 budgets are
   measured, no wording anywhere (README, robot health, release notes) may describe the int8
   artifact as validated. The artifact is `converted`; it becomes `qualified` only past its
   gates. This distinction enters the models-state vocabulary (8lx).
5. **Wiring-debt metric.** 7,900 LOC (~22% of src) currently has zero in-src consumers.
   Tracked per tick from here; the S1–S3 DONE definitions include their subsystems' consumer
   edges existing. Target: <2% by S3 close (residual = deliberately-staged P5/P7 surfaces).
6. **Bridge-phase non-goals (scope walls):** no streaming token API before j47 freezes the
   NDJSON contract; no mmap before R0/ej1a rules; no speculative/loop-draft work (P7 cards
   own it); no new dependencies of any kind (the closed universe is load-bearing); no SIMD
   intrinsics before the unsafe-island policy expansion lands with its CI test.

## 7. Risks & standing blockers

- **cwr6** (Zen-3 measurement) and the SUITE-pin ratification remain owner-gated.
- fs_tx production opener is fail-closed by design until the platform surface is ratified —
  the new bead makes ratification+implementation explicit instead of implicit.
- The README stays present-tense-as-spec per its tense note; `hef` (P6 docs truth pass)
  trues it up at release; no wording change needed now.
