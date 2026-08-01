# Reality Check & Bridge Plan — 2026-07-31

Produced by the orchestrator's full reality-check pass (code audit by independent read-only
agent + bead-coverage cross-check + campaign evidence). Revised in place; do not fork this
document.

**Audit binding.** The source/document census was read through immutable repository commit
`1af4161e63532fd17c45e25a408865e2bb554a1b`; the final audit Bead graph is the separately
immutable ledger commit `b5876bb080c661ce11a35b33164a98f9a0c907bb`, whose exported
`.beads/issues.jsonl` SHA-256 is
`df9136cef793caa933d3191096b3ca15b44387949b7fcacbaa27ea605c746ddd`.
The initial closure repair is preserved in `3653d6449770a74a1e7e27f35d2f78a38834f62a`.
At the source binding, tracked source
paths were clean while eleven audit-owned documentation paths were modified and the peer-owned
untracked `.claude/` directory was deliberately excluded. Publisher settings were queried at
`2026-08-01T03:16:55Z`: Actions returned `enabled=true`, `allowed_actions=all`,
`sha_pinning_required=false`; immutable releases returned `enabled=false`,
`enforced_by_owner=false`. The current document bytes are identified by the Git commit that
contains them. Later source, graph, or publisher changes require an explicit delta review;
none silently updates this snapshot.

## 0. Proof-state correction — 2026-07-31 fresh-eyes pass

The implementation inventory and bridge ordering below remain useful, but the original
`WORKING`, `proven`, `green`, and `closed` labels in earlier revisions are **superseded and
not acceptance authority**. A fresh literal-contract audit found that they merged
code-first/static progress with DSR, model-gated, host-gated, and release evidence that does
not yet exist. The individually rewritten §2 rows and their strict vocabulary are the only
current status record.

Current load-bearing corrections:

- No retained DSR `scripts/check.sh` receipt exists for the bound source snapshot
  `1af4161e63532fd17c45e25a408865e2bb554a1b` or the later audit-only ledger/docs commits.
  Local Cargo, direct RCH, and GitHub Actions results are non-authoritative for this campaign.
  The owner policy disables GitHub Actions, but the retained workflow still auto-triggers on
  push/pull request and must be made inert. One occasional isolated DSR checkpoint is run only
  after the moving shared tree reaches a deliberately selected stable SHA and the named
  `production` feature graph exists.
- The claims checker is itself not yet a clean static gate. Its mutation/fixture self-test
  passes, but the full public-surface scan rejects `src/cli.rs:12` because `NUMERIC_RE` matches
  the trailing `6` inside the Rust identifier `bf16`. That is a lexical false positive, not an
  unregistered public claim. The exact repair and a missing compliant-identifier fixture are
  recorded on open Bead `k3i`; weakening the numeric/superlative policy is not an acceptable
  workaround.
- The documentation cross-reference gate is also open for two separate reasons. The plan's
  required `docs/truth-pack/LICENSE_PROVENANCE.md` artifact is genuinely absent and remains a
  close gate on `r32`. Separately, `scripts/check.sh` recognizes repo-root `docs/...` prose
  references but resolves them relative to each nested source document, producing false
  `docs/adr/docs/...` failures; the resolver/fixture repair is recorded on `xu1`. Neither issue
  is hidden by skipping the documentation leg.
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
  fixture coverage. The formerly reported empty-cache `--check-only` false-success is fixed in
  both current scripts: a missing member sets the aggregate failure state and exits 1. This is
  a static source finding, not a shell/PowerShell execution receipt. Trusted custom-catalog
  parity, pre-contact redirect refusal (the current Unix effective-host check occurs after
  transfer), real full-closure transcripts, and a full Windows receipt remain open.
- `fnlp pull`, `install.sh`, and `install.ps1` are still absent. The required end-user flow
  remains: install the exact binary, delegate all model acquisition to that binary's
  release-bound `fnlp pull`, stream-verify every fixed 1,957,046,720-byte part (except the
  tail), reassemble, validate, derive native packing, and atomically activate.
- Eleven Beads remained marked closed at the bound audit snapshot despite newer retained
  `KEEP OPEN` or `KEEP OPEN/BLOCKED` comments, live open blockers, or an unresolved authority
  conflict. This pass reopened `6gi`, `6wt`, `72s`, `g6f`, `ilz`, `mzr`, `n27`, `o2y`,
  `sdb`, `snp`, and `vsx` with issue-specific reasons and preserved every old receipt/comment.
  Graph analytics remain structural hints, not proof that a gate passed.

The highest-severity correction is format authority: three mutually incompatible `.fnlpq` v1
families simultaneously called themselves frozen or ratified. One requires an
80-byte-per-entry binary directory, one a 256-byte-per-entry binary section table, and one
makes canonical JSON the sole range directory.
They also disagree on flags, caps, digest framing, and fixtures. All three candidate specs and
ADRs are now visibly quarantined as **AUTHORITY CONFLICT / REOPENED**; none may authorize
writer, reader, converter, receipt, package, pull, or release acceptance until
the owner ratifies one candidate and `franken_nlp-g6f` records the choice,
regenerates evidence, and marks rejected records historical.

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
- Pinned source-fetch implementations and fixture transcripts. The former empty-cache
  false-success is repaired in current source but has no fresh cross-shell execution receipt;
  policy/evidence gaps and real full-closure receipts remain before source acquisition closes.

**REAL but UNWIRED or SEMANTICALLY INCOMPLETE (the old ~7,900 LOC/~22% estimate is retained
only as a historical snapshot, not a current measurement):** tokenizer assets are now embedded,
but `EmbeddedTokenizer::pinned()` consumes only `tokenizer.model` plus `added_tokens.json`; it
does not apply the pinned tokenizer-config/special-map BOS/EOS authority and artifact activation
hash-checks only `TOKENIZER_MODEL`. Template rendering still flattens trusted controls and
untrusted content into one `String`, bypassing the existing control-excluding untrusted encoder.
Grammar masks, validation, receipts, calibration, storage, native-cache packing, and the forward
pass remain unwired to a shipping inference path; the real artifact loader still refuses and the
only materializing bridge is explicitly synthetic/non-authoritative.

**MISSING outright:** every NLP task (the task modules remain one-line stubs), autoregressive
decode loop, sampler, batch scheduler, batch daemon, durable jobs, `fnlp pull`, SIMD kernel
bodies (the dispatch registry says only scalar is implemented), and the product
`generate/chat/eval/...` surfaces. The tokens command is now being implemented code-first, but no
CLI command runs real model inference at the bound source snapshot.

### 1.1 Proof-scope ledger at this audit snapshot

| Evidence scope | Current state | What it may support | What it may not support |
|----------------|---------------|---------------------|-------------------------|
| Source/document inspection | **BOUND to `1af4161…` plus the containing documentation commit** | architecture, reachability, contradiction, static bounds, and mock/stub findings at those bytes | compilation, runtime correctness, performance, model parity, release readiness, or later peer deltas |
| Formatting/diff hygiene | **PASS for this documentation delta** | `git diff --check`, fence checks, and documentation-only UBS scan cleanliness | documentation cross-reference closure, Rust formatting, build/test, runtime correctness, model evidence, or semantic proof |
| Documentation cross-references | **BLOCKED** | root-aware inspection isolates the exact defects | genuine missing `docs/truth-pack/LICENSE_PROVENANCE.md` (`r32`) plus nested-doc resolver bug (`xu1`); no clean doc-link gate |
| Static governance validators | **MIXED / OPEN DEFECT** | ADR validator and its self-test pass structurally; claims-checker self-test passes | full claims gate: `--check` rejects the `bf16` identifier false-positive tracked by `k3i`; none of these results is DSR or model proof |
| Cargo/Rust code proof | **NOT RUN by this audit** | nothing yet | any `cargo check/test/clippy` claim |
| DSR code proof | **ABSENT for current SHA** | nothing yet | repository `PASS`, no-Rayon production closure, target portability |
| Oracle/model proof | **PARTIAL / STALE CHAIN** | the verified-source smoke and retained traces support narrow source/model structure | current nondeterminism floor, L2 parity award, native tokens, quantized fidelity |
| Native product/model run | **ABSENT** | nothing | real `NlpEngine`, CLI generation, tasks, batch, quality, long-context claims |
| Host performance | **ABSENT** | roofline hypotheses only | tok/s, docs/min, Apple/AVX2/AVX-512 dispatch winners, energy, p99 |
| Platform filesystem authority | **BLOCKED** | honest refusal behavior | real pull/derive/activation durability |
| Publisher/repository state | **QUERIED `2026-08-01T03:16:55Z`: Actions enabled; immutable releases disabled** | the need for `wdne`, `i2o3`, and `w9e.1` | artifact authenticity, immutable publication, release certification, or later settings state |
| Human/task-quality authority | **ABSENT** | synthetic mechanics only | extraction/NER/sentiment/classification quality or corpus acceptance |
| Bead graph | **BOUND to `b5876bb…`: 184 total, 163 open, 16 blocked, 5 in progress, 0 closed** | ownership and prerequisite planning at that ledger commit | proof that any implementation gate passed or later graph state |

Every later report must preserve these separations. In particular, a local or synthetic test
may be useful code-first evidence while its enclosing model/release gate remains open.

## 2. Vision checklist (promise → current reality → acceptance owner)

Status vocabulary is intentionally strict: `WORKING` means a current end-to-end product path
has the evidence named by its contract; `PARTIAL` means substantive implementation exists but
the path or proof is incomplete; `STUB` means only an interface/scaffold or deliberate refusal
exists; `UNPROVEN` means implementation may exist but the required evidence does not;
`REGRESSED` means later evidence invalidated an earlier award; `WRONG_APPROACH` means competing
authority or semantics make further acceptance unsafe; `NOT_STARTED` and `NO_BEAD` are literal.
Static hygiene, DSR code proof, model proof, host performance proof, publisher proof, and human
acceptance are separate columns of authority even when this compact table names only one status.

### Core product requirements

| # | Goal | Current status | Acceptance owner / missing evidence |
|---|------|----------------|-------------------------------------|
| 1 | One canonical Generic `.fnlpq` v1 | **WRONG_APPROACH** | `g6f`: choose exactly one of the quarantined 80-byte, 256-byte, or JSON-only authorities; mark the other two historical |
| 2 | Frozen envelope fixtures and digests | **REGRESSED** | `g6f`: recompute field-inventory and hostile-corpus digests after the owner decision; make multiple active v1 authorities a validator failure |
| 3 | Pinned ten-file source acquisition | **PARTIAL** | `mzr` plus fetch follow-up: source says empty-cache failure is fixed; real full Unix/Windows closure, custom-catalog, resume, corrupt, quarantine, and pre-contact redirect receipts remain |
| 4 | Deterministic Generic conversion | **PARTIAL** | `xmy`, `rsk`, `vk7`: emitted bytes are not qualified until two-clean-dir determinism or a narrowed canonical-publisher claim, checked reload, and receipt binding pass |
| 5 | Bounded converter memory | **UNPROVEN** | converter RSS ADR and real shard-scale host receipt; synthetic range arithmetic is insufficient |
| 6 | Qualified conversion completion | **STUB** | checked reconstruction, independent reload, self-tests, strict receipt, native derivation, and atomic activation must all succeed before `qualified` |
| 7 | Fixed-part release package | **PARTIAL** | `7nk`: zero/arbitrary inputs and zero-part PASS are forbidden; validate a real qualified artifact across the exact 1,957,046,720-byte boundary |
| 8 | Raw checksum versus semantic identity | **WRONG_APPROACH** | `g6f`, `7nk`, receipt work: standard raw SHA-256 and domain-framed identity need distinct frozen fields and cross-component vectors |
| 9 | `fnlp pull` end-user acquisition | **STUB** | `g9y`, `s3c`, `kzu`: release-bound preflight, ordered mirrors, bounded resume, streamed verification, reassembly, validation, native packing, activation |
| 10 | Binary-owned installers | **NOT_STARTED** | `doj`: install exact binary, then invoke that binary's `fnlp pull`; shell/PowerShell never parse the model manifest or join chunks |
| 11 | Secure model root and activation | **STUB** | `7u3d`: the uninhabited capability and unconditional refusal are honest; safe owner/ACL, handle-relative, lock, no-replace, durability authority is still absent |
| 12 | Artifact-to-engine bridge | **STUB** | `6qz7`: real Nanbeige load refuses; only an explicitly synthetic bridge materializes weights |
| 13 | Tokenizer identity and configured special IDs | **REGRESSED** | `0w9e`, `l44`, `eni`: bytes are embedded, but config/special-map semantics and configured `im_start`/`im_end` authority are not applied or fully artifact-bound |
| 14 | Chat-template identity and trust segmentation | **PARTIAL** | `fae`, `95v`, `k2h`, `sth`: 72-cell/reference work exists, but flattened trusted controls plus untrusted text bypass the typed untrusted encoder |
| 15 | Native model semantics | **PARTIAL** | scalar/test paths encode 22×2, 44 KV slots, 128 head dim, and two norms; no shipping artifact-backed forward proves the complete contract |
| 16 | Fresh L0–L5 fidelity chain | **REGRESSED** | `80z` and oracle/trace work: fixture-existence PASS is invalid; rerun the nondeterminism floor on verified source and reseal current traces before parity awards |
| 17 | Asupersync request/team ownership | **PARTIAL** | `7eu`, `wlb`, OQ-35 work: resource machinery exists, but product inference has no proved real blocking-pool crossing plus sealed `scoped_cpu` team |
| 18 | No-Rayon named production graph | **STUB** | `79m`, `xu1`: Cargo lacks `production`; `check.sh` does not select it and can PASS skipped legs; upstream unconditional Rayon remains blocking |
| 19 | Process-aggregate admission | **PARTIAL** | `17h` and resource ledger: formulas/guards exist; real weights+KV+scratch+caches+jobs+staging product admission is not wired or host-proved |
| 20 | Autoregressive generation and sampling | **STUB** | `w6b7`, `9tz`, `k45`: decode/sampler/scheduler production modules are one-line scaffolds; no real greedy-token loop exists |
| 21 | Valid-by-construction structured output | **PARTIAL** | compiler/mask metadata exists; no shipping automaton consumes logits, and same-implementation self-validation cannot be the acceptance judge |
| 22 | Grounded spans and verbatim fields | **PARTIAL** | validator/offset machinery exists; real task outputs do not yet prove exact source UTF-8 byte boundaries and typed ambiguity refusal |
| 23 | Usable library plus two CLI names | **PARTIAL** | thin-binary/resource-lease scaffolding exists; `NlpEngine` has no inference/task methods and no CLI command performs real model inference |
| 24 | Built-in NLP task portfolio | **STUB** | 4B/4C task Beads: all production task modules are one-line scaffolds |
| 25 | Bounded batch fabric | **STUB** | `4zh`, `mki`, `bdn`, `7eu`, `17h`: no layer-major product scheduler, COW KV fabric, prefix cache, or batch daemon |
| 26 | Durable jobs | **STUB** | `040`, `eas`, `zzh`, `xof`, `zcr`: contracts exist; no spool-first product journal/resume/materializer |
| 27 | Versioned robot protocol | **PARTIAL** | `j47`, `5wty`, `8lx`, `q4g`: schema/scaffolds exist; parsed-but-dead and success-shaped partial surfaces remain |
| 28 | No-network inference authority | **UNPROVEN** | capability design is strong; compile-time/runtime probes must show ordinary inference lacks IO/network/remote authority |
| 29 | Occasional DSR code authority | **STUB** | `rul7`, `xu1`, `79m`: no clean immutable-SHA `scripts/check.sh` receipt on the exact named production graph; no skipped policy legs allowed |
| 30 | Release certification | **NOT_STARTED** | P6 plus governance follow-ups: five targets, model-present run, signatures/SBOM/SLSA, offline replay, immutable release state, and trusted fingerprint all remain |

### Performance, quality, and operational requirements

| # | Goal | Current status | Acceptance owner / missing evidence |
|---|------|----------------|-------------------------------------|
| 31 | Scalar numerical authority | **PARTIAL** | substantial scalar/i64 algebra exists; every model K, tail, int4 group, and product integration path still needs central proof |
| 32 | Apple autovec/SDOT/SMMLA dispatch | **STUB** | dispatch registry says only scalar is implemented; FEAT_I8MM-gated kernels and shape/host measurements remain |
| 33 | x86 AVX2/VNNI/AVX-512 dispatch | **STUB** | plan correctly specifies exact X3a/X3b and zmm-vs-ymm trials; no production SIMD body or host receipt exists |
| 34 | Measured kernel selection | **NOT_STARTED** | no p50/p95/p99 thermal-steady host ledger, validity-domain key, or end-to-end dispatch confirmation |
| 35 | Quantization ladder | **PARTIAL** | `73p`, `n23`, `of3`, `7p1s`: converted int8 bytes are not an executable or fidelity-qualified quantized model; int4 remains gated |
| 36 | Native 48:8 GQA attention | **PARTIAL** | structural scalar code exists; artifact-backed prefill/decode and 44-deep KV reference comparisons remain |
| 37 | Prefix/cache/fork accounting | **NOT_STARTED** | `bdn`, `mki` and fork work: exact equality, COW page pricing, eviction, privacy namespaces, and admission receipts remain |
| 38 | Probability-space honesty | **PARTIAL** | the plan/contracts distinguish full-vocabulary, trie-local, and sequence spaces; shipping outputs/calibration do not yet exercise them |
| 39 | User `eval`/`calibrate`/`qualify` | **STUB** | mechanics/math exists but no model-backed product path or qualified dataset promotion path |
| 40 | Task quality evidence | **NOT_STARTED** | no locked real dataset/prompt/artifact/thinking-mode scorecard authorizes a quality claim |
| 41 | Claims and evidence ledgers | **PARTIAL** | schemas and empty/reserved ledgers exist; no public performance/quality superlative has earned a retained row |
| 42 | Receipt privacy and evidence grade | **PARTIAL** | typed receipt work exists; product-wide HMAC/retention/replay-grade enforcement remains unwired |
| 43 | Five-target portability | **UNPROVEN** | target declarations exist; measured OS/ABI floors and current DSR/model smokes do not |
| 44 | Token/text utilities | **STUB** | token command is code-first in flight; splitting/narrow normalization and byte/scalar coordinate contracts are not a shipped surface |
| 45 | Operational tools | **PARTIAL** | model/robot scaffolds exist; `doctor`, `tune`, licenses/provenance, active identity, dispatch, memory, and refusal reporting are incomplete |
| 46 | Official llama.cpp baseline | **UNPROVEN** | the correct post-support lineage is identified; no current fair retained correctness/performance run exists |
| 47 | Mechanical license closure | **PARTIAL** | Apache-2.0 authority and attribution policy are settled; every artifact/release byte-compare and modification notice is not yet proved |
| 48 | Enumerated unsafe islands | **PARTIAL** | policy is specified; future SIMD/mmap modules and the rejecting policy scan are not yet release-proved |

### Deferred research requirements

| # | Goal | Current status | Promotion boundary |
|---|------|----------------|--------------------|
| 49 | Metal prefill | **NOT_STARTED** | CPU parity first; independent `metal-prefill-v1` numerics, 44+2 evidence, fallback, and same-host fair gate |
| 50 | Long-context R4 | **NOT_STARTED** | no >8K practicality claim before exact admission, peak RSS, KV, latency, cancellation, and official-baseline receipts |
| 51 | Exact loop-1 drafting | **NOT_STARTED** | preregistered exact verification and measured EV; graveyard on failure |
| 52 | Cross-loop wavefront | **NOT_STARTED** | fragmentation/occupancy trace and exactness proof before source |
| 53 | Resident process | **NOT_STARTED** | repeated multi-process load/pool-contention evidence and EV first; owner-only local IPC, never a routable server |
| 54 | Public TaskIR recipes | **NOT_STARTED** | built-in equivalence and security gates first; bounded data-only surface with no tools/network/code execution |
| 55 | Human acceptance audits | **NOT_STARTED** | frozen owned population, preregistered sample/seed, authorized graders, and invalidation rules |
| 56 | Optional language detector | **NOT_STARTED** | licensed versioned data plus locked accuracy/speed evidence before inclusion |
| 57 | NUMA/huge-page/QoS/mmap experiments | **STUB** | remain disabled until a ratified safe FrankenSuite/platform surface exists |
| 58 | Serve and translation | **NOT_STARTED** | intentionally closed until separate product/evaluation decisions are owner-ratified |

The graph covers most planned breadth, but completion counts cannot answer whether these
contracts are satisfied. Eleven closed statuses contradicted later comments, live blockers,
or authority state; this pass reopened all eleven while preserving their prior evidence.
Five genuinely missing owners were then added in §4. Readiness remains evidence-derived,
never inferred from the repaired counts alone.

## 3. The eight broken joints (why no real token can flow today)

1. **Format authority is split.** A writer, reader, receipt, or package cannot be accepted
   while three incompatible v1 byte contracts claim authority.
2. **The production proof graph does not exist.** There is no named `production` feature,
   `check.sh` can report PASS with skipped policy legs, the release graph still contains the
   unconditional upstream Rayon edge, and no current immutable SHA has a DSR receipt.
3. **Conversion is not qualification.** Emitted bytes have not crossed checked reload,
   reconstruction/self-test, strict receipt, two-directory determinism/canonical-publisher,
   and atomic activation gates.
4. **Artifact→engine remains synthetic only, and the current reader shape is incompatible
   with the bridge's memory contract.** The real Nanbeige loader refuses; raw HF safetensors
   are oracle-only. `reader.rs` currently reserves and reads the whole envelope into a
   `Vec<u8>`, so layering decoded weights on it would hold envelope + weights simultaneously.
   `6qz7` therefore needs a preflighted range/section streaming reader (or a separately
   ratified immutable mmap view) before its bounded-RSS bridge can be accepted.
5. **Tokenizer/template semantics are not closed.** Embedded bytes exist, but configured
   special IDs and the artifact copy are incomplete, while trusted controls and untrusted
   document content are flattened before tokenization.
6. **Platform and scheduling authority are absent.** The model-root capability is
   intentionally uninhabited; product inference has no proved real blocking-pool seam,
   sealed asupersync CPU team, aggregate admission, or no-Rayon production closure.
7. **No autoregressive product path exists.** Decode/sampler/scheduler/task modules remain
   scaffolds; `NlpEngine` is a lease/re-entry facade, not yet an NLP engine.
8. **Distribution is absent.** `fnlp pull` and both installers do not exist, GitHub Actions
   remain enabled and auto-triggered contrary to policy, and immutable releases are disabled.

## 4. Bridge and governance Beads

Existing bridge Beads remain useful, but their contracts must follow the authority ordering
above rather than treating source existence as acceptance:

| Bead | Joint | Current contract |
|------|-------|------------------|
| `g6f` | format authority | carry the owner-ratified choice of one v1 contract, recompute fixtures, mark alternatives historical, and block every downstream artifact acceptance until resolved |
| `6qz7` | artifact→engine | first replace whole-envelope buffering with preflighted range/section streaming (or separately ratified mmap), then checked `.fnlpq` → typed bf16/int8 weights with bounded RSS and no raw-safetensors production path |
| `0w9e` + `l44` + `eni` | tokenizer closure | bind all embedded/config/special-map bytes and apply configured control/BOS/EOS semantics before L0/product use |
| `7u3d` | model-root authority | retain refusal until owner/ACL, handle-relative/no-follow, non-reentrant lock, no-replace, same-filesystem, and durability surfaces are ratified |
| `7p1s` | executable int8 profile | consume panels through exact quant algebra under independent preregistered quantized-profile gates |
| `rul7` + `79m` + `xu1` | code proof | DSR-only, occasional clean-SHA check on an explicit no-Rayon `production` graph with no skipped legs |
| `5wty` | robot honesty | honor `convert --robot` with typed versioned stage events and no human/robot stream ambiguity |

Five new owners close the non-duplicative gaps found by this pass:

| Bead | Added owner |
|------|-------------|
| `wdne` | make the retained GitHub Actions workflow inert without deleting history |
| `i2o3` | disable repository Actions and retain pre/post API-state evidence; depends on `wdne` |
| `w9e.1` | enable and verify immutable releases before first model publication |
| `y4w.1` | conditional AA-P1 exact BPE allocation/rescan ladder after L0; span-only rung before any heap |
| `lux.1` | conditional AA-S1 queue-policy qualification from held-out daemon traces; fixed FIFO remains fallback |

The repository-setting tasks are explicit external-authority work, not permission to delete
history or improvise a release during this audit. `lho` now depends on the DSR-authority docs,
disabled-Actions, and immutable-release receipts; AA-P1 depends on `l44` plus the evidence
ledger and preserves the rescan oracle; AA-S1 depends on the real daemon and evidence ledger.

## 5. The spine (authority → first token → first task) — REVISED r2

**The spinal-cord test (campaign north star).** One end-to-end gate, run only after all of S0
and S1 are satisfied: `fnlp generate --greedy -n 64 <fixture prompt>` through the shipping
CLI and a qualified canonical artifact must emit exactly the freshly source-bound oracle's
stable prefix for the same numerics profile. Until then the top-line state is `NOT WORKING`;
fixture presence, synthetic weights, scalar substitution, or `SKIPPED_NO_MODEL` cannot turn it
green.

**Milestone S0 — "One authority and one proof graph."** Resolve `g6f`; reconcile raw versus
framed digests; make the exact `production` graph explicit and Rayon-free (`79m`); make
`scripts/check.sh` refuse missing/skipped legs (`xu1`); then select a quiescent immutable SHA
for one occasional DSR run. In parallel, rerun the verified-source nondeterminism floor and
reseal the current trace manifest. Exit: one byte-format authority, one clean code receipt,
and one current oracle authority—none standing in for another.

**Milestone S1 — "Qualified bytes become one canonical token."** `rsk`/`vk7`/`xmy` qualify
conversion → `6qz7` replaces whole-envelope buffering and streams checked sections into
real weights → `0w9e`/`l44`/`eni` close
tokenizer semantics → `7u3d` supplies a ratified activation root → `7eu`/`17h`/`wlb` supply
the sealed asupersync team and aggregate admission → `w6b7` greedy loop → minimal `k45`
CLI slice. Independent prerequisite lanes may proceed in parallel, but the spinal test runs
only at their convergence. Peak RSS must be admitted as weights + one bounded section/panel +
scratch + KV; never envelope-resident plus a second full weight copy. Exit at n=8, then n=64.

**Milestone S1.5 — "First measured user value."** After the S1 model path exists, a locked
prompt ABI and exact sliced-vs-full lm-head gate may make `classify` and distribution-mode
`sentiment` cheaper first values than unconstrained generation. This is not a shortcut around
S1: one real prefill still requires canonical artifact load, tokenizer/template identity,
native forward, asupersync ownership, and an honest scorecard. No model/self-grading result
authorizes its own quality claim.

**Milestone S2 — "First guaranteed-valid task."** S1 + trusted/untrusted prompt segmentation
+ grammar execution (`95v`, `bw1`, `91j`) + TaskIR (`ngg`) → `bbq` extract. The independent
validator, source-byte verifier, and typed no-result policy judge success; the grammar cannot
grade itself.

**Milestone S3 — "Corpus fabric."** sealed team + aggregate admission + KV slabs + fair
scheduler + prefix cache → bounded batch daemon → spool-first durable jobs. The 176
KiB/token/sequence bf16 KV calculation remains an admission constraint: batch 64 × 8K is
about 88 GiB before other process charges.

**Milestone S4 — "Fast, because measured."** Land SIMD only behind scalar/i64 exactness,
enumerated unsafe-island policy, forced-path tests, and shape/host measurements. On Apple,
autovec competes with SDOT and FEAT_I8MM-gated SMMLA. On x86, exact AVX2 competes with
256-bit and 512-bit VNNI at thermal steady state; AVX-512 capability is never an automatic
routing decision.

**Milestone S5 — "Installable and independently verifiable."** A qualified Generic artifact
is explicitly packaged into fixed parts; an immutable draft release is clean-downloaded and
verified; the exact binary's `fnlp pull` owns compatibility, resume, hashes, reassembly,
native packing, and activation; installers merely verify/install that binary and delegate.
The release remains blocked until repository Actions are disabled, the retained workflow is
inert, immutable releases are enabled, and DSR/offline signature receipts pass.

## 6. Integration gates (each joint gets its own proof)

- **Authority gate:** exactly one active v1 ADR/spec, matching fixture digests, raw/file and
  framed/semantic hashes named distinctly, and a validator that rejects a second authority.
- **Code-proof gate:** the immutable SHA contains the explicit no-Rayon `production` graph;
  every required `scripts/check.sh` leg runs rather than skips; one occasional DSR receipt
  records literal final `PASS|FAIL`. Static hygiene and model/host proof remain separate.
- **Conversion gate:** two clean outputs hash-identically or the canonical-publisher claim is
  narrowed; checked reload, tensor reconstruction, self-tests, strict receipt, durability,
  and activation succeed before the state becomes `qualified`.
- **Bridge gate:** the reader preflights metadata, streams one digest-checked section/range at
  a time, and proves admitted peak RSS; engine-from-artifact ≡ engine-from-source on the
  freshly authorized L2 44+2 fixture set, with named census/identity refusal fixtures.
- **Tokenizer/template gate:** every embedded tokenizer/config/special-map byte matches truth
  pack and artifact; configured control/BOS/EOS IDs are applied; typed trusted/untrusted
  segmentation excludes control IDs while preserving untrusted bytes; L0 is rerun through
  exactly that product path.
- **Asupersync gate:** real blocking-pool handle, single sealed team, no late spawn, bounded
  checkpoints, cancellation/panic drain to actual closure completion, aggregate admission,
  and no Rayon/global-pool escape across the production dependency graph.
- **Decode gate:** N-token greedy transcript ≡ freshly source-bound oracle stable prefix
  under the same profile through the real CLI/artifact path; a fixture-presence test,
  synthetic weights, or scalar substitution cannot satisfy it.
- **Task gate:** extract output validates under the independent validator, verbatim spans
  byte-verify against source, every success is schema-valid, and budget/cancel/resource
  failures remain typed no-results.
- **Distribution gate:** exact fixed-part inventory, clean redownload, pull/reassembly/native
  derivation/activation, installer delegation, DSR signatures/offline verification,
  repository Actions disabled and workflow inert, immutable releases enabled.

## 6b. Process upgrades forced by this audit's evidence — r2

Conversion attempts 1→5 each failed at a NEW joint (hardcoded refusal wiring, half-committed
API, naming grammar at write time, non-atomic failure stub, silent 28-minute logs). The
generalization is a house rule for every integration bead in this plan:

1. **Real-data rehearsal where the contract crosses real data.** Synthetic fixtures
   systematically miss real-scale failure modes (three-pass runtimes, staged-write cadence,
   name-grammar collisions). Artifact-, model-, host-, or corpus-data-path integration Beads
   must include their named real-data run; pure schema, parser, or state-machine Beads use the
   strongest data their own contract requires and are not forced to load a model gratuitously.
   The controller runs build/model gates centrally under the DSR-only discipline; agents never
   build merely to manufacture a close.
2. **Stage-line observability is a contract, not a favor.** The `CONVERT STAGE=… RESULT=…`
   convention (which turned attempt 5 from undiagnosable to self-narrating) extends to every
   long-running surface this plan creates: artifact load, prefill, batch epochs, job
   commit/resume, pull. Robot-parseable, identity-carrying, versioned under j47's contract.
   Validation moves to the EARLIEST stage that can rule (plan-time, not write-time — the
   28-minute lesson).
3. **Fail-closed must also fail-clean.** Attempt 4 left a 0-byte destination stub. Every
   artifact-producing surface (derive, pull, job materialization) must use transaction-owned,
   same-directory `create_new` staging, explicitly sync content and required parent metadata,
   then perform same-filesystem **no-replace** activation under the ratified platform
   contract. A failed run leaves no object at the destination path; retained or quarantined
   staging evidence is allowed only under the named recovery policy and is never active data.
4. **Quantized honesty guard.** The first int8 artifact existing is NOT evidence the int8
   MODEL works: until 7p1s lands and the quantized profile's preregistered L3/L4 budgets are
   measured, no wording anywhere (README, robot health, release notes) may describe the int8
   artifact as validated. The artifact is `converted`; it becomes `qualified` only past its
   gates. This distinction enters the models-state vocabulary (8lx).
5. **Wiring-debt metric.** The prior snapshot estimated roughly 7,900 source lines with no
   in-source consumer; that number is historical, not a current measurement or acceptance
   result. Recompute the reachable production-module census at each milestone. S1–S3 DONE
   requires the named consumer edges and an explained residual consisting only of deliberately
   staged later-phase surfaces; a percentage target cannot hide a critical unwired joint.
6. **Bridge-phase non-goals (scope walls):** no streaming token API before j47 freezes the
   NDJSON contract; no mmap before R0/ej1a rules; no speculative/loop-draft work (P7 cards
   own it); no new dependencies of any kind (the closed universe is load-bearing); no SIMD
   intrinsics before the unsafe-island policy expansion lands with its DSR policy gate.

## 7. Risks & standing blockers

- `g6f` is now the first artifact authority blocker; none of the quarantined v1 records may
  independently authorize writer/reader/converter/package/pull acceptance.
- `6qz7` cannot meet its own memory gate on top of the current whole-envelope `Vec<u8>` reader;
  the streaming/range-reader prerequisite is part of the bridge, not a later optimization.
- `scripts/check.sh` and PASS-shaped test/drivers must adopt typed non-authoritative verdicts;
  missing model, zero artifact, absent policy leg, or scalar substitution never yields an
  unqualified top-level PASS.
- **cwr6** (Zen-3 measurement) and the SUITE-pin ratification remain owner-gated.
- fs_tx production opener is fail-closed by design until the platform surface is ratified —
  the new bead makes ratification+implementation explicit instead of implicit.
- The README stays under `hef` (P6 docs truth pass). This audit did not rewrite it; the release
  truth pass must decide every present-tense-as-spec statement against the then-current
  implementation and evidence rather than inheriting a categorical exemption here.
