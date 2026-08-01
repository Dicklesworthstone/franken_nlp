# AGENTS.md — franken_nlp

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 — THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for confirmation that your understanding is correct. Only then may you execute it.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time.

---

## Branch Policy

- Primary branch is `main`.
- Do not reference `master` in docs/scripts.
- Do not create feature branches unless the user explicitly asks for one.

---

## Project Mission

`franken_nlp` is a **pure-Rust, memory-safe, CPU-hyper-optimized library + one CLI program (`fnlp`, also shipped as `franken_nlp`)** that runs **Nanbeige4.2-3B** with no general ML framework and turns it into a local NLP toolbox. We transform bf16 weights into `.fnlpq` (int8 first, int4 later) and write model-specific kernels for this one model. Plan §5.1/§5.6 is the normative artifact lifecycle: pinned local HF snapshot → canonical Generic conversion (cross-target digest gate or an explicitly narrower canonical-publisher fallback) → fixed 1,957,046,720-byte GitHub Release chunks → release-bound manifest → `fnlp pull` streamed verification/reassembly → native packing → atomic activation. The model weights are **Apache-2.0** — declared by the official model card at the pinned revision, which is the standard Hugging Face license declaration and is settled, not uncertain. Every published artifact/release carries the Apache-2.0 text, Nanbeige attribution, and our modification notice per plan §5.7.

- **Apple Silicon / ARM64** — M4/M5: NEON/SDOT and SMMLA only when the OS actually advertises FEAT_I8MM; autovec remains a measured candidate
- **AMD / Intel x86-64** — Zen 3 Threadripper AVX2 first-class; Zen 4/5 and Xeon compare sustained 512-bit VNNI, 256-bit VNNI, and narrower tiers rather than assuming widest wins

**CPU is the reference implementation, correctness authority, and portable product floor** (most target hosts have no usable CUDA GPU); CUDA is out of scope entirely. The **Apple integrated GPU is an affirmed acceleration track** (owner ruling 2026-07-31): Metal via `ft-kernel-metal` lands after the CPU core is parity-proven, prefill first, under its own named numerics profile and the same L-ladder gates; it never becomes a requirement.

### What we stand on (the closed dependency universe)

- `/dp/frankentorch` (`ft-kernel-cpu`, `ft-core`, `ft-serialize`) — consumed at the serial/range-leaf kernel level. At the audited pin, dynamic int8 linear has SDOT/AVX-512-VNNI/scalar but no AVX2/SMMLA, f32 SDPA is dense per-head rather than native 48:8 GQA, and `ft-kernel-cpu` makes Rayon unconditional. Phase 0 must expose the required no-spawn leaves behind a no-Rayon suite surface; current Rayon-backed entrypoints are never production scheduling. Read plan §3.5 before claiming reuse.
- `../asupersync` — the execution foundation: structured orchestration, cancellation, budgets, IO, bounded admission, and CPU-team lifecycle through `Cx::spawn_blocking` + whole-request/batch-epoch `Cx::scoped_cpu`. At the audited pin, its Rayon entry is dev-only and does not enter the release graph. Leaf kernels perform math but never spawn. Its HTTP/TLS client (`tls-webpki-roots`) is the ONLY networked Rust/product code path (`fnlp pull`); no reqwest/hyper, ever. The explicit out-of-band `scripts/fetch_model.sh` / `scripts/fetch_model.ps1` provisioning tools may use system `curl` / PowerShell web APIs to fetch the pinned upstream source closure; they are not inference or library code.
- `/dp/frankensqlite` (`fsqlite`, `fsqlite-types`) — optional metadata/job-state history, disabled by default (NEVER `rusqlite`; fsqlite tables never contain document/prompt/output text). Durable job content spools are separate, explicit, owner-only files governed by plan §8.4.
- The **complete** direct non-suite release allowlist is `clap`, `serde`/`serde_json`, and `sha2`, each only for the layer named in plan §3.4. Everything else must be `std` or a pinned FrankenSuite surface. **No `thiserror`, `anyhow`, `ctrlc`, `rayon`, `half`, `memmap2`, `uuid`, `num_cpus`, allocator crate, `tokenizers`, `tiktoken`, `minijinja`, or `llguidance`. Rayon is forbidden from the entire FrankenNLP release graph, not merely as a direct dependency.** Tokenizer/template/grammar are ours; the tokenizer bytes (Apache-2.0, attributed) are embedded in the binary and hash-checked against the artifact's copy at load.

Project source license is the exact text in `LICENSE` (MIT + OpenAI/Anthropic Rider); the model weights remain **Apache-2.0, settled by the pinned model-card declaration**. Attribution is factual provenance (model name, pinned revision, "author attribution per the pinned card: Nanbeige Team") carried in artifacts, manifests, `fnlp licenses`, and the one-line `fnlp --version` attribution — never a synthesized copyright notice the upstream never published (plan §5.7). `swiss_army_llama` is the project owner's own prior work — reuse its methods/presets freely with attribution.

**The single source of truth for what we are building and why is [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md).** Read it before writing any kernel.

### What this model is (one paragraph)

Nanbeige4.2-3B is a dense decoder-only Llama-family transformer with a loop. The pinned source/index currently show: 22 physical layers; two passes; KV slot `layer + loop×22`; **final RMSNorm after each pass, so loop 2 consumes loop-1 normalized output**; 44 layer outputs plus two post-loop norm states. Dimensions: hidden 3072, 48 query / 8 KV heads, explicit head_dim 128, SwiGLU 10752, eps 1e-5, split-half RoPE θ=7e7, max positions 262144, vocab 166144, untied lm_head, no bias/window. The 201-name index contains no mHC/depth/LoopSplit/ngram tensors. These are `[OBSERVED@pin]`, not implementation authority until archived under `docs/truth-pack/`. Tokenizer is SentencePiece BPE on the slow reference path; the template uses im/think markers and XML/JSON tool branches.

---

## Product Shape

The project must be both:
1. A reusable Rust library (`NlpEngine`) for embedding the model + task layer, **synchronous and blocking** — the async runtime is an owned implementation detail.
2. A standalone CLI program, shipped under the short binary name `fnlp` and long binary name `franken_nlp`, with:
   - **robot mode** (agent-first, versioned NDJSON, self-describing `robot schema`)
   - human mode (`fnlp extract|ner|classify|sentiment|… <file>` → JSON/text, `--json` everywhere)
   - **batch daemon mode** (`fnlp batch`: NDJSON docs on stdin → NDJSON results on stdout, bounded queues, weights loaded once)
   - opt-in **durable job mode** (`fnlp job`: manifest → journal/resume/verify/materialize with no content persistence by default)
   - user-owned **Assay surfaces** (`eval`, `calibrate`, `qualify`) and a bounded data-only recipe compiler only after its Phase-5 gates

Input: text files/stdin/NDJSON. No Python or foreign ML/runtime ABI in the default inference path, no inference network, no GPU required. Local audited islands are SIMD and opt-in mmap only. NUMA/huge-page/topology/QoS experiments require a ratified safe FrankenSuite surface or remain disabled; do not reach through to `libc`, `sysinfo`, or another transitive crate. The release uses the system allocator.

---

## Porting Workflow (Spec-First)

Implementation follows spec documents, not ad-hoc copying. Read in this order:
1. [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md) — the master plan.
2. The **research-decision register (plan §14)** — observed rows need truth-pack promotion; partial/open/blocked rows gate only their dependent surface. Empirical questions must be measured in their named phase, not “answered” from prose.
3. The **reference sources** (pinned + hashed in `docs/truth-pack/`): `modeling_nanbeige.py` (the primary loop/KV/attention semantics), `configuration_nanbeige.py` (the defaults that decide which features are active), `tokenizer_config.json` (the verbatim chat template), `tokenizer.model`, a tested official `ggml-org/llama.cpp` revision at/after Nanbeige support commit `b77d646…` (secondary CPU/GGUF differential and performance baseline), and the authors' `Nanbeige/llama.cpp` `nanbeige42` fork (historical lineage, not an independent oracle). GPL-3.0 `rlx-nanbeige` may be exercised only out-of-tree as a black-box differential; never copy or depend on it.

**Hard rule: no surface ships against an unresolved dependency.** Promote source observations only with immutable hash + line span + replay fixture.

---

## The franken_nlp Engineering Doctrine (READ THIS BEFORE OPTIMIZING)

Load-bearing, non-negotiable rules distilled from the plan and from the frankentorch / frankensearch / franken_ocr prior art. Violating any of them has burned real days of work before.

1. **Correctness outranks speed, always (G1 > G2), with the comparison surface named.** Parity gate FIRST, perf second. A faster backend that violates the canonical scalar comparison contract or changes greedy tokens for the **same artifact recipe/numerics profile** is reverted—no source landed—and memorialized in `docs/NEGATIVE_EVIDENCE.md`. Integer stages are exact; each floating stage has a named metric/tolerance contract. Fidelity claims are **profile-scoped** (plan §1.1.1): the pinned oracle executes bf16 with explicit cast points, so `hf-bf16-eager` owns HF-match claims, `diagnostic-f32` is the structural bisect oracle and is *not* required to token-match bf16 (flips become named fixtures), and quantized profiles carry preregistered logit/argmax/token/task budgets. Every scoring surface names its probability space (plan §6.10); every cache/receipt key derives from the one `ExecutionIdentity` (plan §4.3.1). Deliberate int8/int4 recipe divergence from bf16 is a separate, preregistered decision and never inherits a blanket BF16-token-identity claim. We ship speed on top of a fixed recipe's semantics, never instead of them.

2. **Valid-by-construction task output (G8) has the same rank.** Every successful structured response validates; budget/cancel/resource failure is typed no-result, never partial success. An unconstrained “retry on parse failure” loop is a design bug.

3. **The loop is the architecture.** Decode logically visits the decoder stack twice; physical DRAM/cache traffic is measured, not inferred. KV is 44-deep; execute `22 layers → final RMSNorm → 22 layers → final RMSNorm`. The L2 ladder has 44 layer outputs **plus two norm states**. Never size/schedule as a conventional 22-layer model.

4. **`head_dim` is 128, not `hidden/heads` = 64.** The config overrides the Llama fallback. q_proj is 3072→6144, k/v are 3072→1024, o is 6144→3072. Any kernel, buffer, or RoPE table built on 64 is silently, catastrophically wrong.

5. **Do not hand-roll wide SIMD for glue without a profile.** Explicit SIMD belongs first in proved int8 MAC kernels. Polynomial exp/sigmoid is only a default-off candidate here; sibling wins are priors, not this project's measurements.

6. **Measured-faster wins; hardware capability is not a routing decision.** franken_ocr shipped Apple Silicon defaulting to LLVM-autovec over SDOT/SMMLA for some dense int8 shapes because that is what measurement said. Dispatch tables record measured choices per shape; forced paths remain for proof runs.

7. **AVX2 is first-class and exact.** Raw `vpmaddubsw` can saturate inside the instruction; accumulation cadence cannot fix it. Implement/measure both plan §6.5 candidates: low-7/high-bit decomposition with proved i16 pair bounds and the widened signed-i16 route. Both equal scalar/i64 over the full i8 domain; raw saturation is not a shippable approximate mode.

8. **Asupersync owns every request region and compute team; leaf kernels never spawn.** Each admitted request or finite batch epoch has one owned asupersync region/child `Cx`; its feeder, timers, wrapper/output tasks, and drain are not detached. The pinned blocking contract has one explicit physical-lifetime exception: an already-running pool closure may outlive its cancelled wrapper, so `EngineResources` tracks that closure to an actual-completion latch and treats it as outstanding drain work. The run crosses `Cx::spawn_blocking` once and uses one `Cx::scoped_cpu` region spanning the whole request or bounded epoch, with checkpoints at tile/morsel boundaries. Production must prove its `Cx` carries a real blocking-pool handle; the pinned inline fallback is lab/test behavior, not a valid engine route. The pinned `ScopedCpu::spawn` enforces the numeric cap but does not itself reject a spawn after its drain latch rises, so the team-formation protocol must create the fixed team before releasing any worker into fallible work, seal spawning forever, and test that no post-start/post-latch worker can appear. Never enter the seam under an engine lock, recursively create a team, detach or lose supervision of a semantic-request task/closure, or nest an asupersync runtime. Concurrent sync callers enter bounded admission or a compatible batch, not independent forwards. Rayon is absent from the release graph. Re-entrant calls fail typed/fast; cancellation/panic joins and scheduler lifecycle are model-checked/replayed, not just watchdog-tested. Admission/memory/output guards remain owned by the blocking closure until its latch fires after the scoped team joins—never release or reuse them merely because the wrapper task resolved cancelled. The capacity certificate inventories the entire process at once: runtime workers + maximum concurrent blocking coordinators + scoped CPU children + separately ratified service/helper threads. A preset name is not a safe thread count; in particular, never pair `high_throughput()` blindly with a physical-core-wide scoped team. Use the foundation's depth, not just its scheduler: capability rows (`[SPAWN, TIME, RANDOM, IO, REMOTE]`, monotone `Cx::restrict`) make no-spawn leaves, no-network inference, and RANDOM-once sampling admission compile-time properties; budgets are typed meet-composed children with a reserved cleanup budget, never ambient milliseconds; two-phase reservations are tracked obligations under an explicit leak-response policy (panic in lab/DSR validation, logged escalation in production); and the async half of every scheduler/daemon/job/pull proof runs on the deterministic lab (oracles, seed-bound chaos, crashpack replay) while the bounded native team-state model covers the `scoped_cpu` OS-thread half — plan §3.3, §9.3, §9.5, OQ-35.

9. **Determinism is scoped.** Semantic greedy output is exact under a named numerics profile; canonical bytes additionally require ordered output and scrub volatile metadata. Batch/prefix equivalence requires the same per-row reduction order. Never promise byte identity for completion-order NDJSON or approximate transcendental modes.

10. **Integer bounds cover every intermediate.** At K=10752: S8×S8 ≤176,160,768; raw U8×S8 ≤350,945,280; correction ≤176,160,768; conservative raw+correction ≤527,106,048. Prove every model K, int4 group, row sum, tail, and SIMD tier against i64.

11. **Honest, measured everything.** Every accepted numeric divergence → `docs/DISCREPANCIES.md` (reference behavior, ours, measured impact, review date, and a rollback mechanism that actually exists: a kernel selector, builder/CLI option, or activation of a prior immutable artifact — weight-quantization stages are immutable artifacts per plan §5.1, so no env var may claim to restore bf16 weights that are not in the file). Every rejected optimization → `docs/NEGATIVE_EVIDENCE.md` (the 5-pass loop; losers reverted with NO source landed). Gauntlet comparisons use thread/allocator/precision fairness controls; slower rows stay in the ledger. Task-quality claims name their dataset, prompt hash, recipe id, and thinking mode. No silent numerics changes, ever.

12. **Two binaries from one entrypoint:** `fnlp` (short) + `franken_nlp` (long). Shared dispatch is `pub fn cli_main() -> ExitCode` in the lib; each binary is a **thin one-line shim in its own file** (`src/main.rs`, `src/bin/fnlp.rs`) — never the same `path` in two `[[bin]]` targets.

13. **Presets are data, prompts are versioned.** Task prompts/presets live as embedded data files; every task response carries the prompt-template hash so quality regressions are attributable to weights vs kernels vs prompts. Changing a prompt is a versioned, diffable event with an eval re-run, not a drive-by string edit.

14. **AVX-512 width is not destiny.** Benchmark sustained zmm-VNNI, ymm-VNNI, AVX2, and scalar per host/shape with p50/p95/p99, clocks/energy where available, and thermal steady state. Dynamic M/T tails still exist; firmware/VM feature exposure is authoritative.

15. **Memory admission precedes allocation and is process-aggregate.** BF16 KV is 176 KiB/token/sequence: 8K is ~1.38 GiB and batch 64 is ~88 GiB. One `EngineResources` ledger charges weights, KV, scratch, caches, jobs, and staging across every engine; per-engine limits cannot each promise the same RAM. Use two-phase reservations, lazy COW pages, and checked worst-case admission. Reported 256K context is not a usability promise.

16. **License compliance is mechanical, not conditional.** The model is Apache-2.0 per the official card's declaration at the pinned revision. Every published artifact/release carries the license text, attribution, and modification notice (plan §5.7), verified by the byte-compare release test. Do not reintroduce fail-closed "license uncertainty" language — the owner has ruled it settled.

17. **Compile known structure; never approximate it.** Stencil may skip illegal lm_head rows only when it evaluates every legal row; forced tokens still run all 44 effective layers; continuation tries never prune a candidate. Sparse=full, forced=sequential-KV, and trie=naïve are exact gates with universal fallbacks.

18. **Constraint pressure is not confidence.** Legal mass and legal-vs-illegal margin require full projection and are `not_computed` on sparse paths. Override/margin/mass can diagnose model/grammar tension, but cannot be called a fabrication detector or authorize acceptance without task-specific calibration.

19. **Durability has an authority boundary.** An owned checksummed spool/materializer may promise one canonical committed record per item. Arbitrary stdout is at-least-once replay with stable ids. Job persistence is opt-in and metadata-only by default; no hidden input/output retention.

20. **Task extensibility is data-only and bounded.** Built-ins compile through `TaskIR` first. Public recipes may never execute code/tools, access the network, interpret a scripting language, read undeclared files, or bypass budgets/provenance/calibration. If the equivalence/security gates fail, the public recipe surface stays closed.

21. **Forced bytes are not forced tokens.** Stencil may bypass token selection only when exactly one token id is legal. A unique byte continuation can have multiple tokenizations with different KV/hidden states; byte jump-forward, retokenization, and token healing are not exact substitutes. If unique-token runs are rare, record negative evidence and keep sequential feeding.

22. **Type the untrusted-document boundary without claiming a firewall.** Only trusted template code may emit role/think/tool control ids. An untrusted segment must decode to the original bytes while excluding every control id, or reject. This proves marker containment only; prose steering remains an empirical per-task attack-success metric, and all derived output remains untrusted data.

23. **A second model pass is not independent proof; a sample is not authority by itself.** Same-model semantic verification stays off until it shows incremental locked-eval value and always reports correlated provenance. Corpus acceptance requires a frozen owned population, preregistered design, and human-authorized grades; model self-grading and post-hoc sample changes cannot authorize a claim.

24. **Local tuning retains defaults unless evidence is durable.** `fnlp tune` may select only already-proved bit-identical fixed-shape kernels and conservative thread caps. Profiles are validity-domain keyed; transient batch/page-cache/NUMA choices are not machine constants. Warm, repeated A/B trials must clear both statistical and practical thresholds, or shipped defaults remain.

25. **Cache authority follows dependency scope.** Every TaskIR stage is `ItemLocal`, `PartitionReduce`, or `CorpusGlobal`. Adding an item may reuse exact local maps but changes the child-set/snapshot authority for reduce/global outputs. Entity clusters are snapshot-qualified; never return a stale global result because an individual document hash was unchanged.

26. **Fork admission prices actual 44-deep tail bytes.** A 16-token bf16 KV page is about 2.75 MiB. Continuation tries and prefix forks must account for page fill/fan-out before allocation and choose among small tail slabs/recompute, breadth, depth-first, short-ID, or naïve exact execution. “Bounded” without a usable byte certificate is not a design.

27. **A resident process is research, not permission to add a server.** AA-R1 begins with evidence of repeated multi-process loads/pool contention. Any experiment is owner-only OS-local IPC over the versioned framed contract, with no routable listener or remote credentials, and retains the in-process/batch fallback.

28. **Source acquisition, release packaging, and client installation are different trust zones.** `scripts/fetch_model.sh` / `scripts/fetch_model.ps1` download the immutable upstream conversion closure; they never masquerade as the end-user installer. Research-only truth-pack evidence does not become a converter prerequisite. The public artifact is the canonical Generic `.fnlpq`, versioned independently from the binary and split at exactly 1,957,046,720 bytes except the tail. Cross-OS/ISA conversion must hash-identically or the release names one canonical publisher target and narrows the reproducibility claim. Never publish from an unreceipted converter output, upload with a wildcard/`--clobber`, or rewrite bytes under an existing tag/name.

29. **One artifact manager owns every client model download.** `install.sh`/`install.ps1` install the binary, then invoke that exact installed binary's `fnlp pull`; shell and PowerShell must not parse the model manifest, concatenate parts, invent cache filenames, or duplicate integrity logic. Default pull trusts only the release-bound embedded manifest; local/private manifests require an expected digest obtained through an independently trusted channel. SHA-256 proves byte identity, not publisher identity; the DSR release job also publishes an SBOM, SLSA provenance, and project-signature bundle binding the binaries and manifest/receipt/model inventory. The signing-key fingerprint must arrive through an independently trusted project channel; a signature and key fetched together from one untrusted mirror prove only self-consistency. Owner-controlled roots, a non-reentrant content lock, `create_new`, no symlink/reparse/device targets, sync/same-filesystem activation, native-pack differential, and honest durability status all pass before the new model becomes visible.

30. **Claims and receipts are typed evidence.** Public numeric/superlative claims must map to `docs/CLAIMS.json`; intentional reference behavior belongs in `docs/BEHAVIOR_NOTES.md`; `.fnlpr` receipts declare `Replayable`, `StructuralReplay`, `VerifiableIfArtifactsSupplied`, or `AuditOnly`, and omit private bytes unless retention is explicitly authorized. Structural cost witnesses distinguish a full target invocation (two loops / 44 layer executions per position) from any AA-D1 loop-1 draft invocation, and count lm_head rows plus KV bytes. “Verified” without scope, validity domain, and evidence grade is forbidden.

31. **Asupersync authority, budgets, and outcomes stay typed.** Narrow each child `Cx` monotonically: pull gets only the spawn/time/IO and TLS entropy its pinned path proves necessary, never REMOTE; ordinary inference gets no network/remote authority; unseeded sampling may draw entropy once at admission, while greedy and kernel leaves get no RANDOM; leaves receive `cap::None` or the ratified checkpoint-only view and cannot spawn. Typed child budgets may only tighten parent deadline/poll/cost and memory/CPU/IO/cleanup/artifact envelopes; request token/output/page/byte counters remain explicit, and drain keeps reserved cleanup budget. Preserve `Outcome::{Ok,Err,Cancelled,Panicked}` until the library/CLI policy boundary. LabRuntime proves region/obligation/cleanup/quiescence properties, not every native `scoped_cpu` OS-thread interleaving; pair it with the bounded team-state model and hostile native stress.

32. **Asupersync names are exact contracts, not permission to infer stronger semantics.** At the pinned revision, `combinator::first_ok_outcomes` classifies an already-completed outcome vector in order; it does not execute futures. `ExecPlan::first_ok` drives every child concurrently to completion and only then selects the first successful result in input order; it neither short-circuits nor cancels/drains losers. Neither surface is the desired sequential mirror fallback, so `fnlp pull` uses an explicit ordered `for`/`await` loop with per-attempt child budgets. Any concurrent hedge needs a separately ratified first-success primitive, duplicate-work/byte budget, and cancel-then-drain proof for every loser. `bracket` is useful on the normal awaited path, but its drop-time release is bounded best-effort; durable artifact cleanup remains RAII + explicit await + journal/recovery + atomic activation. Required state transitions, output delivery, and durable spool/journal writes use `GenServer::call` or reserve→permit plus an explicit processing/commit acknowledgement; `cast().await` proves enqueue only, and `try_cast` applies its declared overflow policy. Preserve all eleven pinned `CancelKind` variants (`User`, `Timeout`, `Deadline`, `PollQuota`, `CostBudget`, `FailFast`, `RaceLost`, `ParentCancelled`, `ResourceUnavailable`, `Shutdown`, `LinkedExit`) until the policy boundary. Lab's DPOR-style explorer supplies bounded guided coverage, not exhaustive proof; a TLA+ export is a TLC input, not a model-check result. A bounded-model-check claim requires retained TLC version, config, command, result, property scope, and counterexample when one exists.

---

## Alien-Artifact Engineering Contract

Start with a deterministic baseline and measured hotspot. A candidate needs `EV=(Impact×Confidence×Reuse)/(Effort×AdoptionFriction)≥2`, explicit state/actions/loss/units, assumptions, p50/p95/p99, proof/tolerance authority, interaction/composition check, provenance, and fallback. Runtime adaptation follows offline replay → shadow → canary → explicit default; at most one controller per subsystem. Use CVaR alone for quality-tail optimization; do not stack tail jargon. Exact loop speculation or graveyard it. Cross-loop wavefront work requires a fragmentation trace before code. Every implemented card carries `env.json`, `manifest.json`, `repro.lock`, and `LEGAL.md` when external IP/data may matter; it records primary-source review status and a fixed search/profile budget whose exhaustion means no promotion. See plan §10.5 cards AA-K1/N1/L1/B1/Q1/S1/W1/C1/T1/D1/U1/A1/R1 and §10.6's rejected/deferred ideas.

---

## Code Editing Discipline

### No Script-Based Changes
**NEVER** run a script that mass-edits code files. Brittle regex transforms create more problems than they solve. Make code changes manually (use parallel subagents for many simple changes; do subtle/complex changes methodically yourself).

### No File Proliferation
Revise existing files in place. **NEVER** create `mainV2.rs` / `nn_improved.rs` / `decoder_enhanced.rs`. New files are reserved for genuinely new functionality; the bar is incredibly high.

---

## Backwards Compatibility

We are in early development with **no users**. Do things the **RIGHT** way with **NO TECH DEBT**. Never create compatibility shims or wrappers for deprecated APIs. Just fix the code directly.

---

## Toolchain

- Rust 2024 edition. Nightly toolchain (`rust-toolchain.toml`) — **required** for `stdarch` i8mm/dotprod intrinsics and `portable_simd`.
- `[lints.rust]` sets both `unsafe_code = "deny"` and `unsafe_op_in_unsafe_fn = "deny"`; crate roots also use `#![deny(unsafe_code)]`—**not `forbid`**, which cannot be locally overridden. Only enumerated SIMD/mmap modules may opt into scoped `#![allow(unsafe_code)]`; they never allow `unsafe_op_in_unsafe_fn`, so every unsafe operation remains inside an explicit `unsafe {}` block with a local `// SAFETY:` proof. Integer kernels exactly differential-test against scalar/i64; floating candidates use named tolerances; mmap uses separate range/lifetime/immutability tests. NUMA, huge-page, topology, and QoS experiments must use a ratified **safe FrankenSuite surface** or remain off; they are not permission for another local unsafe island or direct transitive dependency. A policy test rejects any unlisted island or lint relaxation.
- Cargo only. Persistence via `fsqlite`, never `rusqlite`. Direct non-FrankenSuite release dependencies are frozen to `clap`, `serde`/`serde_json`, and `sha2`; a written justification is not enough to add another—change requires explicit owner approval and a plan revision.

---

## Mandatory Checks After Substantive Changes

These commands describe the validation inventory implemented by
`scripts/check.sh`; they are **not** pane-level instructions. Ordinary swarm
panes run no Cargo, RCH, DSR, or GitHub Actions command. After code-first
contention quiesces, the controller selects one clean immutable commit and may
request an occasional DSR checkpoint that runs `scripts/check.sh` against the
explicitly named `production` feature graph. Direct Cargo, direct RCH, and
GitHub Actions output are diagnostic only and cannot authorize a close or
release.

```bash
cargo fmt --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
ubs --diff
```

If the controller's DSR checkpoint reports a failed leg, fix the root cause in
the next code-first wave; only the controller requests the next DSR run.

### The `cargo test --locked` gate (green-bar requirement once the crate exists)

After `Cargo.toml` and `scripts/check.sh` exist, the locked test leg is a hard
gate and must exit 0 in the retained DSR checkpoint before a code-complete bead
is closed. Before Phase 0 scaffolding, report Cargo gates as **not applicable
(crate absent)** and run only truthful document/link/whitespace/UBS checks;
never create a fake empty crate merely to manufacture a green bar. The
controller-authorized DSR recipe invokes `scripts/check.sh` as its single
validation entrypoint and records the clean source SHA plus the exact
`production` feature graph. Until that graph is explicitly selected by the
script/recipe and the receipt says exactly `PASS` or `FAIL`, build authority is
**BLOCKED** rather than inferred from the default or all-features graph.

Build-surface note: both binaries compile from thin shims over `cli_main()` (doctrine #12); `cargo check --locked --all-targets` MUST be free of the "present in multiple build targets" warning.

---

## Testing Policy

Each module includes unit tests for happy path, edge cases, error handling. Beyond that, the conformance ladder is the heart of the project (plan §9):

- **Reference oracle**: pinned CPU HF environment running `NanbeigeForCausalLM` (`trust_remote_code`, `use_fast=False`) via `scripts/gen_reference_fixtures.py` — **establish the oracle's own nondeterminism floor first** with at least five independently launched processes at each of at least two preregistered thread counts; if anything varies, use the preregistered additional-repetition/convergence rule and freeze only the exact stable prefix. Never derive a tolerance distribution from two observations.
- **Independent-enough checks**: a deliberately simple in-repo scalar f32 specification engine localizes failures; a tested official post-`b77d646…` llama.cpp revision is the secondary CPU/GGUF differential and performance baseline. The authors' fork is shared lineage, not another vote; RLX remains an out-of-tree GPL black box only.
- **Parity ladder L0–L5**: L0 exact; L1 same-recipe integer exact plus named floating metric vectors (not cosine alone); L2 all 44 layer outputs + two loop norms; L3 measured per-profile tolerance; L4 `hf-bf16-eager` greedy exact on oracle-reproducible prefixes, `diagnostic-f32` under its structural/logit contract, and optimized-vs-scalar greedy-token exact for one fixed recipe/profile; quantized-vs-bf16 token agreement is measured, not presumed; L5 task semantics.
- **Invariant gates**: scoped determinism, batch/prefix equivalence, independent structured-output/source-grounding validation, untrusted-segment bytes preserved with all control ids excluded (or rejection), sparse=full, forced=sequential KV, every trie traversal=naïve, uninterrupted=resumed owned job, dependency-scope snapshot mutation, integer SIMD ≡ scalar/i64, overflow/fork-tail/admission proofs, a real `spawn_blocking` pool handle with executor-heartbeat proof (no production inline fallback), one admitted `scoped_cpu` team with exact coordinator/child width, bounded checkpoints, disconnect-safe cancel/panic full join, wrapper-cancel versus actual-completion lifetime with no untracked region-to-supervisor gap, re-entry refusal, no per-op thread creation, no-spawn leaves and no Rayon in the release graph, one-time resource-host config conflict plus cgroup/job-object-aware or explicit memory authority and aggregate multi-engine reserve/commit/abort accounting, scheduler model/interleaving tests, cache privacy, source-manifest verification, conversion-twice plus cross-target identity-or-fallback, split/reassembly identity, streamed part/whole verification, hostile-root/atomic-output safety, resume/concurrent-pull safety, installed-basename discovery, activation-fork fail-closed behavior, and claims/behavior/receipt/structural-cost registry consistency. The scheduler/cancellation portion executes as deterministic asupersync Lab runs (quiescence, obligation-leak, loser-drain, and cancellation-protocol oracles; seed-bound chaos; crashpack replay artifacts) combined with the repository's bounded native team-state model per plan §3.3/§9.5 — a failing interleaving must replay from its seed, never survive as a one-off batch-verification ghost.
- **Task evals** (plan §9.6): named, versioned datasets per task; scorecards carry recipe id + prompt hash + thinking mode. Quality regressions gate releases like parity regressions.
- **Adversarial-content and evidence gates**: matched clean/attack fixtures report task-specific steering rather than a firewall claim; semantic second-reader uplift is measured against labeled fields; acceptance-audit arithmetic uses exact small-population goldens and requires human grades.
- **Model-gated e2e**: an unarmed DSR code checkpoint reports
  `SKIPPED_NO_MODEL`; release certification separately requires an armed pass
  with artifact digest and fallbacks pointed at `/nonexistent`. A public
  model-asset release additionally proves fresh HOME/LOCALAPPDATA installer →
  `fnlp pull` → no-flag inference, followed by a byte-perfect cache hit whose
  inference run opens no network. A green code checkpoint never promotes this
  model gate by implication.
- **`many_docs_without_deadlock`** concurrency watchdog.

---

## Agent Ergonomics Requirements

Robot mode must be: stable versioned schema, deterministic where possible, explicit exit codes, line-oriented NDJSON, easy to pipe. Do not mix human decoration with machine output in robot mode. `robot schema` self-describes the contract; a contract test validates emitted events against a frozen JSON schema fixture. Stable exit codes are documented in `error.rs` (plan §8.5). stdout is data, stderr diagnostics; bare `fnlp` prints help, never a TUI; honor `NO_COLOR`, `CI`, `TERM=dumb`.

---

## Session Completion ("Landing the Plane")

Before finishing a work session you MUST:
1. Once `.beads/` is initialized **after the plan's ≥4 review rounds**, file issues for remaining implementation work. During pre-Beads planning, do not initialize Beads early; record review gaps in the plan/handoff.
2. Ordinary panes run only the explicitly permitted cheap, non-Cargo hygiene
   checks and record that compilation/tests remain pending; the controller
   schedules the occasional clean-SHA DSR checkpoint for tests, clippy, fmt,
   and the combined `ubs` leg.
3. Update the handoff/status truthfully; ordinary panes leave work
   `in_progress`. Only the controller closes after the retained DSR receipt and
   every bead-specific gate exist.
4. If Beads exists, `br sync --flush-only` and stage only the intended `.beads/` changes.
5. Hand off — summarize what changed, gates run + results, remaining risks/gaps, concrete next steps.

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer for agents to coordinate via MCP tools/resources: identities, inbox/outbox, searchable threads, advisory file reservations with human-auditable Git artifacts.

- **Register identity:** `ensure_project(project_key=<abs-path>)` → `register_agent(project_key, program, model)`.
- **Reserve files before editing:** `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-###")`.
- **Communicate with threads:** `send_message(..., thread_id="br-###")`, `fetch_inbox`, `acknowledge_message`.
- **Prefer macros:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`.
- Common pitfalls: `"from_agent not registered"` → `register_agent` in the right `project_key` first; `"FILE_RESERVATION_CONFLICT"` → adjust patterns / wait / use non-exclusive.

---

## Beads (br) — Dependency-Aware Issue Tracking

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`). Issues live in `.beads/` and are tracked in git. **`br` is non-invasive — it NEVER runs git.** After `br sync --flush-only`, manually `git add .beads/ && git commit`.

```bash
br ready                 # issues ready to work (no blockers)
br list --status=open
br show <id>             # full detail with dependencies
br create --title="..." --type=task|bug|feature|epic --priority=2   # 0=critical..4=backlog (NUMBERS)
br update <id> --status=in_progress
br close <id> [<id2> ...] [--reason "..."]
br dep add <issue> <depends-on>
br sync --flush-only     # export to JSONL (NO git ops)
```

Conventions: use the bead ID (e.g. `br-123`) as the Agent-Mail `thread_id` and prefix subjects with `[br-123]`; put the issue ID in the file-reservation `reason`; include `br-###` in commit messages.

---

## The Implementation Swarm — Code-First / Batch-Verify Strategy (NTM)

Implementation of this project runs as an NTM tmux swarm of **12 codex panes on `gpt-5.6-terra` at `xhigh` reasoning effort**, governed by `/ntm` and `/vibing-with-ntm`, after the final plan has been converted to beads via `/beads-br`, `/beads-bv`, and `/beads-workflow`. Ordinary panes run **no Cargo, RCH, DSR, or GitHub Actions commands**. After a code-first wave has quiesced, the controller may select one clean immutable SHA for an occasional `/dsr` checkpoint. That DSR job is the sole build/release authority and runs `scripts/check.sh` against the explicitly named `production` graph. The swarm follows the **Code-First / Batch-Verify** methodology below. It exists to defeat build contention; every agent in the swarm must know it and follow it.

> Operational note: NTM's codex launch template pins the reasoning effort. Before spawning this swarm, set the template's `model_reasoning_effort` to `xhigh` (and restore it afterward), then `ntm spawn franken_nlp --cod=12:gpt-5.6-terra --no-user`.

### The core problem it solves

A swarm of N agents sharing one repo can make builds, rather than coding, the
bottleneck. Two conditions make uncoordinated per-bead building especially
costly here:

1. **The planned crate is broad.** `franken_nlp` is a single crate by design (plan §4.1), but this greenfield repository has no measured clean-build duration yet. Record that duration once a real crate exists; do not repeat “takes minutes” as fact before then.
2. **The DSR build hosts have finite capacity.** The controller checks the DSR
   host/config posture before a checkpoint and dispatches only after ordinary
   editing has quiesced. Direct RCH output and GitHub Actions status may be
   retained as historical diagnostics, but neither is current proof authority.

### The insight

Separate the cheap, parallelizable work (writing code) from the expensive,
serialized work (building/testing). Reading and writing code is parallel across
12 agents; building is the scarce resource. Let the panes write and commit
without building, then have the controller submit one DSR checkpoint for the
selected clean SHA and accumulated proof batch.

### The two-phase loop

```
┌─ PHASE 1: CODE-FIRST WAVE (all 12 agents, parallel, no builds) ─┐
│  each agent:  claim a ready bead (via bead-assignee)            │
│               → WRITE the real code + test                      │
│               → cheap non-Cargo checks only                      │
│               → COMMIT immediately ("…— code-first, batch-test  │
│                 pending"), leave bead in_progress               │
│               → next bead                                       │
│  diagnostic = commit flow + ready-pool depth (prior campaign:   │
│               ~20–40 commits/10 min at full wave — a historical │
│               reference point, not this project's target)       │
└─────────────────────────────────────────────────────────────────┘
                              │  (quiescence + meaningful proof batch)
                              ▼
┌─ PHASE 2: OCCASIONAL DSR CHECKPOINT + CLOSE (controller) ──────┐
│  1. commit-flush; wait for editing/build contention to quiesce │
│  2. require a clean tree and select one immutable source SHA    │
│  3. ONE DSR job runs `scripts/check.sh` for graph `production`  │
│  4. retain exact DSR PASS|FAIL receipt; triage/re-run if needed │
│  5. CLOSE only beads whose own gates and combined proof passed  │
└─────────────────────────────────────────────────────────────────┘
                              │  closing beads UNBLOCKS dependents
                              ▼
                    ready pool refills → next Phase-1 wave
```

(This project is one crate, so "the union of touched crates" collapses to the
crate's full locked test suite. `cargo check` is still a real whole-crate
compilation—not “syntax only”—so an ordinary pane never runs it. The named
`production` aggregate must include the actual shipping asupersync runtime and
no-Rayon leaf path; default-empty and all-features graphs are not substitutes.)

### Why the close step is the engine that refills work

This is the subtle, critical part. `br` unblocks a dependent bead only when its blocker is **closed**, not when it is committed-but-in_progress. During a code-first wave, agents commit but deliberately leave beads in_progress, so:

- The ready pool drains (claimed → in_progress) and does not refill; the blocked beads stay blocked because nothing has closed.
- The unblock wave fires only at the Phase-2 close step: when the WIP beads pass tests and close, their dependents flip to ready in a burst.

So the loop is not "code forever then test once at the very end"; it is a
**pump**. Each controller-authorized DSR checkpoint may close a proved layer,
which unblocks the next layer of the pool. Checkpoints remain occasional: the
controller batches enough compatible work to amortize host contention and does
not turn a temporary ready-pool dip into twelve independent builds.

### The trigger (when to flip Phase 1 → Phase 2)

**Quiescence plus a meaningful proof batch.** While agents have claimable work,
reviewed commits and in-progress changes arrive steadily. A ready-pool/commit
rate dip is one input, not authorization by itself. The controller first asks
panes to commit or pause, confirms the selected SHA is immutable and the source
tree is clean, checks there is no competing build/edit wave, and then decides
whether enough compatible work has accumulated to justify DSR. The prior
campaign's 20 → 12 → 5 commits-per-ten-minutes shape is historical context,
not a FrankenNLP target or trigger threshold.

### Enforcement (what keeps it honest)

Agents want to build to "prove" their work, so the model has to be actively enforced:

- **Build enforcement:** every controller tick detects unauthorized pane-owned
  Cargo, RCH, DSR, or GitHub Actions processes/jobs. Terminate or cancel only an
  exact validated process group or job owned by that pane—never a broad
  `pkill`, user-wide process-name match, or guessed target-directory match.
- **Explicit directive:** ordinary agents run no Cargo, RCH, DSR, or GitHub
  Actions command. There are no pane-specific build-slot exceptions. Commit
  code-first work promptly and leave proof-sensitive beads `in_progress` until
  the controller's retained DSR receipt and their own named gates exist.
- **KPI reframed:** success during Phase 1 is measured by the commit stream, not per-bead closures (closures are deferred and arrive in bursts during Phase 2).

### Failure triage in Phase 2 (the part that makes it safe)

Code committed without running its tests will have failures; that is expected and fine, because the batch pass catches them all at once:

- **Cargo's early-abort trap:** a single test-target compile error makes
  `cargo test` abort early, so the pass/fail line is a misleading prefix (a
  prior campaign saw "240/0 green" that was actually 793/17 once it compiled).
  Fix compile errors first; after fixes are committed and contention quiesces,
  only the controller may request another DSR run for the true count.
- **Cluster failures by file**, then dispatch each cluster to one agent
  (file-exclusive, no collisions) with the exact assertion + location. Repeat
  code-first repair plus controller-requested DSR until zero failures.
- **Close only green, with evidence.** Cite the exact clean source SHA, retained
  DSR `PASS` receipt for `scripts/check.sh` on `production`, **and each bead's
  own named contract/test/eval evidence** in its close reason. A code checkpoint
  proves compatibility only; model-present parity, target-host performance,
  platform-native behavior, and human review remain separate authorities.
  Leave genuinely incomplete beads in progress; never false-close.
- **The project's quality gates move to close time, not per bead.** The
  "Mandatory Checks After Substantive Changes" section above applies to the
  controller's DSR checkpoint (including the combined UBS leg), and the plan's
  parity/eval gates are unchanged: a bead whose contract includes an L-gate or
  scorecard closes only when that separate gate also ran and was retained.

### Expected benefit and the measurement obligation

- Phase 1 parallelizes independent editing and removes all routine builds from
  the per-bead path.
- An occasional DSR checkpoint can cover several compatible changes at one
  immutable SHA instead of repeating compilation per bead.
- The intended effect is less build contention, local-fallback thrash, and disk pressure; campaign telemetry must show whether that effect materialized.

Those are design hypotheses, not a promise: the originating campaign estimated
a ~20–100× reduction in build-serialized wall time on its own workload, and
that figure is retained here only as the historical motivation. Each campaign
records wall time, queue time, build count, failure/rework rate, and
escaped-defect rate against a representative prior or pilot wave; if batch
verification increases integration failures or does not save time, shrink the
wave. Do not create pane-level build exceptions.

### Hard-won gotchas (baked in from prior campaigns)

- **Shared `main`, no git surgery.** A `git reset` by one agent can orphan a peer's commit. Verify no commit was lost via `git merge-base --is-ancestor`; never rewrite shared history.
- **Exact-path commits in the shared tree.** Reserve the bead's paths, stage only explicit owned paths (`git add -- path...`, never `git add -A`), inspect `git diff --cached --name-only` and the staged patch, and refuse a commit containing a peer's file or unrelated hunk. Another agent's working-tree dirt is never a reason to stash, revert, or absorb it.
- **Stale rate-limit display.** A codex "usage limit" message persists in-buffer and codex won't auto-retry; nudge the pane to confirm before assuming an outage (a prior campaign lost ~5.5 hours idling on a false outage).
- **Degraded Agent Mail.** If the mail layer is slow or down, fall back to bead-assignee locking (`br update <id> --assignee <agent>`) instead of blocking on mail reservations.
- **Disk/build-storm watch.** If free disk drops fast or the build-process count
  spikes, enforcement slipped; identify only the exact pane-owned offender,
  stop that exact job/process safely, and re-issue the directive.

---

## bv — Graph-Aware Triage

`bv` computes PageRank/betweenness/critical-path/cycles over `.beads/issues.jsonl` (generated by `br sync --flush-only`). **Use ONLY `--robot-*` flags — bare `bv` launches a blocking TUI.** Start with `bv --robot-triage` (counts + top picks + quick wins + blockers). `bv --robot-plan` for parallel tracks; `bv --robot-insights` for full metrics (check `.Cycles` — must be empty).

---

## UBS — Ultimate Bug Scanner

Run `ubs --diff` over working-tree changes and `ubs --staged` immediately before each commit. Exit 0 = safe; exit >0 = fix and re-run.

```bash
ubs --diff                  # modified files relative to HEAD
ubs --staged                # staged files immediately before commit
ubs --only=rust .           # restrict a project scan to Rust
```
Parse `file:line:col` → location, 💡 → suggested fix. Fix root cause, not symptom. Critical (always fix): memory safety, UB, data races. Important: unwrap panics, resource leaks, overflow.

---

## DSR — Sole Build and Release Authority

`/dsr` governs the occasional controller-selected build/checkpoint/release job.
Ordinary panes never invoke DSR, Cargo, RCH, or GitHub Actions. The controller
waits for the code-first wave to quiesce, selects one clean immutable SHA, and
submits a DSR recipe whose single validation entrypoint is `scripts/check.sh`
against the named `production` feature graph. Retain the exact source SHA,
expanded command/recipe identity, DSR and toolchain versions, host/target, exit
code, and literal `PASS|FAIL` result.

GitHub Actions may remain disabled in the repository for historical context,
but its status and logs are non-authoritative. Direct RCH runs are likewise
diagnostic history only; because RCH can fail open to local execution, they are
especially unsuitable as release proof. DSR registration, host readiness, and
the explicit production-graph wiring are real prerequisites—if any is absent,
the build verdict is `BLOCKED`, not an inferred pass. A green DSR code receipt
does not satisfy model-present, target-host performance/platform, artifact,
security-review, or human-authorization gates unless those exact legs also ran
and were retained under their own evidence contracts.

A release DSR job additionally requires configured signing, SBOM, SLSA, exact
asset-inventory, and offline-verification surfaces. Missing release tooling or
key material blocks release; it never downgrades publisher authentication to a
checksum-only claim.

---

## ast-grep vs ripgrep vs warp_grep

- **`ast-grep`** when structure matters (refactors/codemods, policy checks, safe rewrites): `ast-grep run -l Rust -p '$X.unwrap()'`.
- **`ripgrep`** for raw text/literal hunts and pre-filtering.
- **`mcp__morph-mcp__warp_grep`** for exploratory "how does X work?" — an AI agent expands the query, reads files, returns line ranges with context. Don't use it to find a known symbol (use `rg`); don't use `rg` to understand architecture (use `warp_grep`).

---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations so we can reuse solved problems. **Never run bare `cass` (TUI)** — always `--robot` or `--json`.

```bash
cass search "int8 simd gemm" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
```
stdout is data-only, stderr diagnostics, exit 0 = success. The franken_ocr / frankensearch kernel campaigns solved many of this project's problems already (tiled GEMM, dispatch, deadlocks, gauntlet mechanics) — search before re-solving.

---

## Note for Codex/GPT agents — unexpected working-tree changes

If `git status` shows edits you did not make (in `Cargo.toml`, `src/*.rs`, etc.), those are from the **other agents working on this project concurrently** — a normal, frequent occurrence. **NEVER** stash, revert, or overwrite another agent's work. Treat those changes exactly as if you made them yourself. Do not stop to ask about them.

---

## Note on Built-in TODO Functionality

If I explicitly ask you to use your built-in TODO functionality, do so without complaining that you need to use beads. Always comply with such orders.
