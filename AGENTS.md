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

`franken_nlp` is a **pure-Rust, memory-safe, CPU-hyper-optimized library + one CLI program (`fnlp`, also shipped as `franken_nlp`)** that runs **Nanbeige4.2-3B** with no general ML framework and turns it into a local NLP toolbox. We transform bf16 weights into `.fnlpq` (int8 first, int4 later) and write model-specific kernels for this one model. Plan §5.1/§5.6 is the normative artifact lifecycle: pinned local HF snapshot → deterministic Generic conversion → fixed 1,957,046,720-byte GitHub Release chunks → release-bound manifest → `fnlp pull` streamed verification/reassembly → native packing → atomic activation. The model weights are **Apache-2.0** — declared by the official model card at the pinned revision, which is the standard Hugging Face license declaration and is settled, not uncertain. Every published artifact/release carries the Apache-2.0 text, Nanbeige attribution, and our modification notice per plan §5.7.

- **Apple Silicon / ARM64** — M4/M5: NEON/SDOT and SMMLA only when the OS actually advertises FEAT_I8MM; autovec remains a measured candidate
- **AMD / Intel x86-64** — Zen 3 Threadripper AVX2 first-class; Zen 4/5 and Xeon compare sustained 512-bit VNNI, 256-bit VNNI, and narrower tiers rather than assuming widest wins

**CPU is the product** (most target hosts have no usable GPU); CUDA is out of scope; Metal is at most a late stretch experiment.

### What we stand on (the closed dependency universe)

- `/dp/frankentorch` (`ft-kernel-cpu`, `ft-core`, `ft-serialize`) — consumed at the kernel level. At the audited pin, dynamic int8 linear has SDOT/AVX-512-VNNI/scalar but no AVX2/SMMLA, and f32 SDPA is dense per-head, not native 48:8 GQA. Read plan §3.5 before claiming reuse.
- `../asupersync` — structured-concurrency runtime, for **orchestration / cancellation / IO only** (not for intra-op math parallelism). Its HTTP/TLS client (`tls-webpki-roots`) is the ONLY networked Rust/product code path (`fnlp pull`); no reqwest/hyper, ever. The explicit out-of-band `scripts/fetch_model.sh` / `scripts/fetch_model.ps1` provisioning tools may use system `curl` / PowerShell web APIs to fetch the pinned upstream source closure; they are not inference or library code.
- `/dp/frankensqlite` (`fsqlite`, `fsqlite-types`) — optional metadata/job-state history, disabled by default (NEVER `rusqlite`; fsqlite tables never contain document/prompt/output text). Durable job content spools are separate, explicit, owner-only files governed by plan §8.4.
- The **complete** direct non-suite release allowlist is `clap`, `serde`/`serde_json`, and `sha2`, each only for the layer named in plan §3.4. Everything else must be `std` or a pinned FrankenSuite surface. **No `thiserror`, `anyhow`, `ctrlc`, `rayon`, `half`, `memmap2`, `uuid`, `num_cpus`, allocator crate, `tokenizers`, `tiktoken`, `minijinja`, or `llguidance`.** Tokenizer/template/grammar are ours; the tokenizer bytes (Apache-2.0, attributed) are embedded in the binary and hash-checked against the artifact's copy at load.

Project source license is the exact text in `LICENSE` (MIT + OpenAI/Anthropic Rider); the model weights remain **Apache-2.0 © 2026 Nanbeige Team**, with attribution carried in artifacts, manifests, and `fnlp --version`. `swiss_army_llama` is the project owner's own prior work — reuse its methods/presets freely with attribution.

**The single source of truth for what we are building and why is [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md).** Read it before writing any kernel.

### What this model is (one paragraph)

Nanbeige4.2-3B is a dense decoder-only Llama-family transformer with a loop. The pinned source/index currently show: 22 physical layers; two passes; KV slot `layer + loop×22`; **final RMSNorm after each pass, so loop 2 consumes loop-1 normalized output**; 44 layer outputs plus two post-loop norm states. Dimensions: hidden 3072, 48 query / 8 KV heads, explicit head_dim 128, SwiGLU 10752, eps 1e-5, split-half RoPE θ=7e7, max positions 262144, vocab 166144, untied lm_head, no bias/window. The 201-name index contains no mHC/depth/LoopSplit/ngram tensors. These are `[OBSERVED@pin]`, not implementation authority until archived under `docs/truth-pack/`. Tokenizer is SentencePiece BPE on the slow reference path; the template uses im/think markers and XML/JSON tool branches.

---

## Product Shape

The project must be both:
1. A reusable Rust library (`NlpEngine`) for embedding the model + task layer, **synchronous and blocking** — the async runtime is an owned implementation detail.
2. A standalone CLI binary `fnlp` with:
   - **robot mode** (agent-first, versioned NDJSON, self-describing `robot schema`)
   - human mode (`fnlp extract|ner|classify|sentiment|… <file>` → JSON/text, `--json` everywhere)
   - **batch daemon mode** (`fnlp batch`: NDJSON docs on stdin → NDJSON results on stdout, bounded queues, weights loaded once)
   - opt-in **durable job mode** (`fnlp job`: manifest → journal/resume/verify/materialize with no content persistence by default)
   - user-owned **Assay surfaces** (`eval`, `calibrate`, `qualify`) and a bounded data-only recipe compiler only after its Phase-5 gates

Input: text files/stdin/NDJSON. No Python or foreign ML/runtime ABI in the default inference path, no inference network, no GPU required. Enumerated OS-facing islands/features (SIMD, opt-in mmap/NUMA hints, optional allocator) are governed by plan G3/G4 rather than hidden under an absolute “no FFI” slogan.

---

## Porting Workflow (Spec-First)

Implementation follows spec documents, not ad-hoc copying. Read in this order:
1. [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md) — the master plan.
2. The **research-decision register (plan §14)** — observed rows need truth-pack promotion; partial/open/blocked rows gate only their dependent surface. Empirical questions must be measured in their named phase, not “answered” from prose.
3. The **reference sources** (pinned + hashed in `docs/truth-pack/`): `modeling_nanbeige.py` (the loop driver, KV binding, attention), `configuration_nanbeige.py` (the defaults that decide which features are active), `tokenizer_config.json` (the verbatim chat template), `tokenizer.model`, and the `Nanbeige/llama.cpp` `nanbeige42` fork (a second, easier-to-read implementation of the same semantics).

**Hard rule: no surface ships against an unresolved dependency.** Promote source observations only with immutable hash + line span + replay fixture.

---

## The franken_nlp Engineering Doctrine (READ THIS BEFORE OPTIMIZING)

Load-bearing, non-negotiable rules distilled from the plan and from the frankentorch / frankensearch / franken_ocr prior art. Violating any of them has burned real days of work before.

1. **Correctness outranks speed, always (G1 > G2).** Parity gate FIRST, perf second. A faster kernel that drifts decoded tokens is reverted — no source landed — and memorialized in `docs/NEGATIVE_EVIDENCE.md`. We ship speed *on top of* parity, never instead of it.

2. **Valid-by-construction task output (G8) has the same rank.** Every successful structured response validates; budget/cancel/resource failure is typed no-result, never partial success. An unconstrained “retry on parse failure” loop is a design bug.

3. **The loop is the architecture.** Decode streams decoder weights twice; KV is 44-deep; execute `22 layers → final RMSNorm → 22 layers → final RMSNorm`. The L2 ladder has 44 layer outputs **plus two norm states**. Never size/schedule as a conventional 22-layer model.

4. **`head_dim` is 128, not `hidden/heads` = 64.** The config overrides the Llama fallback. q_proj is 3072→6144, k/v are 3072→1024, o is 6144→3072. Any kernel, buffer, or RoPE table built on 64 is silently, catastrophically wrong.

5. **Do not hand-roll wide SIMD for glue without a profile.** Explicit SIMD belongs first in proved int8 MAC kernels. Polynomial exp/sigmoid is only a default-off candidate here; sibling wins are priors, not this project's measurements.

6. **Measured-faster wins; hardware capability is not a routing decision.** franken_ocr shipped Apple Silicon defaulting to LLVM-autovec over SDOT/SMMLA for some dense int8 shapes because that is what measurement said. Dispatch tables record measured choices per shape; forced paths remain for proof runs.

7. **AVX2 is first-class and exact.** Raw `vpmaddubsw` can saturate inside the instruction; accumulation cadence cannot fix it. Implement/measure both plan §6.5 candidates: low-7/high-bit decomposition with proved i16 pair bounds and the widened signed-i16 route. Both equal scalar/i64 over the full i8 domain; raw saturation is not a shippable approximate mode.

8. **NEVER nest rayon under a held lock; NEVER nest asupersync runtimes.** The process has one shared runtime/kernel resource set; engines hold leases. Concurrent sync callers enter bounded admission, not independent forwards. Re-entrant calls from the same runtime fail typed/fast. Scheduler lifecycle is model-checked/replayed, not just watchdog-tested.

9. **Determinism is scoped.** Semantic greedy output is exact under a named numerics profile; canonical bytes additionally require ordered output and scrub volatile metadata. Batch/prefix equivalence requires the same per-row reduction order. Never promise byte identity for completion-order NDJSON or approximate transcendental modes.

10. **Integer bounds cover every intermediate.** At K=10752: S8×S8 ≤176,160,768; raw U8×S8 ≤350,945,280; correction ≤176,160,768; conservative raw+correction ≤527,106,048. Prove every model K, int4 group, row sum, tail, and SIMD tier against i64.

11. **Honest, measured everything.** Every accepted numeric divergence → `docs/DISCREPANCIES.md` (reference behavior, ours, measured impact, kill-switch env var, review date). Every rejected optimization → `docs/NEGATIVE_EVIDENCE.md` (the 5-pass loop; losers reverted with NO source landed). Gauntlet comparisons use thread/allocator/precision fairness controls; slower rows stay in the ledger. Task-quality claims name their dataset, prompt hash, recipe id, and thinking mode. No silent numerics changes, ever.

12. **Two binaries from one entrypoint:** `fnlp` (short) + `franken_nlp` (long). Shared dispatch is `pub fn cli_main() -> ExitCode` in the lib; each binary is a **thin one-line shim in its own file** (`src/main.rs`, `src/bin/fnlp.rs`) — never the same `path` in two `[[bin]]` targets.

13. **Presets are data, prompts are versioned.** Task prompts/presets live as embedded data files; every task response carries the prompt-template hash so quality regressions are attributable to weights vs kernels vs prompts. Changing a prompt is a versioned, diffable event with an eval re-run, not a drive-by string edit.

14. **AVX-512 width is not destiny.** Benchmark sustained zmm-VNNI, ymm-VNNI, AVX2, and scalar per host/shape with p50/p95/p99, clocks/energy where available, and thermal steady state. Dynamic M/T tails still exist; firmware/VM feature exposure is authoritative.

15. **Memory admission precedes allocation.** BF16 KV is 176 KiB/token/sequence: 8K is ~1.38 GiB and batch 64 is ~88 GiB. Use lazy COW pages, an explicit memory budget, and checked worst-case admission. Reported 256K context is not a usability promise.

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

28. **Source acquisition, release packaging, and client installation are different trust zones.** `scripts/fetch_model.sh` / `scripts/fetch_model.ps1` download the immutable upstream conversion closure; they never masquerade as the end-user installer. Research-only truth-pack evidence does not become a converter prerequisite. The public artifact is the deterministic Generic `.fnlpq`, versioned independently from the binary and split at exactly 1,957,046,720 bytes except the tail. Never publish from an unreceipted converter output, never upload with a wildcard/`--clobber`, and never rewrite bytes under an existing tag/name.

29. **One artifact manager owns every client model download.** `install.sh`/`install.ps1` install the binary, then invoke that exact installed binary's `fnlp pull`; shell and PowerShell must not parse the model manifest, concatenate parts, invent cache filenames, or duplicate integrity logic. Default pull trusts only the release-bound embedded manifest; local/private manifests require an explicit expected digest. Part + whole + source/license/census identities, native-pack differential, and atomic activation all pass before the new model becomes visible.

---

## Alien-Artifact Engineering Contract

Start with a deterministic baseline and measured hotspot. A candidate needs `EV=(Impact×Confidence×Reuse)/(Effort×AdoptionFriction)≥2`, explicit state/actions/loss/units, assumptions, p50/p95/p99, proof/tolerance authority, interaction/composition check, provenance, and fallback. Runtime adaptation follows offline replay → shadow → canary → explicit default; at most one controller per subsystem. Use CVaR alone for quality-tail optimization; do not stack tail jargon. Exact loop speculation or graveyard it. Cross-loop wavefront work requires a fragmentation trace before code. See plan §10.5 cards AA-K1/Q1/S1/W1/C1/T1/D1/U1/A1/R1 and §10.6's rejected/deferred ideas.

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
- `unsafe_code = "deny"` in `[lints.rust]` + `#![deny(unsafe_code)]` at crate roots—**not `forbid`**, which cannot be locally overridden. Only enumerated SIMD/mmap modules may opt into scoped `allow`; every unsafe operation needs `// SAFETY:`. Integer kernels exactly differential-test against scalar/i64; floating candidates use named tolerances; mmap uses separate range/lifetime/immutability tests. A policy test rejects any unlisted island.
- Cargo only. Persistence via `fsqlite`, never `rusqlite`. Direct non-FrankenSuite release dependencies are frozen to `clap`, `serde`/`serde_json`, and `sha2`; a written justification is not enough to add another—change requires explicit owner approval and a plan revision.

---

## Mandatory Checks After Substantive Changes

```bash
cargo fmt --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
ubs --diff
```

If any check fails, fix root causes before handing off.

### The `cargo test --locked` gate (green-bar requirement once the crate exists)

After `Cargo.toml` and `scripts/check.sh` exist, `cargo test --locked` is a hard gate and must exit 0 before code is handed off or a bead is closed. Before Phase 0 scaffolding, report Cargo gates as **not applicable (crate absent)** and run only truthful document/link/whitespace/UBS checks; never create a fake empty crate merely to manufacture a green bar. CI invokes `scripts/check.sh` as its single test step once present.

Build-surface note: both binaries compile from thin shims over `cli_main()` (doctrine #12); `cargo check --locked --all-targets` MUST be free of the "present in multiple build targets" warning.

---

## Testing Policy

Each module includes unit tests for happy path, edge cases, error handling. Beyond that, the conformance ladder is the heart of the project (plan §9):

- **Reference oracle**: pinned CPU HF environment running `NanbeigeForCausalLM` (`trust_remote_code`, `use_fast=False`) via `scripts/gen_reference_fixtures.py` — **establish the oracle's own nondeterminism floor first** (two runs × two thread counts) before setting any tolerance.
- **Parity ladder L0–L5**: L0 exact; L1 metric vector (not cosine alone); L2 all 44 layer outputs + two loop norms; L3 measured tolerance; L4 f32 greedy exact where oracle reproducible; L5 task semantics.
- **Invariant gates**: scoped determinism, batch/prefix equivalence, independent structured-output/source-grounding validation, untrusted-segment bytes preserved with all control ids excluded (or rejection), sparse=full, forced=sequential KV, every trie traversal=naïve, uninterrupted=resumed owned job, dependency-scope snapshot mutation, integer SIMD ≡ scalar/i64, overflow/fork-tail/admission proofs, scheduler model/interleaving tests, cache privacy, source-manifest verification, conversion-twice identity, split/reassembly identity, streamed part/whole verification, resume/concurrent-pull safety, and installed-basename discovery.
- **Task evals** (plan §9.6): named, versioned datasets per task; scorecards carry recipe id + prompt hash + thinking mode. Quality regressions gate releases like parity regressions.
- **Adversarial-content and evidence gates**: matched clean/attack fixtures report task-specific steering rather than a firewall claim; semantic second-reader uplift is measured against labeled fields; acceptance-audit arithmetic uses exact small-population goldens and requires human grades.
- **Model-gated e2e**: ordinary CI reports `SKIPPED_NO_MODEL`; release certification requires an armed pass with artifact digest and fallbacks pointed at `/nonexistent`. An authorized public-model release additionally proves fresh HOME/LOCALAPPDATA installer → `fnlp pull` → no-flag inference, followed by a byte-perfect cache hit whose inference run opens no network.
- **`many_docs_without_deadlock`** concurrency watchdog.

---

## Agent Ergonomics Requirements

Robot mode must be: stable versioned schema, deterministic where possible, explicit exit codes, line-oriented NDJSON, easy to pipe. Do not mix human decoration with machine output in robot mode. `robot schema` self-describes the contract; a contract test validates emitted events against a frozen JSON schema fixture. Stable exit codes are documented in `error.rs` (plan §8.5). stdout is data, stderr diagnostics; bare `fnlp` prints help, never a TUI; honor `NO_COLOR`, `CI`, `TERM=dumb`.

---

## Session Completion ("Landing the Plane")

Before finishing a work session you MUST:
1. Once `.beads/` is initialized **after the plan's ≥4 review rounds**, file issues for remaining implementation work. During pre-Beads planning, do not initialize Beads early; record review gaps in the plan/handoff.
2. Run quality gates (if code changed) — tests, clippy, fmt, `ubs`.
3. Update issue status — close finished work, update in-progress.
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

## bv — Graph-Aware Triage

`bv` computes PageRank/betweenness/critical-path/cycles over `.beads/beads.jsonl`. **Use ONLY `--robot-*` flags — bare `bv` launches a blocking TUI.** Start with `bv --robot-triage` (counts + top picks + quick wins + blockers). `bv --robot-plan` for parallel tracks; `bv --robot-insights` for full metrics (check `.Cycles` — must be empty).

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

## RCH — Remote Compilation Helper

RCH offloads `cargo build/test/clippy` to remote workers to avoid local compilation storms. Installed at `~/.local/bin/rch`, hooked into Claude Code's PreToolUse — usually transparent. Manual: `rch exec -- cargo build --locked --release`. Health: `rch doctor`, `rch status`. Fails open (builds run locally if workers unavailable). **Codex/GPT users:** no auto-hook — manually `rch exec -- <cmd>` for heavy builds.

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
