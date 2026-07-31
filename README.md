# franken_nlp

<div align="center">

[![License: MIT + Rider](https://img.shields.io/badge/License-MIT_+_OpenAI/Anthropic_Rider-blue.svg)](./LICENSE)
[![status: design review](https://img.shields.io/badge/status-design_review-yellow.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
[![target: Rust 2024 nightly](https://img.shields.io/badge/target-Rust_2024_nightly-orange.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
[![model card: Apache--2.0; assets blocked](https://img.shields.io/badge/model_card-Apache--2.0%3B_assets_blocked-teal.svg)](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
[![target: valid by construction](https://img.shields.io/badge/target-valid_by_construction-red.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)

**Designing a pure-Rust, memory-safe, CPU-hyper-optimized local NLP engine around exactly one model: Nanbeige4.2-3B. The target is a reusable library plus one CLI program (`fnlp`, also shipped as `franken_nlp`) with model-specific M4/M5 and high-core-count x86 kernels, valid-by-construction structured output, and a corpus-scale task layer.**

</div>

> **Current state (2026-07-30): design only.** The repository currently contains the plan, agent instructions, source license, changelog, and preserved idea/cross-score/reaction records. There is no Cargo crate, executable, installer, model artifact, benchmark, or implemented task yet. Commands and APIs below are explicitly **target interfaces**, not runnable examples. The plan is in external review round 1 of at least 4; its immutable-source observations still require a committed truth pack.
>
> **Model-license state:** the pinned Hugging Face card declares Apache-2.0, but the pinned repository contains no LICENSE or NOTICE file. Public transformed-weight assets and an embedded tokenizer are therefore **blocked** until the fail-closed provenance gate in plan §5.7/LG-1 is cleared. Local user-side conversion remains the fallback design.

---

## TL;DR

**The problem.** Extraction, entity work, corpus classification, dimension scoring, local redaction, and RAG verification sit awkwardly between fast classical pipelines and capable but costly cloud LLMs. Generic local runtimes solve completion, not the whole validity, offset, calibration, privacy, and batch-throughput contract. The Nanbeige authors report unusually strong small-model results, while its looped architecture currently requires their llama.cpp fork rather than upstream.

**The design.** Specialize the full stack to its fixed K/N shapes and exact loop: 22 layers → final RMSNorm → the same 22 layers → final RMSNorm, with KV slot `layer + loop×22`. Build exact int8/int4 kernels, continuous layer-major batching, lazy paged KV, privacy-scoped prefix reuse, and a bounded grammar execution compiler that can evaluate all-and-only legal lm_head rows, teacher-feed runs of uniquely forced **token ids**, and constrain declared fields to source substrings. Add exact continuation tries, crash-resumable corpus jobs, typed untrusted-document segments, and user-owned qualification. Every performance and quality advantage remains a target until the pinned oracle, task scorecards, and fair fork gauntlet measure it.

**Why `fnlp`:**

| | `franken_nlp` |
|---|---|
| Model observation | 4.1698B params (3.149B decoder), 22 layers used twice, 44 KV slots, max-position config 262144, thinking/tool template. Benchmark and license values are author/card claims until independently evidenced. |
| Target packaging | Self-contained release executables; pinned sovereign source fetch + deterministic local conversion. If LG-1 clears: one Generic `.fnlpq`, fixed 1,957,046,720-byte GitHub Release parts, a release-bound embedded manifest, and installer-delegated `fnlp pull`. No network during inference. |
| Target output contract | Successful structured results validate against the supported schema subset by construction; declared `verbatim` fields are source substrings by construction after OQ-17; budget/cancellation/unsatisfiable constraints return typed no-result, never truncated “success.” Repeated occurrences and confidence scope stay explicit. |
| The loop, exploited | Decode streams the full 3.15B non-embedding weights **twice per token**; the engine schedules both passes (per-loop KV binding, loop-aware prefetch, loop-corrected rooflines) instead of replaying a generic graph — and every byte saved by int8/int4 quantization pays out double. |
| Corpus throughput target | Layer-major batching streams each physical layer once per loop for all compatible rows—two shared streams rather than 2M—plus privacy-scoped prefix pages, exact finite-language projection, and opt-in durable snapshot jobs with scope-correct cache reuse and verified resume. |
| Target determinism | Semantic greedy determinism under a named numerics profile; canonical bytes additionally fix ordering and omit volatile telemetry. Batch/prefix equivalence requires canonical reduction order. |
| Hardware campaign | Apple M4/M5 runtime-detected autovec/SDOT/I8MM; exact AVX2 on Zen 3; sustained 256/512-bit VNNI comparisons on Zen 4/5 and Intel. Widest-capable is never assumed fastest. |
| Fidelity ladder | L0 tokenizer/template; L1 ops; 44 layer outputs **plus two post-loop RMSNorm states**; logits; greedy tokens; task outputs. |
| Target safety | `unsafe` denied crate-wide and allowed only in enumerated SIMD/mmap islands; integer SIMD has exact scalar/i64 differential proof and mmap has range/lifetime policy tests. |
| Dependencies | Immutable git+rev FrankenSuite foundations plus exactly three approved direct commodity families: `clap`, `serde`/`serde_json`, and `sha2`. Local sibling overrides must match the pins; no open-ended allowlist. |

---

## Target CLI sketch (not implemented)

```bash
# If and only if LG-1 clears: fetch a catalog-pinned, hash-verified artifact.
fnlp pull

# Structured extraction with a result that cannot be invalid.
fnlp extract --schema invoice.schema.json invoice_email.txt --json
# Optional Phase-5 same-model semantic second read; diagnostic, never a certificate.
fnlp extract --schema invoice.schema.json invoice_email.txt --verify-semantic --json

# Entities with source-verified character offsets; canonicalize across a corpus.
fnlp ner report.txt --types PERSON,ORG,GPE,DATE,MONEY
cat mentions.ndjson | fnlp batch --task resolve > entities.ndjson

# Candidate-conditional score distribution; full-vocab mass is a separate audit mode.
fnlp sentiment earnings_call.txt --focus-area earnings_calls
# → {"dimension":"optimistic","normalization":"candidate_conditional", ...}

# Classify 100K support tickets overnight on the Threadripper (batch daemon).
cat tickets.ndjson | fnlp batch --task classify \
    --task-args '{"labels":["billing","bug","feature_request","churn_risk"]}' \
    > labeled.ndjson

# For an overnight corpus whose completion matters: journal, resume, verify.
fnlp job start tickets.manifest.ndjson --task classify --output labeled.ndjson
fnlp job resume <job-id>
fnlp job verify <job-id>

# Redact PII before anything leaves the machine.
fnlp redact transcript.txt --policy pii-default --map-out map.json

# Verify RAG answers against their sources.
fnlp judge --faithfulness --source retrieved_passages.txt answer.txt

# Chat with thinking mode, reproducibly.
fnlp chat --think --seed 42 "Plan a Python 2→3 migration for a 100kLOC codebase"

# Agent surfaces: the contract, diagnostics, and an on-CPU kernel self-proof.
fnlp robot schema
fnlp robot selftest

# Measure a candidate on your own locked data before activation.
fnlp qualify --baseline active --candidate nanbeige-int4-v3 \
  --suite support-suite.json --policy production-gates.json

# Phase-7, if AA-A1 passes: freeze a human-graded acceptance sample for one owned job.
fnlp audit plan <job-id> --risk audit-policy.json
fnlp audit grade <audit-id> --grades human-grades.ndjson

# Sovereign path: fetch one immutable upstream conversion closure, then convert it locally.
# On Windows, use scripts/fetch_model.ps1 with equivalent arguments.
scripts/fetch_model.sh --dest /path/to/nanbeige-source
fnlp convert --source /path/to/nanbeige-source \
  --source-manifest docs/truth-pack/nanbeige4.2-3b.source.json \
  --recipe nanbeige42-int8-v1 --arch generic \
  -o nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
```

---

## The six bets

No single trick makes this worth building. The **composition** of six bets does — each one feasible precisely because the engine serves exactly one model.

| Bet | One-line statement |
|---|---|
| **B1 · One model, zero framework** | Fixed K/N dimensions enable shape-specialized packing/kernels; dynamic batch/context/candidate tails remain explicit and tested. Scratch is bounded; KV is admitted and paged lazily rather than preallocating impossible `ctx×batch` slabs. |
| **B2 · The loop is the moat** | Explicit two-pass schedule, two final norms, 44 KV slots, loop-corrected traffic models. Prefetch and every claimed speedup remain measured candidates. |
| **B3 · Compile finite languages into execution** | Closed-set tasks use exact row slices/continuation tries; breadth, depth-first, short-ID, and naïve modes price actual 44-deep KV tails before allocation. Constrained states compute every legal row or full fallback; forced tokens still update all 44 KV slots. |
| **B4 · Valid, grounded, and structurally contained** | The finite schema subset compiles to bounded execution; declared `verbatim` fields intersect a source language. Untrusted document bytes cannot become role/think/tool control ids. Successful output validates and grounds; unsatisfiable/budget/cancel returns typed no-result. This does not make semantic correctness or prompt-injection immunity. |
| **B5 · Corpus-scale, crash-resumable fabric** | Layer-major batching amortizes the double weight stream; prefix pages use byte-certified fork tails; NDJSON stays composable; opt-in fsqlite jobs add exact snapshot keys, scoped local/reduce/global reuse, journal/resume/verify, and owned materialization without retaining text by default. |
| **B6 · Evidence-native honesty** | The L0–L5 ladder (44 layer states + two loop norms), measured ISA dispatch, losing-row ledgers, staged quantization, disjoint calibration/test, and user-owned qualification—every claim is observed, partial, reported, targeted, hypothetical, open, blocked, or evidenced. |

---

## Design philosophy

These are the constitutional, non-negotiable constraints the whole system is built under. They read like restrictions; they are the moat.

1. **The dependency universe is closed.** Target dependencies are `std`, the pinned nightly, audited FrankenSuite layers, and exactly `clap`, `serde`/`serde_json`, and `sha2` as direct commodity exceptions. Frankensqlite is optional metadata/job state. The tokenizer/template/grammar implementations are in-house; upstream tokenizer bytes still require LG-1.
2. **Correctness outranks speed, always.** The parity ladder gates every kernel; a faster kernel that drifts decoded tokens is reverted, no source landed, and recorded in the negative-evidence ledger. Speed ships *on top of* parity, never instead of it.
3. **Valid-by-construction output has the same rank.** A constrained-decode change that could emit schema-invalid JSON is reverted like a parity break. There is no "retry on parse failure" anywhere in the engine, by law.
4. **The loop is the architecture.** Never size as a conventional 22-layer model: 44 layer executions/KV slots and two loop-final norm states are pinned before kernels.
5. **Determinism is scoped, not hand-waved.** Semantic, byte, batch, and prefix claims name numerics/order/telemetry conditions. Approximate math never inherits the exact profile's promise.
6. **Measured-faster wins; width is not routing.** Apple candidates are runtime-detected; Zen/Intel compare sustained zmm/ymm/AVX2 paths. AVX2 has two exact constructions; raw saturating `vpmaddubsw` is banned.
7. **Honesty is enforced, not aspired to.** Every accepted numeric divergence lives in `docs/DISCREPANCIES.md` with a kill-switch; every rejected optimization in `docs/NEGATIVE_EVIDENCE.md`; benchmark comparisons are thread/allocator/precision-fair against the strongest real baseline with slower rows published; task-quality claims name their dataset, prompt hash, recipe id, and thinking mode.
8. **Trust boundaries name exactly what they prove.** Typed document encoding prevents control-token smuggling but not instruction-shaped content steering. A same-model verifier is correlated evidence, not independent proof. A corpus audit needs frozen sampling and human grades, and its claim stays scoped to that population.

---

## How it works

`franken_nlp` is seven named subsystems around one model, one artifact, and one batch fabric (plan §4–§9; codenames map onto the module tree, they don't add structure).

```
                        ┌──────────────── fnlp CLI / library (sync, blocking) ────────────────┐
   text / NDJSON ──►  ATELIER · the task layer
                        │  extract · ner · resolve · sentiment · classify · judge · redact
                        │  summarize · keyphrases · answer · generate · tokens · split
                        │  presets-as-data · prompt hashes · calibration · map-reduce
                        ▼
     LEXICON · SentencePiece BPE (approved asset, id-exact) + native chat-template builder
               trusted control segments · byte-preserving forbidden-control-id document path
     STENCIL · schema/source languages → bounded execution program
               full projection · every-legal-row projection · forced causal runs
                        ▼
     CONVEYOR · batch fabric: layer-major continuous batching · byte-certified prefix tails
                bounded NDJSON · scope-correct snapshot journal/resume/verify corpus jobs
                        ▼
     OUROBOROS · the loop-scheduled model core
                embed → 22 layers → norm → same 22 layers → norm → lm_head (full | sliced)
                RMSNorm→RoPE(θ=7e7)→GQA 48:8 @128 (44-deep KV) → SwiGLU 10752
                kernels: int8/int4 tiled GEMM/GEMV — NEON SDOT/SMMLA · AVX-512-VNNI
                (Zen4/Zen5/Intel measured separately) · AVX-VNNI · exact AVX2 · scalar
                        ▼
     samplers (greedy/seeded · grammar-mask AND) → validators (schema · offsets · calibration)
   ── FOUNDRY · local convert → .fnlpq → authorized assets only after LG-1
   ── ASSAY · L0–L5 (44+2 states) · proofs/properties/model checks · task evals · gauntlet
   ── process-shared asupersync resources · optional metadata/job-state fsqlite
```

- **Foundry** — target weight pipeline: strict census, deterministic staged quantization, canonical Generic artifact, local arch packing, checked format, and atomic content-addressed install. Public split release assets and embedded tokenizer bytes remain LG-1-conditional.
- **Ouroboros** — the model core, named for what it is: the snake that runs its own layers twice. The loop schedule is explicit (per-`(layer, loop)` KV binding resolved at engine build; loop-aware software prefetch), attention is GQA 48:8 at head_dim 128 (explicitly 128 — the config overrides the Llama fallback of 64, and the whole engine is built on that fact), and the lm_head has two personalities: full 166K-vocab GEMV fused with argmax for generation, and a **row-sliced GEMV** for scoring tasks that need only candidate-token logits (~510M MACs → <1M).
- **Conveyor** — target throughput fabric: compatibility-keyed layer-major groups, fair prefill morsels, checked memory admission, lazy COW KV pages with separately priced fork tails, shipped-prefix-only caching by default, bounded NDJSON, and opt-in durable snapshot jobs. TaskIR stages declare item-local, partition-reduce, or corpus-global authority so unchanged documents never launder stale global results. Exactly-once-style authority stops at the owned spool/materializer; arbitrary stdout is documented at-least-once replay.
- **Stencil** — target constrained execution compiler over the plan's finite schema subset and optional byte-exact source language. It chooses only equivalent primitives: full projection, every-legal-row projection, or sequential/causal feeding of uniquely forced tokens. Universal fallbacks remain; budget/cancel/unsatisfiable returns no result, never invalid success.
- **Lexicon** — target pure-Rust SentencePiece BPE and typed fixed-template implementation. Only trusted template code may emit role/think/tool-control ids; an untrusted document byte-preserves without them or rejects. That structurally contains marker smuggling, not prose steering. Embedding bytes waits on LG-1; exact slow-tokenizer fixtures gate L0.
- **Atelier** — target task layer over a bounded internal `TaskIR`. A later public data-only recipe surface opens only if built-in equivalence, caps, and no-code/no-network gates pass. Tasks graduate individually on scorecards.
- **Assay** — deterministic claims use bounds, differential/property tests, bounded model checking, and hostile interleavings; statistical task claims use locked data and confidence intervals. Users get the same digest-scoped `eval/calibrate/qualify` machinery; no e-process substitutes for proof.

The full specification lives in [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md), including its explicit evidence vocabulary and research-decision register.

## Target task portfolio

These are staged targets, not current commands. Tasks graduate individually only after their plan §9.6 scorecard; a successful structured response must validate, while resource/cancellation failure returns a typed no-result error.

| Command | What it does | Decode strategy |
|---|---|---|
| `fnlp extract --schema s.json` | **Flagship.** Supported finite JSON-Schema subset → valid successful JSON; optional `x-fnlp-source: verbatim` → source-grounded value after OQ-17; optional Phase-5 same-model semantic verification only if scorecard-promoted | constrained JSON/source |
| `fnlp ner` | Typed source-grounded spans with exact UTF-8 byte + Unicode-scalar half-open offsets; repeated occurrences remain explicitly ambiguous | constrained JSON/source |
| `fnlp resolve` | Entity canonicalization across a doc/corpus: clusters, canonical names, aliases | blocked pairwise judging |
| `fnlp sentiment --focus-area X` | Candidate-conditional bucket scores plus a full-vocab audit path; sampled/justified mode; default chosen only after held-out quality/calibration/throughput gates | sliced distribution / sampled |
| `fnlp classify --labels …` | Zero-shot single/multi-label with calibrated probabilities and exact shared-prefix continuation tries when they win | prefill-only, sliced/trie |
| `fnlp judge` | Rubric scoring, RAG faithfulness (`entailed/contradicted/unsupported` + evidence), order-debiased pairwise preference | mixed |
| `fnlp redact` | PII detection (LLM ∪ rules) + masking/pseudonymization, offset-exact, optional map, and `--verify` residual scan | constrained + rules |
| `fnlp summarize` | Length/style presets, `--grounded` span evidence, map-reduce over long docs | free text |
| `fnlp keyphrases` | Ranked, source-anchored keyphrases | constrained list |
| `fnlp answer` | Context-grounded QA with span citations and **calibrated abstention** | constrained + free |
| `fnlp generate` / `chat` | Completion/chat: thinking on/off, XML/JSON tool calls, `--seed`, streaming | free |
| `fnlp tokens` / `split` | Non-LLM utilities at wire speed: exact token counts for *this* tokenizer, sentence/chunk splitting — no model load | n/a |

**Deliberate non-goals:** POS tagging, dependency parsing, lemmatization (a finite-state tagger does those 100–1000× cheaper — keep spaCy for that slice); embeddings (that's [frankensearch](https://github.com/Dicklesworthstone/frankensearch)'s job; `fnlp` interoperates over shared NDJSON conventions); training/fine-tuning; a model zoo.

## Intended positioning (not a benchmark result)

Honest framing. `fnlp` is the only one of these built as an NLP *product* around this specific model, rather than a chat runtime it happens to fit into.

| | `fnlp` | llama.cpp (`nanbeige42` fork) | Python + transformers | spaCy | Cloud LLM APIs |
|---|---|---|---|---|---|
| Runs Nanbeige4.2-3B | **Planned: only this model** | Yes (authors' fork) | Yes | No | n/a |
| Packaging | **Target: executable + sovereign conversion; authorized artifact if cleared** | C++ build + GGUF | Python env | Python env | SaaS |
| NLP task layer | **Planned, scorecard-gated** | Completion surface | DIY | Classical pipeline | Vendor/DIY |
| Valid JSON | **Target: supported subset by construction** | Generic grammar options | DIY/framework-dependent | n/a | Vendor-dependent |
| Corpus engine | **Planned layer-major/paged/prefix-scoped NDJSON** | Generic server/batching | Framework-dependent | Excellent for classical tasks | Rate/cost constrained |
| Performance | **Unknown until gauntlet** | Baseline | Baseline oracle | Different task class | Not comparable locally |
| Data path | **Target: local inference; explicit pull only** | Local | Local | Local | Remote |

## Target `fnlp` CLI

> Robot mode emits line-oriented, versioned NDJSON an agent can pipe and validate against a frozen contract (`fnlp robot schema`). stdout is data, stderr is diagnostics, exit codes are stable and documented, bare `fnlp` prints help and never opens a TUI.

```bash
# Tasks (each: --json, -o file, --think/--no-think, batch daemon via `fnlp batch --task <t>`)
fnlp extract --schema s.json doc.txt
fnlp ner doc.txt --types PERSON,ORG,DATE | jq '.entities[]'
fnlp sentiment doc.txt --focus-area support_tickets --mode distribution
fnlp classify doc.txt --labels urgent,normal,spam --multi
fnlp judge --pair answer_a.txt answer_b.txt --criterion "factual accuracy"

# The corpus pipeline surface
cat corpus.ndjson | fnlp batch --task extract --task-args '{"schema_file":"s.json"}' > out.ndjson
fnlp job start corpus.manifest.ndjson --task extract --output out.ndjson
fnlp job status <job-id> --json
fnlp job resume <job-id>
fnlp job verify <job-id>
# Phase-7 only after single-host scope proofs:
fnlp job partition <snapshot-id> --parts 8 -o shards/
fnlp job merge shards/*.receipt --snapshot <snapshot-id> -o merged/

# Schema/task/evidence surfaces (staged: schema in Phase 4, recipe/eval in Phase 5)
fnlp schema check s.json
fnlp schema sample s.json -n 5
fnlp recipe explain support-routing.fnlptask.json --json
fnlp eval --task classify --dataset tickets.test.ndjson --gold label
fnlp qualify --baseline active --candidate candidate-id --suite support-suite.json

# Model artifacts
fnlp pull                     # public default only after LG-1; private pinned manifests remain explicit
fnlp convert --source /path/to/pinned-snapshot -o model.fnlpq --quant int8 --arch generic
fnlp models                   # installed artifacts, recipe ids, hashes
fnlp models activate candidate-id --qualification qualification.json

# Agent & ops surfaces
fnlp robot schema             # self-describing event/contract schema (versioned)
fnlp robot health             # artifact present? recipe? KV cost table? thread budget?
fnlp robot backends           # detected ISA tiers + the measured dispatch table
fnlp robot selftest           # re-prove dispatched kernels ≡ scalar oracle on THIS cpu
fnlp runs --limit 10 --json   # optional metadata-only history
fnlp doctor                   # idempotent self-check/repair
# AA-R1 Phase-7 research only, if multi-process traces justify local IPC:
fnlp resident start --endpoint auto
```

## Installation

There is nothing to install yet: no `Cargo.toml`, toolchain file, installer, or release binary exists. Phase 0 creates the crate and pins the inspected FrankenSuite commits; Phase 6 creates and verifies installers.

The intended source workflow, **after Phase 0**, is:

```bash
git clone https://github.com/Dicklesworthstone/franken_nlp
cd franken_nlp
cargo build --locked --release
```

The target release installer follows FrankenOCR's proven first-run shape. It installs and verifies the small `fnlp` binary first. If—and only if—LG-1 has authorized a public model catalog, an interactive terminal then gets a clear `y/N` offer to run the installed binary's own `fnlp pull`. `--with-model` explicitly enables that transfer in automation; `--no-pull` suppresses it; quiet/non-interactive installs never silently start a multi-gigabyte download. Shell and PowerShell do not implement model downloading themselves: both invoke the same Rust artifact manager used by a later manual `fnlp pull`.

That artifact manager streams the GitHub Release parts in manifest order, verifies every part and the reassembled whole, validates the `.fnlpq` census/provenance/license identity, derives the measured host-native packing, runs its selftest, and only then atomically activates it. The canonical local artifact lands at:

```text
Unix:    ~/.cache/franken_nlp/models/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
Windows: %LOCALAPPDATA%\franken_nlp\models\nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
```

Before LG-1 clears, the binary-only installer does not offer a nonexistent public model. It prints the sovereign `scripts/fetch_model.sh` / `scripts/fetch_model.ps1` → `fnlp convert` instructions and the explicit private-manifest form instead. The two workflows are intentionally different: maintainers/users converting locally download the pinned 8.34 GB bf16 conversion closure; ordinary authorized installs download only the already-converted `.fnlpq` chunks.

The intended embedding surface is:

```toml
# Cargo.toml
[dependencies]
franken_nlp = { git = "https://github.com/Dicklesworthstone/franken_nlp" }
```

```rust
use franken_nlp::{NlpEngine, ClassifyRequest};

fn main() -> franken_nlp::Result<()> {
    // Synchronous, blocking API; process-shared runtime/kernel resources stay internal.
    let engine = NlpEngine::builder().build()?;   // resolves the installed .fnlpq

    let result = engine.classify(ClassifyRequest {
        text: "The invoice total does not match the PO.".into(),
        labels: vec!["billing".into(), "bug".into(), "praise".into()],
        ..Default::default()
    })?;
    println!("{} ({:.2})", result.label, result.confidence); // calibrated
    Ok(())
}
```

## Target end-to-end workflow

Once implemented, the sovereign path is: fetch and fully hash the immutable HF revision → convert twice-identically to the canonical Generic `.fnlpq` → derive the selected native packing → run `robot selftest` → execute a frozen inference fixture. If LG-1 authorizes publication, the release path adds: deterministic 1,957,046,720-byte splitting → remote asset re-download/reassembly verification → fresh-machine installer → `fnlp pull` → no-flag model discovery → the same inference fixture → a second byte-perfect cache hit with no inference network. Only then does ordinary task/batch use begin. The release documentation will add copy-paste commands only when CI has executed those exact commands.

## Target configuration

The CLI snapshots environment into its builder; the library uses explicit builder values and does not read process environment behind the caller's back.

| Env | Default | Meaning |
|---|---|---|
| `FNLP_MODEL_DIR` | `~/.cache/franken_nlp/models` (Unix); `%LOCALAPPDATA%\franken_nlp\models` (Windows) | artifact search/install root; explicit builder/`--model-dir` wins |
| `FNLP_THREADS` | measured table | shared kernel-pool cap; direct thread sweep, not USL extrapolation |
| `FNLP_CTX` | `8192` target | per-sequence cap; KV pages allocate lazily |
| `FNLP_MEMORY_BUDGET` | host-derived, explicit in robot output | hard engine admission budget across KV/scratch/cache/output |
| `FNLP_BATCH` | `1` (CLI) / auto (daemon) | max in-flight sequences for Conveyor |
| `FNLP_QUANT` | best installed | artifact selection (int8 / int4) |
| `FNLP_FORCE_ARCH` | auto | pin the ISA tier (proof/bench runs) |
| `FNLP_NUMA` | measured/portable | bind-local / replicate / interleave; admission includes replication bytes |
| `FNLP_MMAP` | off | opt-in read-only weight mmap (audited island) |
| `FNLP_KV_INT8`, `FNLP_INT8_ATTN`, `FNLP_INT8_LMHEAD` | off / per-recipe | staged-quantization kill-switches |

## Performance

No FrankenNLP performance number exists yet. The eventual ledger uses randomized paired A/B trials through thermal steady state, matched prompt/output/quant/thread/allocator conditions against the pinned `nanbeige42` fork, and reports distributions—not cherry-picked best runs. Results are keyed by host fingerprint, artifact/recipe, kernel table, compiler, and exact command. Phase-6 `fnlp tune` may retain only proved bit-identical per-shape kernels/thread caps after repeated warm trials clear a practical effect threshold; transient workload/page-cache/NUMA choices are not permanent machine facts, and ambiguity retains shipped defaults.

| Gate | Requirement |
|---|---|
| PG-1 · Fidelity | L0 exact; full L1 metric vectors; 44 layer + two loop-norm L2 states; f32 L4 exact where oracle reproducible |
| PG-2 · Integer kernels | Scalar/SDOT/I8MM/AVX2/VNNI integer accumulators exactly equal i64, including full-domain extremes and tails |
| PG-3 · Decode (R1) | Target: meet/beat the fork at matched quant on every M4/M5/Zen host for which the project publishes a claim |
| PG-4 · Corpus (R2/R3) | Target: meet/beat the fork completion engine under identical prompts, while separately attributing grammar/prefix/task-layer gains |
| PG-5 · Structural levers | Batch, prefix, sliced/trie lm_head, forced runs, paging, jobs, and NUMA each get an isolated baseline curve; no fixed 5×/90%/1% target is invented before profiles |
| PG-6 · Tails/resources | p50/p95/p99, peak RSS, admitted/rejected bytes, energy where available, and cancellation/fairness under load |
| PG-7 · Quality | Every task uses locked scorecards; calibration and test are disjoint; structured success validates independently |
| PG-8 · Footprint | Converter and loader print measured section bytes, cache/KV commitments, binary size, and load/first-token distributions |

For orientation only, a hypothetical 3.7 GB/token mixed recipe divided into nominal memory bandwidth yields about 74 tok/s at 273 GB/s, 111 at 410 GB/s, 55 at 205 GB/s, and 90 at 333 GB/s. These are bandwidth-only ceilings, not measurements; realized artifact traffic, attention, compute, thermals, and effective bandwidth lower them.

## Determinism, trust & verification

- **Truth pack first.** Phase −1 promotes pinned observations to line-backed evidence and measures the oracle floor. Later empirical research items remain open until their phase; license absence is recorded as negative evidence, not papered over.
- **Three-oracle triangulation.** The pinned CPU HF reference (with its own nondeterminism floor measured first), the authors' llama.cpp fork, and — opportunistically — the independent `rlx-nanbeige` Rust implementation, differentially compared on loop semantics.
- **The ladder.** L0 exact → L1 per-op metric vector → L2 44 layer outputs + two norm states → L3 logits → L4 f32 greedy → L5 task outputs. Quantized modes carry their measured contracts rather than inheriting f32 exactness.
- **The right proof for each claim.** Checked bounds for capacity/overflow; scalar/i64 differential and properties for integer SIMD; bounded model checking + hostile interleavings for scheduler/cancellation; independent validation/fuzz for grammar/source grounding; exact sparse=full, forced=sequential-KV, trie=naïve, batch/prefix, and uninterrupted=resume fixtures. Statistical monitoring does not replace deterministic proof.
- **Task scorecards.** Named/versioned/licensed data with disjoint development/calibration/test ids; recipe, prompt, thinking/numerics mode, calibration validity, and confidence intervals recorded.
- **Typed trust, scoped evidence.** Untrusted-segment control-id exclusion is a byte-level invariant; content steering is a matched per-task attack metric. Same-model semantic verification must show incremental labeled-data value. Acceptance audits require frozen designs and human-authorized grades.
- **Target `fnlp robot selftest`.** Re-run integer dispatched kernels against scalar/i64 and print exact artifact/kernel/CPU provenance; floating candidates report their separate tolerance suite.

## Limitations

A few honest boundaries:

- **One model, by design.** `fnlp` runs Nanbeige4.2-3B and nothing else. A new checkpoint means a new truth pack, new parity fixtures, and new artifacts — a deliberate ratchet, not a config change. If you need arbitrary-model chat, use the fork or Ollama; this is an NLP appliance, not a runtime.
- **A 3B model has a ceiling.** On hard extraction/judging, frontier cloud models will beat it on raw accuracy. The pitch is *usable accuracy × zero marginal cost × data-never-leaves* — and published scorecards on named datasets so you know exactly where the ceiling is instead of discovering it in production.
- **POS/dependency/lemma pipelines are out of scope, forever.** Finite-state tools do those better per dollar by orders of magnitude; pretending otherwise would violate the honesty doctrine.
- **Long context is priced, not free.** BF16 KV is exactly 176 KiB/token/sequence: 704 MiB at 4K, 5.5 GiB at 32K, 44 GiB at 256K before overhead. Pages allocate lazily under admission; map-reduce is a primary operating mode.
- **Thinking mode trades latency for unproven task benefit.** Conservative structured-task default is off until a locked scorecard justifies changing it; thinking has hard time/token bounds.
- **Grounded does not mean semantically correct.** A `verbatim` field can be guaranteed to occur in the source and still select the wrong occurrence/value. Repeated occurrences stay explicit, and task accuracy still needs a scorecard. Constraint tension is not a fabrication probability.
- **Control-token containment is not a prompt-injection firewall.** The target encoder prevents untrusted bytes from becoming template control ids; ordinary instruction-shaped document text can still steer the model. Per-task attack-success rows expose that residual risk, and every derived string remains untrusted data.
- **A second read is not a certificate.** Optional semantic-field verification uses the same model and can share its errors. It stays off unless a domain scorecard proves incremental benefit, and its four-state result is evidence for review—not truth.
- **An audit is population-scoped.** Optional Phase-7 acceptance sampling requires a frozen owned job, preregistered design, and human grades. It estimates a stated error-rate property of that population; it never proves every item correct.
- **Durable does not make stdout exactly-once.** Canonical per-item commits apply only when `fnlp job` owns its checksummed spool/materialization. An arbitrary downstream pipe receives stable ids and documented at-least-once replay.
- **Unchanged input does not imply unchanged corpus-global output.** Local extraction can be reused exactly; a changed child set forces map-reduce reducers and entity clustering to rerun. Entity IDs are snapshot-qualified unless a separate lineage record relates snapshots.
- **Many local callers do not imply a server.** The v1 sharing surface is one cooperating batch pipe. An owner-only resident process is Phase-7 research only after traces show duplicate multi-GB loads or pool contention; no routable inference service is authorized.
- **Public model assets are not yet authorized.** Card metadata alone is not being treated as a complete redistribution file. Local conversion is the design fallback until LG-1 clears.
- **Multilingual quality is measured before it is marketed.** The tokenizer and model card promise broad coverage; `translate` stays out of the portfolio until honest evals justify it (Phase 7).
- **No GPU.** CPU is the product. Metal via `ft-kernel-metal` is a Phase 7 experiment gated on the CPU product being finished; CUDA is out of scope entirely.

## FAQ

**Is this production-ready today?** No — see the tense note at the top. This repository currently contains the complete engineering plan, AGENTS.md, LICENSE, and this README; Phase −1 (the source/oracle truth pack) is the next milestone, and the plan's phase gates (−1 → 6) are the honest progress tracker.

**Why one model instead of a zoo?** Because the premise is specialization: fixed shapes, exact loop scheduling, per-arch packing, and prompts evaluated against one recipe. Every generality knob spends performance and verification budget.

**Why not just use llama.cpp?** The inspected upstream does not support this architecture; the authors' fork does. That fork is therefore the semantics/performance baseline. FrankenNLP's reason to exist is the specialized CPU/task/validity/batch contract, but it must prove the engine advantage with the same prompts, quantization class, and fair threads—and publish losing rows.

**Why is a 3B model enough for real NLP work?** Because the tasks are extraction-shaped, not open-ended-generation-shaped, and because this particular 3B is anomalous: 44 effective layers of compute in a 3B-non-embedding footprint, with model-card scores (SWE-Bench Verified 63.6, GPQA-Diamond 87.4) above models 2–4× its size. Our own task scorecards — not the card — are the numbers that ultimately matter, and they ship with the releases.

**What do "valid" and "grounded by construction" actually mean?** At every decode step, Stencil executes only grammar-legal choices; invalid JSON is unrepresentable. For a declared `verbatim` field, the logical unescaped bytes must also traverse a bounded source-substring language, making off-source text unrepresentable. That proves syntax/source membership—not that the model chose the right fact—and repeated source occurrences remain explicit.

**Where will quantized weights come from?** The sovereign route downloads one revision-scoped HF closure with exact length/SHA-256 checks, then deterministically converts it to the Generic `.fnlpq`. If LG-1 later authorizes public derivatives, a separately versioned model release carries fixed 1,957,046,720-byte chunks plus a canonical manifest/receipt/license bundle. The consuming binary embeds that immutable manifest; the installer simply invokes `fnlp pull`, which streams and verifies part/whole/source/license/census identities, derives the local native packing, and atomically activates the model. No public asset exists today.

**How does this relate to frankensearch?** Cleanly: frankensearch owns embeddings/retrieval; `fnlp` owns generation-adjacent NLP (extraction, scoring, judging, redaction). A RAG pipeline uses frankensearch to retrieve and `fnlp judge --faithfulness` to verify — they meet over NDJSON.

**What about my data?** The target inference path opens no network. Pull is explicit; metadata history is opt-in and schema-forbidden from storing document/prompt/result text; prefix caching defaults to shipped task prefixes rather than user content. These become tested contracts before release.

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

The `franken_nlp` source code is licensed under the **MIT License with an OpenAI/Anthropic Rider**, Copyright (c) 2026 Jeffrey Emanuel (see [`LICENSE`](./LICENSE)). The rider withholds all rights from OpenAI, Anthropic, their affiliates, and anyone acting on their behalf, including any use of the software or derivative works in a machine-learning dataset, training corpus, evaluation harness, or pipeline. In any conflict between the rider and the rest of the license, the rider controls.

Nanbeige materials are **not licensed by this repository's source license**. The pinned model card declares Apache-2.0, but the pinned model repository currently contains no LICENSE/NOTICE file; public transformed weights and embedded tokenizer bytes are blocked pending plan §5.7/LG-1 evidence and project-owner review. This repository currently distributes no model weights.

## See also

- [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md), the master plan: evidence-state dossier, exact loop/KV/norm contract, conditional artifact mechanics, AVX2/AVX-512 campaigns, bounded batch/task design, verification authority, alien-artifact recommendation cards, roadmap, risks, and research-decision register.
- [`AGENTS.md`](./AGENTS.md), conventions for human and AI agents working in this codebase, including the engineering doctrine and the testing policy.
- [`WIZARD_IDEAS_COD.md`](./WIZARD_IDEAS_COD.md), [`WIZARD_IDEAS_CC.md`](./WIZARD_IDEAS_CC.md), [`WIZARD_SCORES_CC_ON_COD.md`](./WIZARD_SCORES_CC_ON_COD.md), [`WIZARD_SCORES_COD_ON_CC.md`](./WIZARD_SCORES_COD_ON_CC.md), [`WIZARD_REACTIONS_CC.md`](./WIZARD_REACTIONS_CC.md), and [`WIZARD_REACTIONS_COD.md`](./WIZARD_REACTIONS_COD.md) preserve non-normative review provenance; plan §10.6 records the accepted, rewritten, deferred, and rejected dispositions.
