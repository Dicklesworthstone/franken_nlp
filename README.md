<!-- fnlp-claim: readme-target-system-specification; wording=targeted -->

# franken_nlp

<div align="center">

[![License: MIT + Rider](https://img.shields.io/badge/License-MIT_+_OpenAI/Anthropic_Rider-blue.svg)](./LICENSE)
[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
[![toolchain: nightly](https://img.shields.io/badge/toolchain-nightly-purple.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
[![unsafe: audited islands only](https://img.shields.io/badge/unsafe-audited_islands_only-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![model: Nanbeige4.2--3B (Apache--2.0)](https://img.shields.io/badge/model-Nanbeige4.2--3B_(Apache--2.0)-teal.svg)](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
[![output: schema--valid by construction](https://img.shields.io/badge/output-schema--valid_by_construction-red.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
[![deps: closed universe](https://img.shields.io/badge/deps-closed_universe-black.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)

**A pure-Rust, memory-safe, CPU-hyper-optimized NLP engine built around exactly one model. `franken_nlp` takes Nanbeige4.2-3B, whose model card reports unusually strong sub-4B results, transforms its weights into a custom quantized format redistributed as digest-verified, provenance-attested GitHub release assets, runs it through kernels written for its exact shapes (including its one structural novelty: 22 layers executed twice, with a final RMSNorm after each pass), and wraps it in a complete local NLP toolbox: schema-guaranteed structured extraction, source-grounded NER + entity resolution, sentiment distribution scoring, zero-shot classification, PII redaction, faithfulness judging, summarization, keyphrases, QA, and generation. One library and one CLI program (`fnlp`, also installed as `franken_nlp`); no Python, no CUDA, no GPU required, and no network after the one-time model pull.**

</div>

> **A note on tense (read this first).** This README is written in the **present tense, as if the entire design in [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md) is fully realized**: the 1.0 target state where every parity gate is green and every subsystem is live. This is a deliberate choice. It lets the document describe the *finished* system so it gets **trued-up in place as milestones land** (the plan's Phase −1 → Phase 6 exit gates) rather than rewritten from scratch later. Where the plan itself stages something as genuinely future work or a research card (document-major packs, acceptance audits, the resident experiment; all Phase 7), the README says so plainly. Everything else below is the spec of the system this repository builds.

> **Current build-proof state.** Ordinary implementation panes run no Cargo,
> RCH, DSR, or GitHub Actions commands. Once a code-first wave quiesces, the
> controller may choose one clean immutable SHA for an occasional DSR checkpoint
> whose sole validation entrypoint is `scripts/check.sh` over the named
> `production` feature graph. GitHub Actions and direct RCH are
> non-authoritative. [WIRING.md](./WIRING.md) currently records this authority as
> **BLOCKED** until an exact DSR `PASS|FAIL` receipt and fresh Beads/`bv`/Agent
> Mail wiring exist. Even a green code receipt is separate from model-present,
> target-host performance/platform, artifact, and human-review gates.

> **Current executable surface (static source inventory).** The binary includes
> `robot schema|health|backends`, `schema check|sample`, conversion/package
> scaffolding, and explicit local inference and corpus routes under `candidate`.
> Candidate execution requires the `asupersync-runtime` feature. The
> `candidate source-int4` namespace includes generation/chat, classification,
> single-document extraction, [NER/keyphrases/cited summaries/passage QA](#implemented-candidate-int4-source-tasks),
> text cohorts/corpora,
> [ordered extraction batches](#implemented-candidate-int4-extraction-batches),
> [durable extraction jobs](#implemented-candidate-int4-extraction-jobs),
> [durable source-task jobs](#implemented-candidate-int4-source-task-jobs), and
> [whole-document extraction](#implemented-candidate-int4-whole-document-extraction).
> Candidate results retain their explicit provenance; build verification,
> model quality and release qualification remain separate pending gates.
> The top-level task, batch and job examples below remain the disclosed 1.0
> target contract.

---

## TL;DR

**The problem.** The workhorse NLP jobs currently force a bad trade: pulling structured records out of messy text, finding and canonicalizing entities, scoring sentiment along dimensions you actually care about, classifying at corpus scale, redacting PII *before* text leaves the machine, verifying RAG answers against sources. Cloud LLM APIs cost per token, transmit the submitted text to a provider, and rate-limit the corpus. SpaCy-class pipelines are fast and local but do not provide LLM-style semantic reasoning. Running a local LLM through a general stack (Python + transformers, or a generic GGUF runtime) leaves model-specific CPU optimizations and a complete NLP product layer on the table, then hands you text that still needs parsing and validation. Nanbeige4.2-3B changes the calculus: its card reports strong reasoning/agentic results despite a 3.149B non-embedding stack, and official llama.cpp now supports its looped architecture. That gives `fnlp` a real, maintained baseline to beat, and no excuse to pretend the baseline does not exist.

**The solution.** The reported capability makes this small model worth evaluating across the job family; the locked task scorecards decide which tasks actually graduate. `franken_nlp` runs Nanbeige4.2-3B through model-specific Rust kernels in which every weight-side dimension is a compile-time constant, schedules its **looped architecture** explicitly (22 physical layers → final RMSNorm → the same 22 layers → final RMSNorm; 44 KV slots at `layer + loop×22`), which doubles the no-retention logical weight traffic affected by quantization because the decoder stack is visited twice per token, and exposes it through a task layer where **every structured output is grammar-constrained during decoding**: the JSON is valid by construction, not by retry, and fields declared `verbatim` cannot contain off-source bytes at all. Corpus throughput comes from a layer-major batch engine that applies each logical layer operation to all compatible in-flight rows before advancing, a prefix cache that prefills shared task prompts once and forks them copy-on-write, prefill-only scoring paths that skip generation entirely, and crash-resumable durable jobs for the overnight run. Packed-panel rereads, cache residency, and physical DRAM traffic are measured rather than inferred from that schedule.

**Why `fnlp`:**

| | `franken_nlp` |
|---|---|
| Model | Nanbeige4.2-3B: 4.17B params (3.149B non-embedding), 44 effective layer executions via the 2-pass loop, 256K max positions, thinking mode, tool calls; **Apache-2.0**, with model-card scores of 63.6 SWE-Bench Verified / 87.4 GPQA-Diamond at a fraction of competitors' size. |
| Ships as | Two self-contained executable names per target (`fnlp` + `franken_nlp`) over one shared library dispatch, plus a provenance-attested, digest-verified model artifact: `fnlp pull` streams fixed 1,957,046,720-byte release chunks, verifies part/whole/source/license digests, derives the host-native packing, and atomically activates. Offline thereafter. |
| Output contract | Successful structured results are **schema-valid by construction** (compiled grammar masks illegal tokens every step); `verbatim` fields are byte-exact source substrings by construction; offsets are UTF-8-byte + Unicode-scalar, independently byte-verified against the original source, with ambiguity explicit and no approximate relocation; budget/cancel returns a typed no-result, never truncated "success." |
| The loop, exploited | Decode logically visits the full 3.149B-parameter stack **twice per token**; the engine schedules both passes (per-`(layer, loop)` KV binding, two post-loop norm states, loop-corrected rooflines) instead of replaying a generic graph. Physical DRAM/cache traffic is counter-measured. |
| Corpus fabric | `fnlp batch`: an NDJSON daemon with layer-major continuous batching, fair prefill morsels, checked memory admission, lazy copy-on-write KV pages, and shipped-prefix caching. `fnlp job`: opt-in durable corpus runs with content-addressed semantic keys, transactional resume, verify, and owned materialization. Metadata-only is the default; source/result bytes are stored only through explicit owner-authorized spool or materialization options. |
| Determinism | Scoped and named: semantic greedy output is exact under a declared numerics profile; batch-M ≡ batch-1 and prefix-fork ≡ cold-prefill are gated invariants; canonical byte replay fixes ordering and scrubs volatile telemetry. |
| Hardware | Apple M4/M5 (NEON SDOT/SMMLA/autovec, measured per shape), AMD Threadripper/EPYC with **two exact AVX2 constructions for Zen 3** (raw saturating `vpmaddubsw` is banned) and AVX-512-VNNI benchmarked separately on Zen 4 (double-pumped), Zen 5 (native 512), and Intel; runtime feature detection within one two-name executable pair per target arch, not a separate binary per ISA tier. |
| Fidelity | A profile-scoped parity ladder against the pinned HF reference: tokenizer-exact, per-op metric vectors, hidden states at **all 44 layer executions plus both post-loop norms**, logits, greedy tokens, task outputs. `hf-bf16-eager` owns reproducible HF-match claims; `diagnostic-f32` is a structural/bisect oracle and any bf16 token flip is a named fixture; optimized backends preserve one fixed recipe's scalar semantics; deliberate int8/int4 divergence from bf16 is separately measured and ledgered. |
| Safety | `unsafe_code` and `unsafe_op_in_unsafe_fn` are denied crate-wide; only enumerated SIMD/mmap modules may allow `unsafe_code`, while every operation remains in an explicit block with a local safety proof and a DSR-run policy gate. Each integer kernel is differentially proven against scalar/i64. NUMA/topology/QoS/huge-page experiments require a safe FrankenSuite surface or remain off. |
| Dependencies | **Closed universe.** Pinned FrankenSuite foundations (frankentorch kernels, asupersync runtime/HTTP, optional fsqlite) plus exactly three commodity families: `clap`, `serde`/`serde_json`, `sha2`. The SentencePiece BPE tokenizer, chat-template builder, and grammar engine are built in-house. |

---

## Quick example

```bash
# One-time: fetch + verify + install the quantized model artifact; offline thereafter.
fnlp pull

# Structured extraction with a result that cannot be invalid, and grounded fields
# that cannot be invented (x-fnlp-source: verbatim makes off-source bytes unrepresentable).
fnlp extract --schema invoice.schema.json invoice_email.txt --json

# Entities with source-verified byte+scalar offsets; canonicalize across a corpus.
fnlp ner report.txt --types PERSON,ORG,GPE,DATE,MONEY
cat mentions.ndjson | fnlp batch --task resolve > entities.ndjson

# Sentiment along domain dimensions: one shared-prefix prefill per dimension,
# a bucket distribution out (plus a full-vocab audit mode and a sampled/justified mode).
fnlp sentiment earnings_call.txt --focus-area earnings_calls

# Classify 100K support tickets overnight on the Threadripper, crash-resumably.
fnlp job start tickets.manifest.ndjson --task classify \
    --task-args '{"labels":["billing","bug","feature_request","churn_risk"]}' \
    --output labeled.ndjson
fnlp job resume <job-id>        # resumes after the last committed item; in-flight work may repeat

# Redact PII before anything leaves the machine and run the residual-detector check
# over that command's redacted output.
fnlp redact transcript.txt --policy pii-default --map-out map.json \
  --verify -o transcript.redacted.txt

# Verify RAG answers against their sources.
fnlp judge --faithfulness --source retrieved_passages.txt answer.txt

# Chat with bounded thinking and explicit reproducible sampling.
fnlp chat --think --sample --preset nanbeige --seed 42 \
  "Plan a Python 2→3 migration for a 100kLOC codebase"

# Measure it on YOUR data before you believe anyone's benchmark.
fnlp eval --task classify --dataset tickets.test.ndjson --gold label

# Agent surfaces: the contract, diagnostics, and an on-CPU differential kernel self-check.
fnlp robot schema
fnlp robot selftest

# Sovereign path: fetch the pinned upstream snapshot and convert locally.
# The release reports whether cross-target identity is certified or names its canonical publisher target.
scripts/fetch_model.sh --dest /path/to/nanbeige-source
fnlp convert --source /path/to/nanbeige-source --recipe nanbeige42-int8-v1 \
  --arch generic -o nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
```

---

## The six bets

No single trick makes this worth building. The **composition** of six bets does; each is feasible precisely because the engine serves exactly one model.

| Bet | One-line statement |
|---|---|
| **B1 · One model, zero framework** | Every weight-side dimension is a compile-time constant (hidden 3072, q 6144, KV 1024, MLP 10752, head_dim 128, 22 layers × 2 loops, vocab 166,144): shape-specialized kernels, offline per-arch weight packing, bounded scratch, lazily paged KV under checked admission; no generality tax, and dynamic batch/context/candidate tails stay explicit and tested. |
| **B2 · The loop is the moat** | The looped architecture (44 KV slots; a final RMSNorm after *each* pass; the stack logically visited twice per token) is scheduled explicitly with loop-corrected cost models and model-specific execution paths, doubling the no-retention logical traffic affected by quantization while leaving realized cache/DRAM savings to measurement. |
| **B3 · Compile finite languages into execution** | Most NLP tasks need **logits over a finite set, not free generation**: classification reads sliced lm_head rows after one prefill; multi-token label sets compile to exact continuation tries; constrained states project every legal row and no illegal one; uniquely-forced tokens feed through causal micro-prefill. Every optimized route has an exact-equality gate and a universal fallback. |
| **B4 · Valid, grounded, and structurally contained** | The supported JSON-Schema subset compiles to a bounded automaton over a vocab byte-trie: schema-valid always, `verbatim` fields byte-exact from source, EOS legal only at accept states, untrusted document bytes structurally unable to become role/think/tool control tokens. (That contains marker smuggling; semantic-injection resistance is a separate per-task measurement.) |
| **B5 · Corpus-scale, crash-resumable fabric** | Layer-major batching amortizes logical weight visits across compatible in-flight rows; the prefix cache forks shared task prompts copy-on-write; the NDJSON daemon adds backpressure and per-doc isolation; durable jobs add semantic execution keys, transactional resume, and owned materialization, with text persistence strictly opt-in. |
| **B6 · Evidence-native honesty** | An eight-state evidence vocabulary, the L0–L5 ladder (44 layer states + two loop norms), measured-not-assumed ISA dispatch, losing rows kept in the ledger, disjoint calibration/test splits, and user-owned `eval`/`calibrate`/`qualify`: every claim carries its evidence state or is labeled TARGETED. |

---

## Design philosophy

These are the constitutional, non-negotiable constraints the whole system is built under. They read like restrictions; they are the moat.

1. **The dependency universe is closed.** `std`, the [exact dated nightly pin](docs/TOOLCHAIN.md), pinned FrankenSuite foundations ([frankentorch](https://github.com/Dicklesworthstone/frankentorch) serial/range leaf kernels; [asupersync](https://github.com/Dicklesworthstone/asupersync) as the execution foundation for orchestration, cancellation, budgets, CPU-team ownership, and the one network path `fnlp pull`; optional [frankensqlite](https://github.com/Dicklesworthstone/frankensqlite) for metadata/job state), plus exactly three commodity families: `clap`, `serde`/`serde_json`, `sha2`. The release graph contains no Rayon; the tokenizer, chat template, grammar engine, and every task are built in-house. The DSR checkpoint's dependency-policy gate fails if a new direct root or Rayon appears in the explicitly selected `production` graph.

Every platform facility is governed by the per-target [platform-surface registry](docs/PLATFORM_SURFACES.md): an unavailable reviewed surface disables its optimization or refuses its operation rather than weakening ownership, locking, or memory-safety guarantees.
2. **Correctness outranks speed, always.** The parity ladder gates every kernel; a faster backend that violates the same-recipe scalar comparison contract or changes its greedy tokens is reverted, no source landed, and recorded in the negative-evidence ledger. Integer stages are exact; floating stages carry named metric/tolerance contracts. Deliberate quantization-vs-bf16 changes face separate logit/token/task gates. Speed ships on top of named semantics, never instead of them.
3. **Valid-by-construction output has the same rank.** A constrained-decode change that could emit schema-invalid JSON is reverted like a parity break. There is no "retry on parse failure" anywhere in the engine, by law.
4. **The loop is the architecture.** Never size or schedule as a conventional 22-layer model: it is 44 layer executions, 44 KV slots (176 KiB/token bf16), and two post-loop norm states, everywhere, pinned against the reference source before any kernel existed.
5. **Determinism is scoped, not hand-waved.** Semantic, byte, batch, and prefix claims each name their numerics/order/telemetry conditions. Approximate math never inherits the exact profile's promise.
6. **Measured-faster wins; width is not routing.** Apple candidates (autovec/SDOT/SMMLA) are benchmarked per shape; Zen 4's double-pumped AVX-512 is measured separately from Zen 5's native datapath; AVX2 ships two exact constructions and the raw saturating shortcut is banned outright.
7. **Honesty is enforced, not aspired to.** Every accepted numeric divergence lives in `docs/DISCREPANCIES.md` with a rollback path that actually exists (a kernel selector, a CLI option, or activation of a prior immutable artifact); every rejected optimization in `docs/NEGATIVE_EVIDENCE.md`; benchmark comparisons are thread/allocator/precision-fair against the strongest real baseline with slower rows published; task-quality claims name their dataset, prompt hash, recipe id, and thinking mode.
8. **Trust boundaries name exactly what they prove.** Typed document encoding prevents control-token smuggling, not instruction-shaped steering (measured per task). A same-model verifier is correlated evidence, not proof. A corpus audit's claim stays scoped to its frozen population and human grades.

---

## How it works

`franken_nlp` is seven named subsystems around one model, one artifact, and one batch fabric (plan §4–§9; codenames name regions of the module tree; they don't add structure).

```
                        ┌──────────────── fnlp CLI / library (sync, blocking) ────────────────┐
   text / NDJSON ──►  ATELIER · the task layer
                        │  extract · ner · resolve · sentiment · classify · judge · redact
                        │  summarize · keyphrases · answer · generate · tokens · split
                        │  presets-as-data · prompt hashes · calibration · map-reduce · TaskIR
                        ▼
     LEXICON · SentencePiece BPE (embedded, id-exact) + native chat-template builder
               trusted control segments · byte-preserving forbidden-control-id document path
     STENCIL · schema/source languages → bounded execution programs
               full projection · every-legal-row projection · forced causal runs · source copy
                        ▼
     CONVEYOR · batch fabric: layer-major continuous batching · COW prefix/KV pages ·
                bounded NDJSON daemon · durable snapshot-keyed jobs (resume/verify/materialize)
                        ▼
     OUROBOROS · the loop-scheduled model core
                embed → 22 layers → norm → same 22 layers → norm → lm_head (full | sliced)
                RMSNorm→RoPE(θ=7e7, split-half)→GQA 48:8 @128 (44-deep KV) → SwiGLU 10752
                kernels: int8/int4 tiled GEMM/GEMV — NEON SDOT/SMMLA/autovec · AVX-512-VNNI
                (Zen4/Zen5/Intel measured separately) · AVX-VNNI · two exact AVX2 routes · scalar
                        ▼
     samplers (greedy/seeded · grammar-mask AND) → validators (schema · offsets · calibration)
   ── FOUNDRY · pinned source → deterministic convert → .fnlpq → split release assets → pull
   ── ASSAY · L0–L5 (44+2 states) · proofs/properties/bounded retained model checks · task evals · gauntlet
   ── process-shared asupersync resources · optional metadata-only fsqlite
```

- **Foundry:** the weight pipeline. `fnlp convert` loads the pinned bf16 shards, census-checks all 201 tensors, quantizes through staged immutable recipes (`int8-mlp`, then `int8-mlp-attn`, then `int8-all`, then int4 by measured allocation), each stage its own parity-gated content-addressed artifact, embeds the tokenizer, template, config, and the Apache-2.0 license bundle, and emits the canonical Generic `.fnlpq`. Releases certify cross-OS/ISA digest identity or explicitly name the canonical publisher target and narrower local-reproduction claim. They ship fixed 1,957,046,720-byte chunks with per-part and whole SHA-256 plus a DSR-generated SBOM/SLSA/project-signature bundle over the binary and manifest/receipt/model inventory; `fnlp pull` streams, verifies, derives the measured host-native packing, and activates atomically. Installers delegate to that one Rust artifact manager: shell scripts never touch model bytes.
- **Ouroboros:** the model core, named for what it is: the snake that runs its own layers twice. The loop schedule is explicit (per-`(layer, loop)` KV binding resolved at engine build; final RMSNorm after each pass; two logical stack visits per token, never 2M separate forward schedules), attention is GQA 48:8 at head_dim 128 (explicitly 128; the config overrides the Llama fallback of 64, and the whole engine is built on that fact), KV lives in lazily-allocated copy-on-write pages under a byte-certified admission budget, and the lm_head has two personalities: full 166K-row GEMV fused with argmax for generation, and sliced/trie-scored rows for finite-candidate tasks.
- **Conveyor:** the throughput fabric. One process `EngineResources` host gives every engine the same asupersync admission domain and aggregate memory ledger, so ten embedded engines cannot create ten compute teams or each promise the same RAM. An admitted run crosses a proved real `Cx::spawn_blocking` pool boundary (never its inline lab fallback), then one `Cx::scoped_cpu` team spans the whole request or bounded batch epoch and checkpoints at tile/morsel boundaries; leaf kernels never spawn and the release graph contains no Rayon. Because blocking work is cooperative rather than preemptible, its closure retains the admission/memory/output guards until the team has actually joined, even if cancellation reached the wrapper first. Each engine step executes one logical layer operation across all compatible rows before advancing; the selected packed microkernel may reread panels, and receipts measure actual cache/DRAM traffic rather than promising one physical read. Long prefills are chunked into fair morsels; the prefix cache forks shipped task prompts copy-on-write (user-content caching is explicit opt-in, namespace-isolated). `fnlp batch` wraps it in a bounded NDJSON daemon; `fnlp job` adds durable, content-addressed corpus runs whose cache authority follows declared dependency scope: item-local results reuse exactly; corpus-global results (entity clusters, reduce outputs) rerun when the snapshot changes.
- **Stencil:** the constrained execution compiler. The supported JSON-Schema subset (plus the `verbatim` source language) compiles to a bounded automaton whose per-state execution primitive is chosen exactly: full projection, every-legal-row sparse projection, forced-token causal feeding, or source-copy. No heuristic ever drops a legal token. EOS only at accept states; unsatisfiable/budget/cancel returns a typed no-result.
- **Lexicon:** the text boundary. A pure-Rust SentencePiece BPE implementation (token-id-exact against the reference slow tokenizer; embedded in the binary so `fnlp tokens`/`split` and schema compilation work before any model is installed) and a typed chat-template builder for the pinned `<|im_start|>`/`<think>` template with thinking/tool modes. Only trusted template code can emit control ids; untrusted document bytes are encoded through a path that preserves them exactly while excluding every role/think/tool token, or preflight rejects when both conditions cannot be satisfied.
- **Atelier:** the task layer. The model-backed task families compile through one bounded internal `TaskIR` (exact prompt segments, decode strategy, grammar, budgets, dependency scope); presets are data with hashed, versioned prompts; per-task calibration keeps stated confidences honest; a map-reduce spine handles over-context documents. A public data-only recipe format opens once built-in equivalence and no-code/no-network gates pass.
- **Assay:** the conscience. The pinned CPU HF oracle is primary (its own nondeterminism floor measured first); a deliberately simple in-repo scalar specification engine localizes failures; official post-support llama.cpp is the secondary CPU/GGUF differential and performance baseline. The authors' fork is recorded as shared lineage, while GPL `rlx-nanbeige` remains an out-of-tree black-box cross-check. The L0–L5 ladder pins all 44+2 states; typed claims, completeness-graded receipts, and structural-cost witnesses keep public wording tied to evidence; and the same scorecard machinery ships to users as `fnlp eval`/`calibrate`/`qualify` so upgrades are qualified on *your* corpus before activation.

The full specification lives in [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md): the evidence-state dossier, the exact loop/KV/norm contract, the artifact lifecycle, the AVX2/AVX-512 campaigns, the constrained-execution compiler, the verification authority table, the alien-artifact recommendation cards, the roadmap, and the research-decision register.

## The task portfolio (target 1.0 contract)

At 1.0, every task below is available one-shot, as `--json`, and through the
batch daemon; tasks graduate individually on locked scorecards (plan §9.6).
These commands are not the current executable inventory. The implementation
status near the top of this README and `WIRING.md` remain authoritative until
the corresponding phase gates pass.

| Command | What it does | Decode strategy |
|---|---|---|
| `fnlp extract --schema s.json` | **Flagship.** Supported JSON-Schema subset → valid JSON by construction; `x-fnlp-source: verbatim` fields are byte-exact source substrings; optional same-model semantic verification where scorecards prove it earns its cost | constrained JSON/source |
| `fnlp ner` | Typed source-grounded spans with exact UTF-8-byte + Unicode-scalar offsets; repeated occurrences stay explicitly ambiguous rather than silently guessed | constrained JSON/source |
| `fnlp resolve` | Entity canonicalization across a doc/corpus: deterministic blocking → pairwise judging → snapshot-qualified clusters | blocked pairwise judging |
| `fnlp sentiment --focus-area X` | Adjective-dimension scores on [-100,+100] (swiss_army_llama lineage): candidate-conditional bucket distributions at prefill cost, a full-vocab audit mode, and a sampled/justified mode | sliced distribution / sampled |
| `fnlp classify --labels …` | Zero-shot single/multi-label with calibrated probabilities; large taxonomies compile to exact continuation tries | prefill-only, sliced/trie |
| `fnlp judge` | Rubric scoring, RAG faithfulness (`entailed/contradicted/unsupported` + evidence), order-debiased pairwise preference | mixed |
| `fnlp redact` | PII detection (LLM ∪ rule detectors) + masking/pseudonymization, offset-exact, optional reversible map, `--verify` residual scan | constrained + rules |
| `fnlp summarize` | Length/style presets, `--cited` byte-verified span evidence, map-reduce over long docs | free text |
| `fnlp keyphrases` | Ranked, source-anchored keyphrases | constrained list |
| `fnlp answer` | Context-grounded QA with span citations and **calibrated abstention** | constrained + free |
| `fnlp generate` / `chat` | Completion/chat: greedy product default; explicit `--sample --preset nanbeige`, bounded thinking, XML/JSON tool calls after the OQ-10 parser/fixture gate (parsed, never executed), addressable `--seed`, streaming | free |
| `fnlp tokens` / `split` / `normalize` | Non-LLM, no-model-load utilities: exact token counts, conservative byte-span-preserving splitting, and the deliberately narrow CRLF/ASCII-whitespace normalization contract; speed/language coverage measured rather than assumed | n/a |

**Deliberate non-goals:** POS tagging, dependency parsing, lemmatization (specialized classical tools are usually the right cost/latency choice; keep spaCy for that slice); embeddings (that's [frankensearch](https://github.com/Dicklesworthstone/frankensearch)'s job; `fnlp` interoperates over shared NDJSON conventions); training/fine-tuning; a model zoo.

## How it compares (target 1.0)

Honest framing. `fnlp` is deliberately built as an NLP *product* around this specific model, rather than only as a chat/completion runtime.

| | `fnlp` | official llama.cpp | Python + transformers | spaCy | Cloud LLM APIs |
|---|---|---|---|---|---|
| Runs Nanbeige4.2-3B | Yes (only this) | Yes (supported upstream) | Yes | No | n/a |
| Ships as | One Rust program under two executable names (`fnlp` + `franken_nlp`) + attested/digest-verified artifact | C++ build + GGUF | Python env | Python env | SaaS |
| NLP task layer | **First-class, schema-guaranteed** | None (completion) | DIY prompting | Classical pipeline | DIY prompting |
| Valid-JSON guarantee | **By construction** (+ source-grounded fields) | GBNF (generic, DIY) | DIY constrained decoding/validation | n/a | Vendor JSON modes |
| Corpus engine | **Layer-major batching + prefix cache + durable resumable jobs** | Server mode (generic) | DIY | Excellent (shallow tasks) | Rate-limited, $ |
| Loop-aware execution | **Model-specific kernels + task fusion** | Native model support in a general runtime | Reference model code | n/a | n/a |
| Source-grounded offsets / calibrated confidence | **Task-scoped and scorecard-gated** | No | DIY | Offsets yes; calibration varies | Vendor-specific |
| Semantic reasoning | Yes (3B-class, task-evaluated) | Yes | Yes | Statistical/transformer pipeline-dependent | Yes |
| Default data path | Local/offline after pull | Local | Local | Local | Submitted to provider |

## The `fnlp` CLI (target contract)

> The block below specifies the intended 1.0 surface; most commands shown are
> not runnable today. The current executable inventory is listed near the top
> of this README. At 1.0, robot mode emits line-oriented, versioned NDJSON an
> agent can pipe and validate against a frozen contract (`fnlp robot schema`).
> stdout is data, stderr is diagnostics, exit codes are stable and documented,
> and bare `fnlp` prints help and never opens a TUI.

```bash
# Tasks (each: --json, -o file, --think/--no-think; batch via `fnlp batch --task <t>`)
fnlp extract --schema s.json doc.txt
fnlp ner doc.txt --types PERSON,ORG,DATE | jq '.entities[]'
fnlp sentiment doc.txt --focus-area support_tickets
fnlp classify doc.txt --labels urgent,normal,spam --multi
fnlp judge --pair answer_a.txt answer_b.txt --criterion "factual accuracy"

# The corpus surfaces: composable pipe, or durable job
cat corpus.ndjson | fnlp batch --task extract --task-args '{"schema_file":"s.json"}' > out.ndjson
fnlp job start corpus.manifest.ndjson --task extract --output out.ndjson
fnlp job status <job-id> --json && fnlp job resume <job-id> && fnlp job verify <job-id>

# Schema & evidence tooling
fnlp schema check s.json            # compile-or-fail with the exact unsupported keyword
fnlp schema sample s.json -n 5      # print valid instances of the compiled grammar
fnlp eval --task classify --dataset tickets.test.ndjson --gold label
fnlp calibrate --task classify --dataset tickets.cal.ndjson -o support.cal.json
fnlp qualify --baseline active --candidate nanbeige-int4-v2 --suite support-suite.json
fnlp models activate nanbeige-int4-v2 --qualification qualification.json

# Model artifacts
fnlp pull                            # streamed, verified, atomically activated
fnlp convert --source /path/to/pinned-snapshot --recipe nanbeige42-int8-v1 --arch generic -o model.fnlpq
fnlp models derive --arch auto       # disposable content-addressed host packing
fnlp models                          # installed artifacts, recipes, digests, active state

# Agent & ops surfaces
fnlp robot schema                    # self-describing versioned contract
fnlp robot health                    # artifact, recipe, KV cost table, thread budget
fnlp robot backends                  # detected ISA features + the measured dispatch table
fnlp robot selftest                  # rerun shipped dispatched-kernel ≡ scalar/i64 differential cases
fnlp doctor                          # idempotent self-check/repair

# Phase-7 research surfaces (plan-staged, evidence-gated):
#   fnlp audit plan/grade · fnlp job partition/merge · fnlp resident start
```

### Implemented candidate INT4 extraction batches

`fnlp candidate source-int4 extract-batch` connects the native constrained
extractor to an ordered NDJSON stream. It keeps one model, native engine and
vocabulary resident for the run and compiles the shared schema against each
document. This path requires a candidate build with `asupersync-runtime`.

Each input document supplies `id` and exact UTF-8 `text`. IDs must be unique
within an epoch; an explicit flush starts a new epoch after delivery completes:

```jsonl
{"id":"ticket-1","text":"Alice works in Shanghai."}
{"flush":true}
{"id":"ticket-1","text":"Bob works in Paris."}
```

Run a local pinned source closure with a shared schema:

```bash
fnlp candidate source-int4 extract-batch documents.ndjson \
  --source-dir /models/Nanbeige4.2-3B \
  --source-manifest /models/nanbeige-source-manifest.json \
  --schema extraction.schema.json \
  --memory-mib 8192 \
  --max-requests 1000 \
  --max-total-input-bytes 67108864 \
  --max-total-output-bytes 67108864 \
  > results.ndjson
```

Use `-` instead of the input filename for stdin. Add `--source-membership`
when the schema has `x-fnlp-source: "verbatim"` fields; those fields are bound
to each document's original bytes and retain matching source offsets. Schema
and source-membership validation establish structural and byte-level evidence;
semantic accuracy still requires task evaluation. Any non-null per-record
`task_args` is rejected before document tokenization, including an override
that repeats the shared schema.

A saved candidate snapshot can serve the same command:

```bash
fnlp candidate source-int4 extract-batch documents.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --schema extraction.schema.json \
  --memory-mib 8192
```

The expected digest must come from an independently trusted conversion record.
Keep `--group-width` consistent with that conversion. Source conversion also
accepts `--save-snapshot NEW_FILE`; an existing path is refused.

Every output line wraps a canonical inner event under `record` and carries
`protocol: "fnlp-candidate-int4-extract-batch-v1"`, the actual model-content
digest, group width, kernel and candidate provenance. Successful documents
emit `record.event: "doc"`; recoverable refusals emit `"doc_error"`. A drained
stream ends with `"run_complete"`, whose summary includes succeeded and failed
counts. **Whole-run success requires both `run_complete` and exit code 0.**
A run with any failed document returns a nonzero exit even after normal EOF.
Fatal errors emit `"run_error"` when the output is still writable; a partial
write stops output without appending another terminal frame.

`--max-input-bytes` and `--max-line-bytes` bound each document and transport
record. `--max-requests` counts all nonempty records, including flush controls
and malformed input. Native work, mask visits, checkpoints and total input/
output budgets span the entire run and do not renew at flush boundaries.
Memory admission includes the resident model, full context, preparation,
transport staging and guarded results. Execution is sequential and blocking
IO observes cancellation cooperatively. For retained results and explicit
resume, use the separate extraction-job surface below.

## Implemented candidate INT4 extraction jobs

`fnlp candidate source-int4 extract-job start|resume` connects the same
native schema/source extraction to the authenticated owned-job runner.
This candidate route requires both `metadata-store` and
`asupersync-runtime` on Linux x86-64 or AArch64. It accepts either the
explicit pinned source closure or an independently digest-pinned INT4 snapshot.

Prepare an existing protected job directory, a protected key file containing
exactly 32 random raw bytes, a random 128-bit job ID encoded as 32 lowercase
hexadecimal characters, and an immutable `JobLimits` JSON file. The command
requires explicit `--store-results` consent and creates neither the directory
nor the secret. See [owned job storage](docs/owned-jobs.md) for the existing
storage and verification contract.

```bash
fnlp candidate source-int4 extract-job start documents.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --schema extraction.schema.json \
  --memory-mib 8192 \
  --store-results \
  --job-dir /private/extraction-job \
  --job-id "$JOB_ID" \
  --key-file /private/extraction-job.key \
  --limits extraction-job-limits.json \
  --materialize
```

Use the same arguments with `resume` in place of `start` and provide the
**complete original NDJSON population**, including already committed items.
Inputs are retained in admitted memory during the invocation; the command does
not persist the population for future recovery. The protected journal binds
the original input bytes/order/IDs, exact shared schema and source policy,
actual model/group/strict-v2 identity, compiler settings, and lifetime limits.
Changing that contract refuses before replay or orphan-tail repair.
Non-null `task_args` and live-stream flush controls are rejected.

Each pending item is durably debited before native execution. Its result
frame is synchronized before the journal commits its pointer. Resume skips
committed ordinals, while work spent on interrupted attempts remains spent.
Uncommitted spool bytes or interrupted stages require the explicit
resume-only `--discard-uncommitted-tail` flag; authentication still happens
first. No failed task is retained as a successful result.

The limits file supplies all five native lifetime work counters plus mask
visits. Group-dependent INT4 dot counts are copied exactly into the existing
journal counter shape; use `JobWork::from_int4` with an actual group-aware
`Int4Work` quote when constructing limits through the library. Population,
attempt, stream and checkpoint allowances do not multiply or renew those
counters. This route's `--max-input-bytes` bounds each **complete NDJSON
envelope**, and immutable `max_input_bytes_per_item` must fit within it.
Memory admission includes the complete frozen population, journal and JSON
staging, both preparation lifetimes, resident model, full context and guarded
results; increase `--memory-mib` only within the host's available capacity.

Stdout contains a bounded metadata completion with
`protocol: "fnlp-candidate-int4-extract-job-v1"`, candidate provenance and
authenticated progress. Private output stays in the owner-only spool;
`--materialize` publishes verified ordered `materialized.ndjson` after all
items commit. Existing stored-job status, verification and materialization
remain available independently of model loading. A failed acknowledgement
does not imply rollback; inspect the protected job or resume explicitly.

The library entry point is `NlpEngine::job_int4_extract`, with a sealed
`Int4ExtractionJobPlanner`, charged resident model, caller-owned input,
explicit `SourceJobRequest`, and `JobHostLimits`. It uses one physical
blocking invocation and retains ownership through native cleanup and durable
publication. The command and API remain source-inventoried candidate features:
build, model-present correctness, crash and quality qualification are pending.

## Implemented candidate INT4 source tasks

`fnlp candidate source-int4 ner|keyphrases|summarize|answer` now uses the
pinned source-task prompts and independent task finalizers with the grouped
INT4 native engine. These commands require `asupersync-runtime`, explicit
source-conversion or snapshot inputs, and a process memory ceiling, just like
the existing INT4 extraction path.

```bash
fnlp candidate source-int4 ner document.txt \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 8192
```

Use `keyphrases` or `summarize` with the same document input. The alternative
model route is `--source-dir DIR --source-manifest FILE`; it also accepts
`--save-snapshot NEW_FILE`. Keep `--group-width` consistent with the selected
conversion. The expected snapshot digest comes from an independently trusted
conversion record.

NER returns typed mentions with all exact matching byte and Unicode-scalar
intervals; repeated mentions remain explicitly ambiguous. Keyphrases retain
exact source text and occurrence evidence. Summary bullets carry byte-verified
source citations. Answer citations must fit inside an original passage and
cannot cross a passage separator or quote the question as evidence. Citation
membership establishes a structural guarantee; factual support and task
quality require separate evaluation. A successful answer may abstain with an
explicitly uncalibrated signal. Native, budget or citation failure returns an
error and never becomes a successful abstention.

Single-request `answer` reads a JSON object instead of plain document text:

```json
{"question":"Where does Alice work?","passages":[{"id":"p1","text":"Alice works in Shanghai."}]}
```

```bash
fnlp candidate source-int4 answer question.json \
  --source-dir /models/Nanbeige4.2-3B \
  --source-manifest /models/nanbeige-source-manifest.json \
  --memory-mib 8192
```

Each task accepts `--options FILE` with its complete typed options. For
example, an NER options file can contain
`{"types":["person","location"],"max_entities":8,"max_mention_scalars":128}`.
Unknown fields, duplicate keys and incomplete options reject. Source bytes,
including whitespace, Unicode and literal control-marker text, stay intact.
The trusted planner excludes template control tokens from untrusted input.
The bounded commands use constrained greedy decoding with thinking disabled.
`--max-new-tokens` includes terminal EOS; `--max-result-bytes` bounds the whole
typed result. Prompt plus reserved output must fit the selected context.

### Ordered INT4 source-task batches

`ner-batch`, `keyphrases-batch`, `summarize-batch` and `answer-batch` reuse one
resident model, native engine and vocabulary throughout an ordered NDJSON run:

```bash
fnlp candidate source-int4 keyphrases-batch documents.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 8192 --max-requests 1000 \
  --max-total-input-bytes 67108864 --max-total-output-bytes 67108864 \
  > phrases.ndjson
```

For NER, keyphrases and summary, the ordinary record is `{"id":"d1","text":"..."}`.
Shipped options apply unless a complete typed `task_args` or `--defaults FILE`
supplies bounded alternatives. Batch defaults use `SourceBatchArgs`, including
the task selector, options and budget; they differ from the single-request
`--options` file. Records cannot switch task families or raise admitted
budgets. The fixed resident context must remain fully covered by each
record's KV budget.

For `answer-batch`, each record's `text` is the **question** and its
`task_args.passages` holds the separate evidence. There is no implicit QA
passage default. This complete example uses the default 2048-token context
and 512-token output allowance:

```jsonl
{"id":"q1","text":"Where does Alice work?","task_args":{"task":"answer","passages":[{"id":"p1","text":"Alice works in Shanghai."}],"options":{"max_answer_scalars":2048,"max_citations":8,"max_quote_scalars":256},"budget":{"max_input_tokens":1536,"max_output_tokens":512,"max_output_bytes":1048576,"max_grammar_states":4096,"max_kv_bytes":369098752}}}
```

Output frames use `protocol: "fnlp-candidate-int4-source-batch-v1"` and keep
the canonical inner event under `record`. Every frame names the fixed task,
actual model-content digest, group width, kernel and candidate provenance.
`{"flush":true}` completes an epoch before IDs may be reused. Native work,
mask visits, checkpoints and total transport allowances remain cumulative
across epochs. **Whole-run success requires `record.event: "run_complete"`
and exit code 0.** Earlier completed records remain valid after a later
failure, and any failed document makes the process exit nonzero.

The library surfaces are `SourceTaskPlanner::plan_int4_with_control`,
`NlpEngine::execute_int4_source_task`, and `NlpEngine::batch_int4_source` with
`Int4SourceCorpusConfig`. Execution identities retain strict-quantized-v2,
the actual model/group partition and task-specific prompt, grammar and policy
digests. This addition covers bounded single-context tasks and ordered
item-local streams. Their opt-in durable jobs are described below; long-document
variants of these four source tasks remain separate work. Build verification,
model-present correctness and quality qualification remain pending gates.

## Implemented candidate INT4 source-task jobs

`fnlp candidate source-int4 ner-job|keyphrases-job|summarize-job|answer-job`
adds explicit `start` and `resume` operations for the four source-task families.
One resident INT4 model serves the complete frozen population through the
existing authenticated owned-job runner. Both pinned source conversion and
independently digest-pinned snapshots are supported. This route requires
`metadata-store` and `asupersync-runtime` on Linux x86-64 or AArch64.

Prepare the same protected directory, random job ID, protected 32-byte raw
secret, and immutable `JobLimits` file used by
[INT4 extraction jobs](#implemented-candidate-int4-extraction-jobs).
For a small job of at most three items, this complete limits file provides
explicit ceilings. The work values are allowances, not measured inference
costs; set the required ceilings before starting and preserve them on resume:

```json
{
  "max_items": 3,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 65536,
  "max_snapshot_bytes": 1048576,
  "max_result_bytes": 1048576,
  "max_spool_bytes": 16777216,
  "max_materialized_bytes": 16777216,
  "max_journal_bytes": 67108864,
  "max_attempts": 6,
  "max_work": {
    "model": {
      "forward_positions": 131072,
      "projected_logits": 10000000000,
      "attention_pairs": 1000000000000,
      "projections": {
        "dot_products": 10000000000000000,
        "multiply_accumulates": 10000000000000000
      }
    },
    "mask_node_visits": 12000000000
  }
}
```

Save it as `ner-job-limits.json`. Retention remains explicit:

```bash
fnlp candidate source-int4 ner-job start documents.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 8192 \
  --store-results \
  --job-dir /private/ner-job \
  --job-id "$JOB_ID" \
  --key-file /private/ner-job.key \
  --limits ner-job-limits.json \
  --materialize
```

Use the same arguments and complete original population with `resume`.
Already committed items remain in that input and are skipped after
authentication. The command admits the population in memory; it does not
persist the input for future recovery. IDs must be unique across the whole
population, and live-stream `flush` controls are refused.

Records use the same typed `{id,text,task_args?}` contract as
[ordered source-task batches](#ordered-int4-source-task-batches).
`--defaults FILE` supplies complete `SourceBatchArgs`, including its task
selector, options and budget. The task is fixed by the command, and record
arguments may tighten admitted task limits while retaining the exact full
resident KV allowance. For QA, `text` is the question; its separately typed
passages must be present in each record's arguments or in an explicitly
supplied defaults file. `--options` is refused here because it lacks the
complete durable task contract.

For example, an `answer-job` record using the default 2048-token context and
512-token output allowance is:

```jsonl
{"id":"q1","text":"Where does Alice work?","task_args":{"task":"answer","passages":[{"id":"p1","text":"Alice works in Shanghai."}],"options":{"max_answer_scalars":2048,"max_citations":8,"max_quote_scalars":256},"budget":{"max_input_tokens":1536,"max_output_tokens":512,"max_output_bytes":1048576,"max_grammar_states":4096,"max_kv_bytes":369098752}}}
```

Resume binds the original input bytes, IDs and order, actual model/group
identity, source-task policy, task options, private defaults and all compiler
settings. A changed QA passage, task budget or recipe refuses before replay
or orphan-tail repair. Private defaults and source content participate in
keyed commitments inside the protected job; they are not exposed as raw
content hashes or copied into progress reports.

The immutable limits file controls all five native work axes plus mask
visits across the job's lifetime. Interrupted attempts remain debited, and
larger invocation deadlines, input limits or attempt counts do not replenish
work. Pending outputs pass the same source-task finalizers before the
runner syncs the result frame and commits its journal pointer. Admission
retains result storage through that durable commit. A failed inference or
citation check never becomes successful QA abstention.

Stdout uses `protocol: "fnlp-candidate-int4-source-job-v1"` and contains the
fixed task, public model provenance and authenticated progress only.
Private results stay in the owner-only spool. `--materialize` publishes
verified ordered `materialized.ndjson` only after every item commits.
Existing stored-job status, verification and materialization commands work
without loading the model. Failed final output delivery or a deadline after
commit does not imply rollback; inspect or explicitly resume the job.
`--discard-uncommitted-tail` remains an explicit resume-only policy.

The library entry points are the sealed `Int4SourceJobPlanner::for_model`
and `NlpEngine::job_int4_source`, with `SourceJobRequest` and
`JobHostLimits`. Planning binds the real resident model and complete fixed
context before input/storage access. One physical hosted invocation owns
population ingestion, authenticated resume, native execution, durable commit
and optional materialization. These are candidate source-inventoried
features; build, model-present correctness, crash and quality qualification
remain pending.

## Implemented candidate INT4 whole-document extraction

`fnlp candidate source-int4 extract-map` applies one exact schema to every
chunk of a complete UTF-8 document. The pinned source encoder derives a
lossless partition, and the planner checks each chunk against the actual
schema and prompt capacity. All chunks and cumulative work are admitted
before the first native forward. One resident model and engine serve the
whole document, and only a fully verified completed result is published.

For example, this bounded schema permits a list of exact source strings in
each chunk:

```json
{"type":"array","maxItems":8,"items":{"type":"string","maxLength":128,"x-fnlp-source":"verbatim"}}
```

```bash
fnlp candidate source-int4 extract-map long-document.txt \
  --schema exact-spans.schema.json --source-membership \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 8192 --max-chunks 64 \
  --max-input-bytes 1048576 --max-map-result-bytes 16777216 \
  > document-result.json
```

Use `--source-dir DIR --source-manifest FILE` for source conversion instead.
Optional `--save-snapshot NEW_FILE` occurs after the complete document
preflights; a later task failure preserves the completed snapshot. All model
inputs remain explicit candidate inputs with the actual model digest, group
width and kernel carried in output provenance.

The result retains an ordered sequence of **independent exact-schema JSON
strings**, chunk byte/scalar ranges, cumulative native work and source
evidence. Exact decimal spelling in schema-constrained JSON survives. With
`--source-membership`, each extracted value is checked against every compatible
occurrence **within the chunk that produced it**, and those intervals are
lifted to original-document coordinates. It does not discover occurrences in
other chunks from which that value was not extracted. Repeated occurrences
stay explicit; Unicode-scalar coordinates are distinct from byte coordinates.

Each chunk must independently satisfy the schema. For example, a required
verbatim constant present in only one chunk makes the complete request fail;
the command does not invent the field in other chunks, skip them, or repair
their outputs. Independent objects are not combined into a global object.
Chunked inference also has no overlapping-window or single-context equivalence
guarantee.

`--max-result-bytes` bounds each typed chunk result, while
`--max-map-result-bytes` bounds the complete document result. `--max-chunks`,
`--max-chunk-bytes` and `--max-tokenizer-calls` bound partition planning.
Native work ceilings cover forward positions, projected logits, attention
pairs, actual group-dependent dot products and multiply-accumulates.
`--max-total-mask-node-visits`, source field/occurrence verification budgets,
and live/total reduction bytes remain nonrenewable across chunks. Exceeding a
limit returns an error without publishing a successful prefix.

The CLI admits storage for every prepared schema/prompt, the whole reduction
frontier, complete occurrence vectors and final output, in addition to the
resident model and full context. Its default preparation reserve is 1024 MiB;
larger chunk counts or evidence/output limits may require an explicit larger
reserve and process memory ceiling. The library equivalents are
`Int4ExtractionBatchPlanner::plan_document_with_control` and
`NlpEngine::extract_int4_document` with `Int4ExtractionMapConfig`. This is
source-inventoried candidate functionality; build, model-present correctness
and quality qualification remain pending.

## Installation

**1. Install script: *not yet available*.** The planned script installs and SHA-256-verifies the `fnlp` binary, then offers (interactive `y/N`; `--with-model` / `--no-pull` in automation) to run the installed binary's own `fnlp pull`; shell never touches model bytes itself. It is deliberately not shown as a runnable command: `install.sh` does not exist yet and there are no release binaries for it to install. This section becomes a command again when Phase 6 ships one.

**2. From source — production build not yet available.** The crate has been
scaffolded, but the required named `production` feature graph has not landed.
The default-empty graph is not a production substitute. Once that graph and
its no-Rayon suite leaves are wired and proved at a clean immutable SHA, the
supported source-build command will be:

```bash
git clone https://github.com/Dicklesworthstone/franken_nlp
cd franken_nlp
cargo build --locked --release --bins --no-default-features --features production
```

This future command is a user-local build instruction, not current project
proof. Swarm panes do not run it, and its eventual success cannot replace the
controller's clean-SHA DSR receipt.

**3. Embedded, as a Rust library** (after the first crate release; pin the exact published commit rather than a floating branch):

```toml
# Cargo.toml
[dependencies]
franken_nlp = { git = "https://github.com/Dicklesworthstone/franken_nlp", rev = "<published-release-commit-sha>" }
```

```rust
use franken_nlp::{NlpEngine, ClassifyRequest};

fn main() -> franken_nlp::Result<()> {
    // Synchronous, blocking API; asupersync runtime/admission and CPU teams are owned internals.
    let engine = NlpEngine::builder().build()?;   // resolves the installed .fnlpq

    let result = engine.classify(ClassifyRequest {
        text: "The invoice total does not match the PO.".into(),
        labels: vec!["billing".into(), "bug".into(), "praise".into()],
        ..Default::default()
    })?;
    println!("{} score={} ({})", result.label, result.score, result.score_space);
    if let Some(probability) = result.calibrated_probability {
        println!("calibrated_probability={probability:.3}");
    }
    Ok(())
}
```

**Model artifact.** `fnlp pull` installs the canonical artifact to `~/.cache/franken_nlp/models/` (Unix) or `%LOCALAPPDATA%\franken_nlp\models\` (Windows); `--model-dir`/`FNLP_MODEL_DIR` override. The sovereign alternative, `scripts/fetch_model.sh` + `fnlp convert`, downloads and verifies the pinned 8,360,887,509-byte conversion closure (8.34 GB of weight shards plus tokenizer/config files) and converts locally. Each release states whether that conversion is certified to hash-match across supported OS/ISA/compiler targets; if it is not, the release names the canonical publisher target and gives the local result its honest, narrower equivalence claim.

## Configuration

The CLI snapshots environment into its builder; the library uses explicit builder values and never reads process environment behind the caller's back.

| Env | Default | Meaning |
|---|---|---|
| `FNLP_MODEL_DIR` | platform cache path above | artifact search/install root |
| `FNLP_THREADS` | measured table | process asupersync compute-team cap (direct sweep, not extrapolation), admitted within the aggregate runtime-worker + blocking-coordinator + scoped-child + helper-thread envelope |
| `FNLP_CTX` | `8192` | per-sequence cap; KV pages allocate lazily |
| `FNLP_MEMORY_BUDGET` | safely host-derived or required explicitly | process-aggregate admission budget across weights/KV/scratch/cache/output; never guessed when cgroup/job-object-aware authority is unavailable |
| `FNLP_BATCH` | `1` (CLI) / auto (daemon) | max in-flight sequences for Conveyor |
| `FNLP_QUANT` | best installed | artifact selection (int8 / int4) |
| `FNLP_FORCE_ARCH` | auto | pin an ISA tier for proof/bench runs (selftest first) |
| `FNLP_NUMA` | portable | bind-local / replicate / interleave only through a ratified safe FrankenSuite placement surface; unsupported modes reject, and admission prices replication |
| `FNLP_MMAP` | off | opt-in read-only weight mmap (audited island) |
| `FNLP_KV_INT8` | off | runtime KV-cache quantization toggle (weight precision is a property of the installed artifact recipe, selected with `fnlp models activate`, never an env var) |

## Performance

Every number below is a provisional gate, **TARGETED rather than OBSERVED: no FrankenNLP performance number exists yet, because no kernel exists yet.** The measurement discipline they will be held to is the plan's (§10): randomized paired A/B trials through thermal steady state, per-regime distributions (never best-of-N), thread/allocator/precision/prompt parity against a tested official llama.cpp revision at or after its Nanbeige4.2 support commit, results keyed by host fingerprint + artifact recipe + kernel table, and losing rows published. When measurements land, this section is trued up with OBSERVED rows and their fixtures.

| Gate | Requirement |
|---|---|
| PG-0 · Execution ownership | One admitted asupersync `scoped_cpu` team per request/bounded batch epoch; exact runtime-worker/blocking-coordinator/scoped-child/helper inventory and aggregate runnable-thread ceiling; presets benchmarked rather than blindly stacked; team formed and spawn-sealed before work release; no post-start/latch spawn; bounded checkpoints; disconnect-safe cancel/panic join; closure-owned leases until actual completion; no per-op thread creation or Rayon; one-time host config and aggregate multi-engine memory reservations proven |
| PG-1 · Fidelity | L0 exact; full L1 metric vectors; all 44 layer + two loop-norm L2 states; same-recipe integer stages exact, floating stages within named tolerances, and greedy tokens exact; `hf-bf16-eager` L4 exact on oracle-reproducible prefixes; `diagnostic-f32` uses its structural/logit contract; quantized-vs-bf16 agreement explicitly measured |
| PG-2 · Integer kernels | Scalar/SDOT/SMMLA/AVX2/VNNI accumulators exactly equal i64, including full-domain extremes and every tail |
| PG-3 · Decode (R1) | Meet/beat official llama.cpp against nearest format peers (our int8 vs its Q8_0 class) on every M4/M5/Zen host for which a claim is published |
| PG-4 · Corpus (R2/R3) | Meet/beat official llama.cpp's completion engine under identical prompts, with grammar/prefix/task-layer gains attributed separately |
| PG-5 · Structural levers | Batch, prefix, sliced/trie lm_head, forced runs, paging, jobs, NUMA: each gets an isolated baseline curve; no invented multipliers before profiles |
| PG-6 · Tails & resources | p50/p95/p99, peak RSS, admitted/rejected bytes, energy where available, cancellation/fairness under load |
| PG-7 · Quality | Locked per-task scorecards; calibration and test disjoint; structured success validates independently |
| PG-8 · Footprint | Converter/loader print measured section bytes, KV commitments, binary size, and load/first-token distributions |

For orientation only: a hypothetical 3.7 GB/token mixed int4/int8 recipe against nominal memory bandwidth gives bandwidth-only decode ceilings of ~74 tok/s at 273 GB/s (M4 Pro-class), ~111 at 410 GB/s (M4 Max-class), ~55 at 205 GB/s (Zen 3 5995WX-class), and ~90 at 333 GB/s (Zen 4 7995WX-class). These no-retention ceilings are context, not promises. When enough compatible rows are admitted, the batch fabric can amortize logical weight visits; sparse or latency-bound workloads may still behave much like batch 1.

## Determinism, trust & verification

- **Truth pack first.** Phase −1 promotes every pinned observation (config, 201-tensor index, loop driver, tokenizer, template, generation config, license declaration) to line-backed, hash-bound evidence and measures the oracle's own nondeterminism floor before any tolerance is set.
- **Lineage-aware differentials.** Pinned CPU HF is the semantic authority; a deliberately simple in-repo scalar engine localizes errors; tested official llama.cpp is the secondary CPU/GGUF differential. The authors' fork is shared lineage, not another vote; GPL `rlx-nanbeige` stays an out-of-tree black-box check.
- **The ladder.** L0 tokenizer/template exact → L1 per-op metric vectors → L2 all 44 layer outputs + both post-loop norms (the loop boundary is a named fixture) → L3 logits → L4 greedy exact within the named oracle/recipe/profile scope → L5 task outputs, with constrained-greedy canonical bytes exact only against the frozen golden for that same scope. Quantized and cross-profile comparisons carry their own measured contracts.
- **The right proof for each claim.** Checked bounds for capacity/overflow; scalar/i64 differentials and property tests for integer SIMD; one admitted asupersync CPU team with bounded checkpoints/full join/no Rayon, deterministic-lab DPOR-style guided coverage, retained bounded state-model/TLC results where used, and hostile native interleavings for the scheduler; independent validation + fuzz for grammar/grounding; exact sparse=full, forced=sequential-KV, trie=naïve, batch/prefix, and uninterrupted=resumed fixtures. Lab exports and coverage runs are never mislabeled as exhaustive model-check results. Statistical monitoring never substitutes for deterministic proof.
- **Typed public evidence.** [docs/CLAIMS.json](docs/CLAIMS.json) bounds public wording; `.fnlpr` receipts state whether they are replayable, structural, artifact-dependent, or audit-only while omitting private bytes by default; operation-cost witnesses distinguish full two-loop target calls from any separately counted loop-1 draft work and count projected rows plus KV bytes so equal outputs cannot hide duplicated work.
- **User-owned evidence.** `fnlp eval`/`calibrate`/`qualify` run the project's own scorecard machinery on *your* labeled data, with enforced calibration/test separation and digest-bound qualification receipts gating `models activate`.
- **`fnlp robot selftest`.** Any user, any machine can rerun the shipped dispatched-integer-kernel differential cases against the scalar/i64 oracle and print exact artifact/kernel/CPU/test-corpus provenance. That is reproducible evidence for a finite suite, not a universal proof over every input.
- **Evidence strata stay separate.** The occasional DSR checkpoint proves only
  the code and policy legs actually executed for its clean SHA and named
  production graph. Model-present parity/artifact smoke, target-host
  performance and platform-native behavior, and human review/authorization
  retain their own receipts and cannot be inferred from that green bar.

## Limitations

A few honest boundaries:

- **One model, by design.** `fnlp` runs Nanbeige4.2-3B and nothing else. A new checkpoint means a new truth pack, parity fixtures, and artifacts: a deliberate ratchet rather than a config change. If you need arbitrary-model chat, use official llama.cpp or Ollama; this is an NLP appliance, not a runtime.
- **A 3B model has a ceiling.** On hard extraction/judging, frontier cloud models can beat it on raw accuracy. The pitch is *usable measured accuracy, no per-request API fee, and an offline local data path*, while local compute, electricity, latency, and operations remain real costs. Published scorecards show the ceiling before production does.
- **POS/dependency/lemma pipelines are outside the current one-model roadmap (v1/v2).** Specialized classical tools usually beat a 3B generative model on cost and latency for this slice; keep spaCy or another purpose-built pipeline unless future measured evidence and an explicit plan revision justify changing scope.
- **Long context is priced, not free.** BF16 KV is exactly 176 KiB/token/sequence: 704 MiB at 4K, 5.5 GiB at 32K, 44 GiB at 256K. Pages allocate lazily under a checked budget; map-reduce is a primary operating mode, not an afterthought.
- **Grounded does not mean semantically correct.** A `verbatim` field provably occurs in the source and can still be the wrong occurrence or value; repeated occurrences stay explicit, and task accuracy still needs its scorecard.
- **Control-token containment is not a prompt-injection firewall.** Untrusted bytes structurally cannot become template control ids, but instruction-shaped prose can still steer the model; the residual is measured per task as attack-success rows, never marketed away.
- **A second read is not a certificate.** Optional semantic-field verification uses the same model and can share its errors; it ships per task only where locked evals prove incremental value.
- **Durable does not make stdout exactly-once.** One canonical committed record per item holds when `fnlp job` owns its checksummed spool/materialization; an arbitrary downstream pipe gets stable ids and documented at-least-once replay.
- **Unchanged input does not imply unchanged corpus-global output.** Item-local results reuse exactly; entity clusters and reduce steps rerun when the snapshot's child set changes, and cluster ids are snapshot-qualified.
- **Thinking mode trades latency for measured benefit.** Structured tasks default it off until a locked scorecard justifies otherwise; thinking always has hard time/token bounds.
- **The product generation default intentionally differs from upstream.** Nanbeige's generation config samples at temperature 0.6/top-k 20/top-p 0.95; `fnlp` defaults to greedy for reproducibility. `--sample --preset nanbeige` opts into that recipe, and receipts record the effective 256-bit seed.
- **Cancellation is cooperative.** Asupersync owns and drains the scoped CPU team; the engine never reuses its buffers or memory lease merely because a wrapper task reported cancellation. A stuck kernel that fails to checkpoint is a bug, not something safe Rust can forcibly preempt.
- **Process admission bounds FrankenNLP, not every other program.** The aggregate ledger prevents this process's engines from each promising the same RAM; it cannot stop unrelated processes from consuming memory after admission.
- **Multilingual quality is measured before it is marketed.** `translate` stays out of the portfolio until honest evals justify it (Phase 7).
- **No GPU required.** CPU is the reference implementation and the portable floor; every task runs at full fidelity on CPU alone. The Apple integrated GPU (Metal via `ft-kernel-metal`) is a planned M4/M5 acceleration layer that lands after the CPU core is parity-proven, under its own named numerics profile; CUDA is out of scope entirely.

## FAQ

**Is this production-ready today?** No; see the note on tense at the top. The repository is now in an active Beads-driven, code-first implementation campaign, with Rust scaffolding and several provisional/synthetic surfaces present. Phase −1 and Phase 0 gates remain incomplete, the production feature graph is not wired, and no clean-SHA DSR receipt or model-present parity/release proof exists. Beads and `WIRING.md` report current implementation/proof state; the plan's phase gates (−1 → 6) remain the authority for promotion.

**Why one model instead of a zoo?** Because the entire premise is specialization: compile-time shapes, a hand-scheduled loop, per-arch weight packing, prompts and calibration evaluated against one set of weights. Every generality knob added back spends performance and verifiability. franken_ocr proved this shape ships; `fnlp` inherits it.

**Why not just use llama.cpp?** Official upstream now runs Nanbeige4.2-3B and is our strongest maintained CPU/GGUF baseline. It is a general inference runtime rather than this project's complete NLP product: `fnlp` adds the schema/source execution compiler, task presets and scorecards, calibrated/abstaining surfaces, layer-major corpus paths, and digest-scoped durable jobs. We benchmark against official upstream honestly (same prompts, nearest format class since Q8_0 and our int8 are not bit-equivalent, thread-fair) and publish the rows where it wins.

**Why is a 3B model worth trying for real NLP work?** Many tasks are extraction/scoring-shaped rather than open-ended, and this checkpoint performs 44 effective layer executions in a 3.149B non-embedding footprint. Its model card reports SWE-Bench Verified 63.6 and GPQA-Diamond 87.4, but those benchmarks do not prove NER, extraction, sentiment, or calibration quality. Our locked task scorecards, not the card, decide which capabilities ship as supported.

**What do "valid by construction" and "grounded by construction" actually mean?** At every decode step, Stencil executes only grammar-legal choices: invalid JSON is unrepresentable rather than detected and retried. For a `verbatim` field, the logical unescaped bytes must also traverse a bounded substring language over the source document, so off-source text is equally unrepresentable. That proves syntax and source membership; choosing the *right* fact is what the scorecards measure.

**Where do the quantized weights come from, and why should I trust them?** `fnlp convert` transforms the exact pinned bf16 snapshot under a versioned canonical recipe. The release either certifies one Generic digest across supported targets or names its canonical publisher target and narrower local-reproduction scope. SHA-256 then proves that downloaded bytes match that manifest; it does not identify the publisher, so the controller-authorized DSR release emits an SBOM, SLSA provenance, and project-signature bundle binding the released binaries and manifest/receipt/model inventory. Publisher authentication still depends on obtaining the signing-key fingerprint through an independently trusted project channel; a signature and key fetched together from one untrusted mirror are not enough. The weights are Apache-2.0 as declared by the pinned model card; every artifact and release carries the license text, factual attribution, and modification notice, and `fnlp pull` verifies part, whole, source, license, and census identities before crash-safe atomic activation.

**How does this relate to frankensearch?** Cleanly: frankensearch owns embeddings/retrieval; `fnlp` owns generation-adjacent NLP (extraction, scoring, judging, redaction). A RAG pipeline retrieves with frankensearch and verifies with `fnlp judge --faithfulness`; the two meet over NDJSON.

**What about my data?** Inference opens no network, structurally: the only networked code path in the binary is the explicit `fnlp pull`. The metadata/run-history schema is opt-in and structurally cannot store document, prompt, or result text. A separately enabled owner-only job spool may store content when you explicitly request durable materialization; it is not telemetry or hidden history. Prefix caching defaults to shipped task prefixes, never your content; `redact` exists precisely so text can be sanitized *before* it goes anywhere else.

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

The `franken_nlp` source code is licensed under the **MIT License with an OpenAI/Anthropic Rider**, Copyright (c) 2026 Jeffrey Emanuel (see [`LICENSE`](./LICENSE)). The rider withholds all rights from OpenAI, Anthropic, their affiliates, and anyone acting on their behalf, including any use of the software or derivative works in a machine-learning dataset, training corpus, evaluation harness, or pipeline. In any conflict between the rider and the rest of the license, the rider controls.

The **Nanbeige4.2-3B model weights**, and every transformed `.fnlpq` derivative this project distributes, are **Apache-2.0** as declared by the official model card at the pinned revision, independent of the rider. Every artifact and release carries the license text, the factual attribution (model origin, pinned revision, author attribution per the pinned card: Nanbeige Team), and a modification notice; `fnlp licenses` prints the complete bundle, `fnlp --version` carries the one-line attribution, and the LICENSE file's third-party section states the same.

## See also

- [`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md), the master plan: the evidence-state model dossier, the exact loop/KV/norm contract, the artifact lifecycle, the per-arch SIMD campaigns, the constrained-execution compiler, the batch/job fabric, the verification methodology, the alien-artifact recommendation cards, the phased roadmap, the risk register, and the research-decision register.
- [`AGENTS.md`](./AGENTS.md), conventions for human and AI agents working in this codebase, including the engineering doctrine and the testing policy.
- The review-provenance records: [`WIZARD_IDEAS_CC.md`](./docs/misc/WIZARD_IDEAS_CC.md), [`WIZARD_IDEAS_COD.md`](./docs/misc/WIZARD_IDEAS_COD.md), the cross-scores, the reactions, and [`DUELING_WIZARDS_REPORT.md`](./docs/misc/DUELING_WIZARDS_REPORT.md): historical snapshots; plan §10.6 records the authoritative dispositions.
