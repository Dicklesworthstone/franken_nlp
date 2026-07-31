# COMPREHENSIVE PLAN FOR franken_nlp (FrankenNLP)

**Master engineering plan — v3.1 (artifact-lifecycle augmentation 2026-07-30)**
**Status:** architecture proposal / pre-Phase-0 / external review round 1 of at least 4 (greenfield; nothing implemented yet)
**Audience:** implementing agents (CPU-kernel, model-forward, task-layer, CLI, conformance) and the lead architect
**Target model:** `Nanbeige/Nanbeige4.2-3B` at HF revision `f56ec5a9650268aa098496734743c25ea778bd2d`

> **Evidence vocabulary (normative).** This revision replaces the overloaded word “verified” with eight explicit states: **[OBSERVED@pin]** means directly inspected in a named immutable source revision but not yet archived in this repository; **[PARTIAL]** means only the stated part is observed and the named remainder is unresolved; **[REPORTED]** means claimed by the model card, paper, or another secondary source; **[TARGET]** is a design/release requirement; **[HYPOTHESIS]** is an optimization or quality prediction that measurement may kill; **[OPEN]** is an unanswered question; **[BLOCKED]** names a missing authority or prerequisite. Phase −1 promotes an observation to **[EVIDENCED]** only by committing the source hash, exact source span, extraction command, and replayable fixture under `docs/truth-pack/`. No phase gate may rely on an unresolved `[OPEN]`, `[BLOCKED]`, or unresolved portion of `[PARTIAL]`.

> **Lineage and review record.** The first draft followed `franken_ocr`, the closest technical sibling. This revision also cross-read `franken_markdown` (current-vs-roadmap and output-safety contracts), `franken_lean` (claim-state taxonomy and foundation audits), `franken_manim` (normative decisions and convergence gates), and `frankengraphdb` (invariant, cost, and threat registries). Review round 1 corrected the unsafe-lint contradiction, the AVX2 saturation construction, AVX-512 tiering, loop/KV/norm semantics, license provenance, artifact trust, memory admission, grammar scope, calibration splits, and deterministic-output scope. A second pass within the same round applied the `idea-wizard`, `alien-artifact-coding`, `alien-graveyard`, and `extreme-software-optimization` disciplines: compile known structure into exact execution, add durable corpus semantics and user-owned qualification, and quarantine adaptive/exotic mechanisms behind profiles, deterministic fallbacks, and negative-evidence records. The adversarial cross-scores/reactions then caught forced-byte/token conflation, control-token trust-boundary omission, same-model-verification overclaim risk, audit-authority gaps, tuning-profile overreach, 44-deep trie-fork amplification, corpus-cache scope, and multi-client duplication; those are now bounded by OQ-19–24, AA-A1, and AA-R1. The v3.1 augmentation then traced FrankenOCR's shipped Baidu source-fetch, converter, immutable embedded manifest, 1,957,046,720-byte release split, `focr pull`, Unix/Windows installers, and clean-cache release receipts end to end; §5.1/§5.6 and Phases −1/2/6 now specify the equivalent Nanbeige lifecycle rather than merely naming it. It is **not** steady-state: the planning workflow still requires at least three more independent review rounds before Beads conversion.

### Research snapshot used by this revision

| Source | Immutable revision inspected | What is currently observed |
|---|---|---|
| HF model | `f56ec5a9650268aa098496734743c25ea778bd2d` | config, 201-name tensor index, modeling source, tokenizer model/config/template, generation config, card metadata |
| authors’ llama.cpp fork | `c6640a1c0cf7b38df342b67021a3900b04d092e7` (`nanbeige42`) | independent semantics/performance-baseline pin |
| frankentorch | `523aaf827faf538aa541126ee222fcd7af348410` | foundation symbols audited in §3.5 |
| asupersync | `8eb48575889c81b65f7556db4b26d47a8bc03197` | runtime/HTTP feature surface audited in §3.5 |
| frankensqlite | `5676cb97486a62c4f0a19c053184e0ff3cfb2852` | persistence surface audited in §3.5 |
| swiss_army_llama | `7bd155410ff2cdf71b4ddf4ccd5a626a600690b3` | behavioral sentiment reference; no root LICENSE/NOTICE observed, so copy authority is blocked |

These are research observations, not yet repository evidence. Phase −1 must fetch them again by immutable revision, hash the bytes, and fail if they do not match.

---

## Table of contents

1. [Mission & non-negotiable goals](#1-mission--non-negotiable-goals)
2. [Target model dossier — Nanbeige4.2-3B](#2-target-model-dossier--nanbeige42-3b)
3. [Why pure-Rust + frankentorch + asupersync (the generality-tax wedge, loop edition)](#3-why-pure-rust--frankentorch--asupersync)
4. [System architecture — crate layout & module breakdown](#4-system-architecture)
5. [Weight transformation pipeline & GitHub-asset distribution](#5-weight-transformation-pipeline)
6. [Model-specific CPU kernel strategy](#6-model-specific-cpu-kernel-strategy)
7. [The NLP task layer (the product)](#7-the-nlp-task-layer-the-product)
8. [The `fnlp` CLI design](#8-the-fnlp-cli-design)
9. [Verification & conformance methodology](#9-verification--conformance-methodology)
10. [Performance methodology](#10-performance-methodology)
11. [Phased roadmap](#11-phased-roadmap)
12. [Risks & mitigations](#12-risks--mitigations)
13. [Success metrics](#13-success-metrics)
14. [Research-decision register](#14-research-decision-register)
15. [Skills, methodology & the path to beads](#15-skills-methodology--the-path-to-beads)
16. [Primary-source index](#16-primary-source-index)

> **Where the differentiation lives.** §6 (loop-aware kernels, batched layer-major execution, exact finite-language projection, prefix-cache) is where "make this ONE model fly on M4/M5 and high-core-count AMD" lives. §7 (schema- and source-constrained decoding, exact continuation scoring, the task portfolio) is where "replace the SpaCy/cloud-API workflow with one offline binary" lives. §8 adds the unglamorous product moat: resumable, provenance-bound corpus jobs. §9–§10 are the conscience: user-owned qualification, the three-pillar gauntlet, and the honest per-regime perf ledger that keep every claim true.

---

## 1. Mission & non-negotiable goals

**Mission.** `franken_nlp` is a pure-Rust (Rust 2024, nightly), memory-safe, CPU-hyper-optimized **library + one CLI program (`fnlp`, also shipped under the long name `franken_nlp`)** that runs the **Nanbeige4.2-3B** language model **with no general ML framework** — and turns it into a complete local NLP toolbox: structured extraction, NER + entity resolution, sentiment/dimension scoring, zero-shot classification, PII redaction, faithfulness judging, summarization, keyphrases, QA, and plain generation. `unsafe_code = "deny"` applies at crate roots; only enumerated SIMD/mmap modules may opt into scoped audited islands (§1.1 G4).

We achieve this by transforming the model's bf16 weights into a custom quantized on-disk form (int8 first, int4 in refinement rounds), **targeting** distribution of the transformed artifacts as GitHub release assets (fixed 1,957,046,720-byte parts except the tail, each under GitHub's 2 GiB limit; hash-verified and reassembled by `fnlp pull`) **only after the §5.7 redistribution-authority gate is green**, and writing **model-specific kernels** whose only job is to run *this one model* as fast as possible on:

- **Apple Silicon / ARM64** — M4/M5 family: NEON, FEAT_DotProd (SDOT), FEAT_MATMUL_INT8 (SMMLA / i8mm), high-bandwidth unified memory, big SLC
- **AMD / Intel x86-64** — high-core-count parts first: Threadripper/EPYC Zen 3 (AVX2 ceiling — the reference `trj` machine is a 5995WX-class part), Zen 4/5 and Xeon (AVX-512-VNNI / AVX-VNNI), with AVX2 as a first-class, proof-carrying tier, not a fallback afterthought

CUDA is an explicit **non-goal for v1** and at most a far-future stretch; a Metal path via the new `ft-kernel-metal` crate is a **Phase 7 stretch experiment** only (§11). **CPU is the product** because most machines that need local NLP (laptops, CI runners, agent hosts, edge boxes, privacy-constrained servers) have no usable CUDA GPU — and because this model is small enough (≈ 4.17 B params, ≈ 3.1–4.7 GB for the default int4/int8 recipes; smaller recipes require separately-gated embedding/lm-head quantization) that a well-written CPU path is genuinely fast.

The engine is built on **frankentorch** (custom tensors / CPU kernels — consumed at the *kernel* level, not the autograd level) and **asupersync** (structured-concurrency runtime — orchestration / cancellation / IO only). Optional metadata/run/job state uses **frankensqlite** (`fsqlite` — never `rusqlite`); document text and model output are never persisted by default. Direct dependencies outside the FrankenSuite are frozen to the three owner-approved families in §3.4. It is **agent-ergonomic** (robot / JSON / NDJSON mode, stable versioned schema, explicit exit codes) and **embeddable** as a plain Rust library with a blocking, sync public API.

### 1.1 Non-negotiable goals (the bar a release must clear)

| # | Goal | Operational definition | Verification owner |
|---|------|------------------------|--------------------|
| **G1** | **Model fidelity matches the reference stack** | On a frozen golden prompt corpus, decoded tokens exact-match the pinned HF reference under greedy where the reference is deterministic; logits within a documented, *measured* quantization tolerance; task outputs (JSON, labels, scores) exact or within ledgered tolerance. | §9 conformance |
| **G2** | **CPU speed beats the proven CPU baseline — measured honestly, per stage and per regime** | Decode-per-token, prefill-per-token, and **batch corpus throughput (docs/min)** measured against the Phase −1 proven CPU baseline (the `Nanbeige/llama.cpp` `nanbeige42` fork GGUF, and CPU HF if it reproduces the oracle) with thread/allocator/precision fairness controls. The gating v1 claim is **decode-per-token and batch-scoring throughput faster than the fork at matched quantization**; end-to-end-faster-everywhere is a post-int4 stretch, not a v1 gate. | §10 performance |
| **G3** | **Pure-Rust, self-contained release executable, cross-platform** | One self-contained executable per target (linux x86-64/arm64, darwin x86-64/arm64, windows-msvc x86-64; two entrypoint filenames, `fnlp` + `franken_nlp`, thin shims over one dispatch), no Python, no network at inference time, no GPU required. “Static” is not promised on platforms whose system ABI is dynamically linked. **"No FFI" is defined precisely:** no foreign ML/runtime dependency in the default inference path (no Python, no libtorch/ONNX/BLAS, no GPU driver, no C tokenizer/grammar libs); OS facilities reached through std and the enumerated, audited opt-in islands (mmap; feature-gated NUMA/huge-page hints) are exceptions listed by name, never silent. Model artifacts pulled once, then fully offline. | §8, §11 Phase 6 |
| **G4** | **Memory-safe** | `unsafe_code = "deny"` in `[lints.rust]` + `#![deny(unsafe_code)]` at the crate root (Rust's `forbid` cannot be locally overridden, so `deny` is the strongest level that makes audited islands possible); `unsafe` is confined to enumerated modules. Integer SIMD islands have exact scalar/i64 differentials and load/alignment safety proofs; the mmap island has separate range/lifetime/immutability tests. A policy test fails on any unlisted `allow`. | §4, §6 |
| **G5** | **Agent-ergonomic** | `robot` subcommand emitting versioned NDJSON events with a self-describing `robot schema`; stable exit codes; `--json` everywhere; semantic results deterministic under the named greedy/numerics profile; canonical byte-replay uses ordered output with volatile telemetry omitted; NDJSON batch mode has backpressure. | §8 |
| **G6** | **Embeddable and re-entrant** | Library API (`NlpEngine::extract(...)`, `::classify(...)`, `::score(...)`, `::generate(...)`) is **synchronous and blocking**; the async runtime and kernel pool are owned implementation details; environment/config is snapshotted per builder; multiple engines and concurrent callers obey one explicit admission policy. Only immutable CPU capability detection may be process-global. | §3, §8 |
| **G7** | **Honest** | Every accepted numeric divergence from the reference has a `docs/DISCREPANCIES.md` ledger entry (reference behavior, our impl, **measured** impact, kill-switch env var); every rejected optimization a `docs/NEGATIVE_EVIDENCE.md` entry. No silent numerics changes. Task-quality claims cite the exact eval set and are reproducible. | §9, §10 |
| **G8** | **Valid-by-construction task output** | Every **successful** structured response validates against its declared schema; timeout/cancellation/resource exhaustion returns a typed no-result error, never partial JSON marked successful. Grammar constraints enforce generation; there is no unconstrained parse-retry loop. | §7 |
| **G9** | **Local privacy and bounded untrusted input** | No document text, output, reversible redaction map, or prompt body is logged or persisted by default; inference opens no network path; telemetry is explicit opt-in and metadata-only; artifact/schema/NDJSON parsers and the batch scheduler enforce documented byte, depth, token, queue, and memory limits before allocation. | §5.7, §8.6 |

> **G1 over G2, always.** Correctness outranks kernel speed (frankentorch's Non-Regression Rule). A faster kernel that drifts decoded tokens is reverted. We ship speed *on top of* parity, never instead of it. **G8 is the task-layer analog:** a faster constrained-decode path that can emit invalid JSON is reverted the same way.

### 1.2 Explicit non-goals (v1)

- **Not** a general inference runtime, not a model zoo, not a router. **One model** (Nanbeige4.2-3B), end-to-end. (franken_ocr grew a small certified zoo *after* its core was proven; if that ever happens here it is a post-v2 decision, and nothing in v1 may pay a generality tax for it.)
- **Not** a per-token classical-NLP pipeline. **POS tagging, dependency parsing, and lemmatization are explicit non-goals**: a finite-state/statistical tagger (spaCy-class) does those at ~10⁴–10⁵ tokens/s/core; a 3B autoregressive LLM cannot compete on cost and we will not pretend otherwise. What we replace is the *other* 90% of why people reached for spaCy + cloud APIs: NER, entity resolution, classification, extraction, redaction, scoring.
- **Not** an embedding engine. Dense retrieval embeddings are **frankensearch**'s job; `fnlp` interoperates (shared NDJSON conventions) rather than duplicating it.
- **Not** training or fine-tuning. Inference only. No autograd, no optimizers, no backward kernels — we reach past ft-api's session/tape layer straight to `ft-kernel-cpu`'s free-standing kernel functions.
- **Not** an HTTP server in v1. The persistent-process story is the **NDJSON batch daemon** (stdin/stdout, §8.4); an OpenAI-compatible `fnlp serve` is a Phase 7 stretch decision.
- **No GPU in v1.** No CUDA ever in scope; Metal only as the Phase 7 experiment gated on the CPU product being finished.

### 1.3 Why this model (the one-paragraph pitch)

Nanbeige4.2-3B is reported by its authors as an unusually strong sub-4B open-weights model for reasoning + agentic work **[REPORTED]** (SWE-Bench Verified 63.6, GPQA-Diamond 87.4, HMMT-Feb-2026 82.8, Terminal-Bench 2.0 44.1 — with the card reporting wins over larger comparison models). The model card declares Apache-2.0, a 256K context, and a controllable thinking mode; we independently reproduce the quality/context claims and settle redistribution authority before turning any of them into a release claim. Its **Looped Transformer** design (22 physical layers executed twice → 44 effective layers) is precisely the shape that rewards a bespoke CPU engine: the weight footprint of a 3B non-embedding decoder with the compute schedule of a roughly 6B one, so **every decoder-weight byte is paid twice per generated token** — which doubles the payoff of aggressive quantization, weight-layout specialization, and batch amortization (§3.1, §6). A general framework treats the loop as two generic passes; we treat it as *the* defining property of the machine we are building.

---

## 2. Target model dossier — Nanbeige4.2-3B

> This section uses the normative evidence vocabulary at the top of this document. Facts directly inspected at HF revision `f56ec…` are **[OBSERVED@pin]**, not yet **[EVIDENCED]**; author/card claims are **[REPORTED]**; unresolved authority is **[BLOCKED]**. Mandate: *ground every model-specific claim and expose its proof state.* Phase −1 pins, hashes, line-backs, and replays every load-bearing observation before a dependent kernel ships.

### 2.1 One-paragraph orientation

Nanbeige4.2-3B is a **dense, decoder-only, Llama-family transformer with one structural novelty active in this checkpoint: the loop**. The released checkpoint is a 22-layer GQA transformer (hidden 3072, 48 query heads / 8 KV heads, head_dim 128, SwiGLU intermediate 10752, RMSNorm, RoPE θ=7×10⁷, max positions 262144, vocab 166144, untied embeddings) whose layer stack executes **num_loops = 2** times per forward — 44 effective layer executions from 22 layers' weights **[OBSERVED@pin]**. `_get_loop_cache_layer_idx` maps `(loop, layer)` to `layer + loop × 22`, yielding 44 independent KV slots **[OBSERVED@pin]**. The modeling file also implements mHC hyper-connections, depth attention, LoopSplit, and hashed n-gram embeddings, but the 201-name checkpoint index contains none of their tensors and the released config enables none of them **[OBSERVED@pin]**. It is an instruction-tuned, tool-calling, thinking-mode model intended as a local personal assistant, trained with SFT + RL **[REPORTED]**.

### 2.2 Identity, format, size, and license state

| Field | Value | Source |
|-------|-------|--------|
| HF repo | `Nanbeige/Nanbeige4.2-3B` | HF |
| Architecture class | `NanbeigeForCausalLM` (`trust_remote_code`; `auto_map` → `modeling_nanbeige.py`, 122 kB; `configuration_nanbeige.py`, 26 kB) | `config.json` |
| `model_type` | `nanbeige` | `config.json` |
| Weight dtype | **bfloat16** (`torch_dtype`) | `config.json` |
| Checkpoint | **2 shards**: `model-00001-of-00002.safetensors` (4.97 GB) + `model-00002-of-00002.safetensors` (3.37 GB) = **8.34 GB**; `model.safetensors.index.json` 16.5 kB | HF file listing |
| Params | **≈ 4.17 B total / ≈ 3.149 B non-embedding** (recomputed from config in §2.6; card says "4B total, 3B non-embedding") | card + our census |
| Tokenizer | `tokenizer.model` (2.78 MB, SentencePiece) + `tokenizer.json` (18.5 MB) + `tokenizer_config.json` (11 kB, chat template) + `added_tokens.json` + `special_tokens_map.json`; **model card uses `use_fast=False`** → the slow SentencePiece path is the reference tokenizer **[OPEN OQ-6: byte-exactness of fast vs slow]** | file listing + card |
| Base model | `Nanbeige4.2-3B-Base` | card |
| License metadata | HF card declares **Apache-2.0**; the pinned repository contains **no `LICENSE`, `LICENSE.txt`, `NOTICE`, or `NOTICE.txt` file** | card metadata + pinned file census |
| Extras | `Nanbeige42_report.pdf` (588 kB technical report), `generation_config.json`, `.eval_results/` | file listing |
| Transformers pins | config records `4.42.4`; generation_config records `4.51.0` | files |

**License conclusion — [BLOCKED for public weight redistribution].** Apache-2.0, if it is in fact the operative weight license, permits redistribution and modification subject to its conditions. The pinned repository's card metadata says Apache-2.0, but the repository does not ship the license/NOTICE materials needed to establish the exact grant, copyright notice, and any upstream NOTICE content. That is strong evidence of intent, not enough authority for this plan to pronounce public derivative-weight distribution cleared. Source code may proceed under this repository's own license; local user-side conversion may proceed; **no `.fnlpq` weight asset is published until §5.7 records either (a) an upstream license file/NOTICE at an immutable revision, or (b) written confirmation from the rights holder, reviewed by the project owner.** If cleared, every artifact/release preserves the exact upstream materials and states our modifications. If not cleared, `fnlp pull` targets user-supplied/private manifests and public releases contain binaries only.

### 2.3 Architecture — the decoder **[OBSERVED@pin from config/index unless noted]**

| Field | Value |
|-------|-------|
| `hidden_size` | 3072 |
| `num_hidden_layers` | **22** (physical) |
| `num_loops` | **2** → 44 effective layer executions per forward |
| `num_attention_heads` | 48 |
| `num_key_value_heads` | 8 (**GQA 6:1**) |
| `head_dim` / `kv_channels` | **128 / 128** (explicit — NOT `hidden/heads` = 64; see ⚠ below) |
| Q/K/V/O shapes (derived) | `q_proj` 3072→6144, `k_proj` 3072→1024, `v_proj` 3072→1024, `o_proj` 6144→3072 |
| `intermediate_size` | 10752 (SwiGLU: `gate_proj`/`up_proj` 3072→10752, `down_proj` 10752→3072) |
| `hidden_act` | SiLU |
| Norms | RMSNorm, `rms_norm_eps = 1e-5`, pre-norm; `qk_layernorm` is absent/false and the 201-name index has no q/k norm tensors **[OBSERVED@pin]** |
| RoPE | `rope_theta = 70,000,000` (7e7); `rope_scaling = null` → plain RoPE at native 256K |
| `max_position_embeddings` | 262,144 (256K) |
| `vocab_size` | 166,144 |
| `tie_word_embeddings` | **false** (separate `lm_head` 3072→166144) |
| `attention_bias` / `mlp_bias` | false / false (default) → **no biases anywhere in the hot path** |
| Special ids | bos 166100, eos 166101, pad 0 |
| `loop_loss_weights` | `[]` (empty) |
| `skip_loop_final_norm` | false; the loop driver applies `self.norm(hidden_states)` after **each** loop, so loop 2 consumes loop 1's normalized output **[OBSERVED@pin]** |
| Sliding window | none — full causal attention **[OBSERVED@pin]** |

**⚠ `head_dim` = 128 is explicit and load-bearing.** The generic Llama fallback (`hidden_size // num_attention_heads` = 64) is WRONG for this model; the config overrides it to 128, making the query space 6144 = 2× hidden. This is corroborated by the parameter census (§2.6): only head_dim 128 reproduces the "3B non-embedding / 8.34 GB bf16" totals. Every attention kernel, KV layout, and RoPE table is built on 128. **[OBSERVED@pin]**

### 2.4 The loop — the one structural novelty active in this checkpoint

What the pinned config, checkpoint index, and loop driver say **[OBSERVED@pin; Phase −1 must promote to EVIDENCED]**:

- `num_loops = 2`; `_get_num_loops()` returns it because `loop_loss_weights=[]` and double LoopSplit is disabled: exactly two sequential passes over layers 0…21.
- `_get_loop_cache_layer_idx(layer, loop)` is `layer + loop * num_hidden_layers`; positions/cache-position are reused, but each pass writes its own 22-slot KV region: **44 independent KV entries per token position**.
- `skip_loop_final_norm=false`; after each 22-layer pass the loop driver executes the same final RMSNorm. Loop 2 consumes loop 1's normalized output directly. There is no embedding re-injection or loop-boundary projection.
- Depth attention, mHC/hyper-connections, LoopSplit, and n-gram embeddings are inactive, and their tensor-name families are absent from the 201-entry index. The card describes family-level innovations; this checkpoint's active parameterized surface is the minimal looped decoder.
- The loop's positions, cache positions, masks, and RoPE positions are the same logical positions in both passes. The L2 conformance ladder still names the boundary separately because an implementation can reproduce plausible text while applying this rule in the wrong place.

**Kernel implications (why the loop changes everything):**

1. **Per-token decode streams the full 3,149,008,896-parameter physical layer stack TWICE** (plus the reused 3,072-parameter final norm after each pass). Consequences: quantization saves bytes on both passes; batching shares each pass's stream across M sequences (one stream per physical layer **per loop**, two total, rather than 2M); rooflines include the ×2.
2. **KV cache is 44-deep, not 22-deep.** Per position: 44 × 2 (K,V) × 8 heads × 128 dim × 2 B = **180,224 bytes = 176 KiB/token** bf16. One sequence costs 704 MiB at 4K, 5.5 GiB at 32K, and 44 GiB at 256K before allocator/page metadata; int8 halves payload. This makes lazy paging/admission (§6.8–§6.9), not the advertised context number, authoritative.
3. **Prefill compute is also ×2** (≈ 12.6 GFLOP/token; §10.1) — prefill is compute-bound, so the tiled int8 GEMM quality (SMMLA/VNNI/AVX2) directly sets scoring throughput, where prefill dominates (§7 tasks are mostly prefill-heavy).
4. **The loop suggests—but does not automatically provide—a draft/verify seam.** Loop-1 hidden state can be projected through `lm_head` to propose tokens, but a correct speculative algorithm must verify a *sequence* with the full two-loop model and preserve the target distribution/token stream; merely accepting a loop-1 token on a confidence threshold is approximate decoding, not speculation. No loop-1 loss is configured, so draft quality may be poor. This remains a default-off research card (§10.5), gated on an exact rejection sampler/verification construction and allowed to die in the negative-evidence ledger.

### 2.5 Tokenizer, chat template, generation contract

- **Tokenizer:** the pinned `tokenizer.model` is SentencePiece **BPE** (`TrainerSpec.model_type=2`) with an identity normalizer, byte pieces, `add_bos_token=true`, `add_eos_token=false`, and `legacy=false` **[OBSERVED@pin]**. The card's own usage snippet passes `use_fast=False`, so the slow SentencePiece path is canonical. OQ-6 is narrowed to byte-fallback/normalizer edge semantics, added-token precedence, and fast-vs-slow id equality over an adversarial corpus. Our pure-Rust implementation must be token-id-exact vs the slow reference.
- **Chat template:** the pinned template uses `<|im_start|>` / `<|im_end|>`, `<think>` / `</think>`, `enable_thinking` (default on), `preserve_thinking`, and XML/JSON tool-call branches **[OBSERVED@pin]**. Phase −1 records the full mode matrix and exact generation suffix; `fnlp` implements this one fixed template as a typed Rust program, not a Jinja interpreter.
- **Generation defaults** (`generation_config.json` **[OBSERVED@pin]**): `do_sample=true, temperature=0.6, top_p=0.95, top_k=20`; eos 166101. Card guidance **[REPORTED]**: agentic → temp 1.0 / max 65,536 tokens; reasoning/chat → temp 0.6 / max 131,072. **Parity implication:** the reference decodes stochastically by default, so parity gates run **greedy (temperature=0)** where the oracle is deterministic, exactly as franken_ocr did; seeded-sampler equivalence is a separate, lower-tier gate (§9.2).
- **Thinking-mode policy for the task layer (§7.9):** scoring/extraction throughput wants `enable_thinking=false` (orders-of-magnitude fewer generated tokens); hard-reasoning tasks may justify thinking. Every task declares its default and exposes `--think`/`--no-think`. The measured accuracy-vs-cost tradeoff per task is a Phase 5 eval artifact, not an assumption.

### 2.6 Parameter & byte census — **[OBSERVED@pin by arithmetic from config/index; Phase −1 CI target]**

Per layer: attention 44,040,192 (q 18,874,368 + k 3,145,728 + v 3,145,728 + o 18,874,368) + MLP 99,090,432 (gate/up/down @ 33,030,144) + 2 RMSNorm weights (6,144) = **143,136,768**. × 22 layers = **3,149,008,896**; adding final norm gives **3,149,011,968 non-embedding parameters**. Embedding 166,144 × 3072 = **510,394,368**; untied `lm_head` another **510,394,368**.

**Total ≈ 4,169,800,704 params → 8.34 GB bf16 — matches the published shard sizes exactly.** This census (name → shape → bytes for every tensor, generated from the pinned config and diffed against `model.safetensors.index.json`) is a Phase −1 machine-readable artifact; CI fails if the source drifts. It is also the mechanical resolver for OQ-1 (no mHC/n-gram/depth tensor names may appear) and the input to every buffer-sizing and footprint claim below.

Derived planning numbers (each **recomputed, never inherited**):

- **Bytes streamed per decoded token** (batch 1, weights only): 2 loops × 3.149 GB×(bytes/param) + lm_head + embed row. bf16 ≈ 13.6 GB → int8 ≈ 6.8 GB → int4(decoder)+int8(lm_head) ≈ **3.7 GB**.
- **Decode FLOPs per token:** 2 × 3.149 B × 2 ≈ 12.6 GFLOP + lm_head 1.02 GFLOP (+ attention, context-dependent).
- **KV per token:** 180,224 B = 176 KiB bf16 / 88 KiB int8 (44 loop-layers × 4 KiB or 2 KiB).
- **i32 accumulator bounds** (int8 GEMM, §5.4): at K=10752, full-domain S8×S8 is at most **176,160,768**; the x86 U8×S8 offset-domain raw dot is at most **350,945,280**, and its `128·Σw` correction is at most **176,160,768**. Every materialized intermediate remains below i32 range with >4× headroom. The converter currently targets `[-127,127]`, but the kernel proof covers all stored i8 bit patterns so correctness is not secretly coupled to a quantizer convention.

### 2.7 Capabilities & benchmarks — **[REPORTED]** (model card; not independently reproduced)

- **General/agentic:** GDPval-rubrics 74.3 (vs Qwen3.5-9B 61.9, Gemma4-12B 68.5), Agent-IF-Oneday 67.5.
- **Code-agent:** SWE-Bench Verified **63.6** (vs 53.1 / 44.2), Terminal-Bench 2.0 **44.1** (vs 29.2).
- **Reasoning:** GPQA-Diamond **87.4**, HMMT-Feb-2026 **82.8**.
- **Assistant (OpenClaw framework):** Pinch-Bench-V2 74.7, Claw-Gym 65.0, GDPval 68.8, DeepResearch Bench II 33.4, ResearchRubrics 44.8 — beating Qwen3.5-4B and 9B across all six.
- All evals ran **thinking mode ON with `preserve_thinking=true`** — our task-layer quality evals (§9.6) must state their thinking mode with the same honesty.
- **Implication for §7:** this is not a "small model, small expectations" engine. At 87.4 GPQA-Diamond-class reasoning, LLM-grade NER/extraction/judging quality is a realistic bar; our evals hold it to that.

### 2.8 Prior art & references — **[OBSERVED@pin existence; contents to pin in truth pack]**

1. **`Nanbeige/llama.cpp` branch `nanbeige42`** — the authors' fork; **upstream llama.cpp does NOT support the looped architecture** (GGUF `general.architecture = nanbeige` is understood only by the fork). Community GGUFs exist. Three roles: honest CPU baseline; independent loop/KV/source cross-check; and quantization prior. Its choices seed hypotheses only—AA-Q1 and complete-artifact evaluation decide ours.
2. **vLLM (`nanbeige42` branch) and SGLang (`nbg42` branch)** forks — GPU-side; useful only as additional semantics cross-checks and for the tool-call/reasoning parser conventions (`--tool-call-parser nanbeige`).
3. **MLX/Ollama support [REPORTED]** — Apple-side references; the Ollama MLX path implies the loop is implementable in a few hundred lines of a clean framework, corroborating the "minimal op set" reading.
4. **`MIT-RLX/rlx-models` `rlx-nanbeige` crate** — a third-party Rust implementation of the family exists. **We do not depend on it** (FrankenSuite-only rule) and must not copy from it, but it is a legitimate cross-check target for differential testing of loop semantics in Phase 1 — three independent implementations (HF Python, llama.cpp fork, rlx) triangulate the truth faster than one.
5. **swiss_army_llama `sentiment_score_generation.py`** (the architect's prior project) — behavioral inspiration for adjective dimensions, contextual definitions, sampled scoring, intervals, and focus areas. Its inspected repository exposes no license file, so §5.7 requires recorded owner/provenance authority before any code, prompt, or preset data is copied. The candidate-distribution mode in §7.5 is a new estimator and must win independently against human labels; no “~1/50th” claim survives without an end-to-end measurement.

### 2.9 Required neural-op set (the complete kernel surface)

Given the pinned-source census (to be promoted by Phase −1), the **entire active** op inventory is:

1. **Token embedding lookup** (166144 × 3072, bf16 table; index_select pattern, f32 out).
2. **RMSNorm** (eps 1e-5) — *frankentorch: `rms_norm_forward_f32` exists.*
3. **RoPE** (θ = 7e7, head_dim 128, split-half rotation; reference constructs frequencies/cos/sin in f32 then casts to the q/k dtype **[OBSERVED@pin]**) — build; generate tables only to the admitted runtime context cap.
4. **GQA causal attention** — q 3072→6144, k/v 3072→1024, 48:8 heads @ 128; scale `1/√128`, additive causal mask, f32 softmax then cast **[OBSERVED@pin]**. Frankentorch's audited `sdpa_forward_f32` is dense per-head, not a ready-made GQA kernel; use it only as an L1/reference component with an explicit KV-repeat adapter, then build a GQA-aware prefill/decode path (§3.5, §6.8).
5. **SwiGLU MLP** — gate/up 3072→10752, SiLU, elementwise mul, down 10752→3072 — int8/int4 GEMM/GEMV targets (*frankentorch `linear_int8_dynamic_f32` exists as the seed*).
6. **Residual adds** (autovectorized glue).
7. **The loop schedule** — run layers 0..21, final RMSNorm, then layers 0..21 again, final RMSNorm; bind KV at `layer + loop×22` (pure control flow — ours).
8. **Final RMSNorm + lm_head** (3072→166144) — full GEMV for generation; **row-sliced GEMV for scoring** (§6.10); argmax / top-k / temperature-top_p-top_k samplers; seeded RNG.
9. **Sampler extras** — repetition controls if the reference applies any by default (**[OPEN OQ-8]**: generation_config lists none; confirm the HF `generate` defaults actually in play), logit-bias hooks for constrained decoding (§7.3).
10. **Constrained-decode mask application** — per-step allowed-token bitmask AND into logits before sampling (ours, §7.3).

**Not needed at all** (VLM/OCR machinery franken_ocr had to build): image front end, conv stacks, windowed vision attention, bicubic interpolation, masked-scatter fusion, MoE routing. The op surface is a fraction of franken_ocr's — **the ambition shifts to throughput engineering (§6) and the task layer (§7).**

---

## 3. Why pure-Rust + frankentorch + asupersync

### 3.1 The generality-tax wedge — *one fixed model, compile-time-known shapes, no framework*

A general ML framework pays a generality tax on dynamic op/dtype/graph handling. `franken_nlp` runs one model whose **weight-side and head dimensions** are fixed—hidden 3072, q 6144, KV 1024, head_dim 128, intermediate 10752, 22×2 passes, vocab 166144. Batch rows, sequence length, candidate count, and resource budgets remain dynamic and keep explicit tail paths. Fixed inner shapes let us:

- **Specialize kernels to fixed shapes** — `const`-generic tile sizes; 3072/6144/1024/10752/128 all tile cleanly for SMMLA 8×8, VNNI 16-lane, and AVX2 8-lane geometries; no remainder handling, no runtime shape branching in the hot loop.
- **Pre-pack weights offline** into arch-specific interleaved int8/int4 tiles — the converter knows the exact layout each kernel wants; the runtime never reshuffles (the llama.cpp "aarch64 repack" lesson, done at convert time instead of load time).
- **Pre-size and reuse the admitted hot working set** — activation rails, bounded candidate/logit scratch, and mask words come from engine-owned arenas; KV uses lazy pages from a budgeted pool rather than allocating the impossible configured-context maximum up front. Page acquisition happens only at admission/page boundaries; the inner token/layer loops make zero general allocator calls.
- **Schedule the loop explicitly** — per-loop KV binding, boundary norm, buffer reuse, and batch layer-major execution. Each physical layer still streams once in loop 0 and once in loop 1, but each stream serves all compatible rows instead of one sequence.
- **Skip autograd entirely** — inference only; straight to `ft-kernel-cpu`'s free-standing kernels over `&[f32]`/`&[i8]` slices.
- **Fuse only adjacent dataflow that survives an equivalence proof** — input quantization shared across gate/up, projection epilogue→RoPE, residual→next RMSNorm, GEMV→argmax — because no framework boundary forbids it, but algebra and dependencies still do (§6.11).

This is the same wedge franken_ocr proved (and franken_whisper and the frankensearch int8 reranker before it), with one addition: **the loop doubles the value of every weight-side byte saved and every weight-stream amortized**, because the stream happens twice per token (§2.4).

### 3.2 Why frankentorch (consumed at the kernel level) — and the measured lessons that bind us

At inspected commit `523aaf…`, frankentorch provides a valuable **baseline**, not this engine's finished kernel layer: `quantize_per_output_channel_i8` (symmetric rows, current output range `[-127,127]`), `linear_int8_dynamic_f32` (per-row dynamic s8 activation quantization, i32 accumulation, SDOT / AVX-512-VNNI / scalar dispatch), a prepacked SDOT entrypoint, `rms_norm_forward_f32`, `sdpa_forward_f32`, softmax/SiLU/matmul/argmax primitives, and safetensors loading through `ft-serialize` **[OBSERVED@pin]**. The dynamic linear allocates and repacks work that a fixed-model hot path cannot afford; it has no AVX2 or SMMLA tier; and the inspected SDPA is dense per-head rather than a model-native 48:8 GQA implementation. We reuse these as f32/L1 or scalar/SDOT baselines where their contracts match, then build the missing fixed-shape/prepacked/GQA paths. §3.5 records every reuse claim so “built on frankentorch” never becomes “assumed frankentorch already has it.”

**The measured lessons from the frankensearch/franken_ocr kernel campaigns are inherited as doctrine, not re-litigated:**

1. **The gap to MLAS/ONNX-class baselines is kernels-below-peak, NOT framework overhead.** A naive "fused tape-free forward" that swapped SIMD kernels for scalar ones regressed 3–10×. Every op must stay at peak; fusion is worthless otherwise.
2. **Un-blocked SMMLA is a trap** — its extra MAC density can be lost to load-bound stalls; the candidate needs register/cache blocking + offline pre-packing. The exact compute:load ratio and tile are measured per fixed shape, not inherited as magic constants.
3. **Never hand-roll wide SIMD over elementwise glue** — measured ~5× slower than LLVM autovectorization. Hand-SIMD is reserved for int8 MAC micro-kernels and (measured, gated) vectorized transcendentals (§6.12).
4. **Apple AMX is not directly programmable** — the no-FFI Mac path is NEON SDOT/SMMLA; Accelerate/BNNS is an opt-in FFI feature, never the default. **And on Apple Silicon, measure LLVM-autovec against forced SDOT/SMMLA per shape** — franken_ocr's shipped dispatch found autovec *faster* for some ordinary dense int8 shapes on M-series; we inherit the "measured-faster wins, hardware capability is not a routing decision" rule.
5. **AVX2's common `vpmaddubsw` shortcut can saturate inside each adjacent-pair operation** — no later “split-accumulate cadence” repairs already-clipped bits. The Zen 3 path therefore uses one of the exact decompositions in §6.5, selected by measurement and proved against i64.

**What frankentorch does NOT give us (the build list):** the tiled register-blocked int8 GEMM (both prefill GEMM and batched-GEMM), all int4 (packing, in-register unpack→MAC, group scales), the decode GEMV attention over a 44-deep KV, RoPE, the loop scheduler, batched layer-major execution, prefix-cache/KV management, logit-sliced lm_head, samplers, the SentencePiece tokenizer, the chat-template builder, the constrained-decode engine, and the entire task layer. That is the project.

### 3.3 Why asupersync (orchestration / cancellation / IO only)

The sibling layered pattern is adapted with an explicit process-shared lifecycle:

- **`fn main()` stays SYNCHRONOUS** — clap parse, install the asupersync-owned shutdown/signal controller, call sync `run(cli)`, map typed errors → exit codes. The runtime lives below main, inside the engine, never spanning the process.
- **The process owns one shared runtime/kernel resource set; engines hold leases to it.** `NlpEngine::builder()` snapshots configuration and obtains (or is explicitly given) an `Arc<EngineResources>`; creating ten engines does not silently create ten runtimes or rayon pools. Public methods are sync and blocking. A call made from one of the same runtime's worker threads fails fast with a typed `ReentrantCall` error instead of nesting `block_on`; async hosts call the blocking API from their own blocking boundary. The exact lifecycle is stress-tested with many create/drop/concurrent-call sequences.
- **CPU-bound stages run via `spawn_blocking`** wrapped in `asupersync::time::timeout` for per-request/per-stage budgets (env-overridable).
- **Intra-op math parallelism is the shared kernel pool's**, not asupersync tasks. Exactly one latency forward occupies the pool; concurrent library callers enter a bounded FIFO admission queue or, when compatible, the throughput batcher. They never launch independent oversubscribing forwards. Streaming uses bounded channels integrated with the same cancellation/drain state machine.
- **Cancellation:** a `Copy` cancellation token threaded into long loops (per-doc, per-decode-step `checkpoint()`), cooperative, honored at the next boundary; per-doc budgets in batch mode so one pathological document cannot stall the stream.
- **Capacity certificate:** bounded channels everywhere (backpressure, never unbounded growth); a gauntlet artifact proving no nested-runtime / no rayon-oversubscription; the `many_docs_without_deadlock` CI watchdog (batch of docs ≫ pool size) hangs on regression.

> **Hard rule (the deadlock saga's durable fix): NEVER nest rayon under a held lock; NEVER nest a second asupersync runtime inside a task.** One engine, one runtime, one kernel pool; the outer request loop is sequential (latency mode) or explicitly batch-scheduled (throughput mode); the forward fans out internally.

### 3.4 FrankenSuite dependency policy

The exact sibling commits in the research snapshot become a checked `SUITE.lock`-style manifest before Cargo scaffolding; floating `0.3.x` prose is not reproducibility. The **committed public Cargo manifest uses immutable Git revisions** for FrankenSuite dependencies so a fresh clone builds without the maintainer's sibling-directory layout; `Cargo.lock` and `SUITE.lock` must agree. Local `/dp` or `~/projects` checkouts may be selected only through an untracked developer override, and a preflight refuses the override unless its HEAD equals the reviewed pin. No absolute/local path enters a release manifest. Direct FrankenSuite dependencies are layer-minimal: `ft-kernel-cpu`, `ft-core`, `ft-serialize`; `asupersync` with only the runtime/network/TLS features actually used; and optional `fsqlite`/`fsqlite-types` for metadata/job state.

The **complete direct non-FrankenSuite release allowlist is exactly three families**, approved by the project owner on 2026-07-30:

1. `clap` — CLI parsing only;
2. `serde` + `serde_json` — typed API/NDJSON/schema/manifest data only;
3. `sha2` — SHA-256 artifact/provenance integrity only.

No `thiserror`, `anyhow`, `ctrlc`, `rayon`, `half`, `memmap2`, `uuid`, `num_cpus`, allocator crate, or other commodity crate may be added. Typed errors use `std`; shutdown/signals, bounded tasks, HTTP/TLS, CPU resources, and bf16/safetensors come through the pinned FrankenSuite surfaces; mmap/NUMA remain enumerated audited islands; stable ids derive from canonical hashes. A dependency-policy test parses `Cargo.lock`/`cargo metadata` and fails if a new **direct** non-suite root appears. Transitive dependencies already selected by a pinned FrankenSuite crate or the three approved roots are recorded in the supply-chain manifest; they are not permission to depend on them directly. Benchmark/oracle tooling stays outside the release dependency graph.

**No `tokenizers`, `tiktoken`, `minijinja`, `llguidance`, ONNX, torch, BLAS, or C tokenizer/grammar library.** Asupersync's inspected tree contains `RuntimeBuilder`, `Runtime`, `ShutdownController`, HTTP client modules, and the `tls-webpki-roots` feature; Phase 0 proves the minimal feature combination with an HTTPS range-download fixture before `fnlp pull` depends on it. The only product network path is the explicit pull command. Tokenizer bytes may be embedded only after the redistribution/attribution gate in §5.7 is green; otherwise `tokens` resolves a separately user-provided tokenizer asset and the binary embeds only its expected digest.

### 3.5 Foundation audit — adopt, adapt, or build

| Foundation surface at inspected pin | Decision | Gap/proof before use |
|---|---|---|
| `ft-kernel-cpu::linear_int8_dynamic_f32` | **ADAPT as scalar/SDOT/VNNI L1 baseline** | Allocates/dynamically quantizes; no AVX2 or SMMLA; fixed-shape production path must be prepacked and allocation-free |
| `linear_int8_dynamic_prepacked_f32` | **ADOPT narrowly for SDOT experiments** | Confirm exact packing ABI and bit identity; do not make its layout the `.fnlpq` ABI |
| `sdpa_forward_f32` | **ADAPT as dense reference** | It is not GQA-aware at this pin; an explicit repeat-view adapter is acceptable only for L1 fixtures, never as the claimed optimized kernel |
| `rms_norm_forward_f32`, softmax, SiLU, argmax | **ADOPT for f32 parity baseline** | Shape/dtype/order fixtures first; fused variants remain ours and must prove equivalence |
| `ft-serialize` safetensors loading | **ADOPT in converter** | Tensor-name/shape/dtype/byte-order census and malformed-input limits wrap it |
| asupersync runtime/shutdown/HTTP/TLS | **ADAPT for orchestration and pull** | One-process resource lifecycle, re-entrant-call fail-fast, cancellation/drain automaton, HTTPS/range/resume fixture |
| `fsqlite::Connection` | **ADOPT only behind opt-in metadata/job state** | No document/prompt/result fields in tables; explicit content spools are separate files; disabled mode opens no database |

This table is a revision-bound observation. Phase 0 compiles a tiny probe against the locked pins; a missing or changed symbol changes this table and the plan before implementation—not a compatibility shim after the fact.

---

## 4. System architecture

### 4.1 Repository shape — single crate, two binaries (the proven template)

`franken_nlp` is **one crate** (NOT a workspace), consuming only narrow FrankenSuite crate surfaces pinned by immutable Git revision (with optional same-revision local development overrides) — the franken_whisper/franken_ocr single-model shape, corrected so a fresh public clone is not coupled to the maintainer's filesystem.

```
franken_nlp/                          (crate: franken_nlp)
├── Cargo.toml                        # [[bin]] fnlp -> src/bin/fnlp.rs ; [[bin]] franken_nlp -> src/main.rs
│                                     #   (each a one-line shim calling franken_nlp::cli_main(); never one file in two targets)
│                                     # [lints.rust] unsafe_code = "deny" + #![deny(unsafe_code)] at crate roots;
│                                     #   islands opt in via scoped #![allow(unsafe_code)] + a policy test enumerating them
│                                     # immutable git+rev suite deps; optional untracked same-rev local overrides
│                                     # exact suite commits recorded in SUITE.lock; minimal asupersync features
├── rust-toolchain.toml               # nightly (stdarch i8mm/dotprod + portable_simd)
├── src/
│   ├── main.rs / bin/fnlp.rs         # thin shims -> cli_main()
│   ├── lib.rs                        # re-exports: NlpEngine, task request/result types, Error/Result
│   ├── cli.rs                        # clap-derive surface; cli_main(); to_request() validation
│   ├── orchestrator.rs               # NlpEngine leases shared EngineResources; budgets/cancellation
│   ├── robot.rs                      # versioned NDJSON events, robot schema/health/backends
│   ├── batch.rs                      # the NDJSON batch daemon: bounded queues, scheduler handoff, receipts
│   ├── jobs.rs                       # opt-in durable corpus jobs: journal, resume, verify, materialize
│   ├── storage.rs                    # metadata/job state on fsqlite; text/results only in explicit spools
│   ├── error.rs                      # FnlpError -> stable exit codes
│   ├── tokenizer/                    # pure-Rust SentencePiece (model-file parser, encode/decode, specials)
│   │   ├── mod.rs / sp_model.rs / specials.rs
│   ├── template/                     # native chat-template builder: roles, thinking, tools (xml|json)
│   ├── grammar/                      # schema/source languages -> bounded execution program (§7.3)
│   │   ├── schema.rs / compiler.rs / mask.rs / source.rs / execution.rs / jsonval.rs
│   ├── native_engine/                # THE MODEL PACKAGE (self-contained, plain Rust over slices)
│   │   ├── mod.rs                    # NanbeigeModel: cached Arc + resolve_model, header sniff
│   │   ├── weights.rs                # .fnlpq reader + manifest census + (opt-in) mmap island
│   │   ├── tensor.rs                 # Mat/QuantMat representations
│   │   ├── nn.rs                     # frankentorch facade: gemm/gemv/int8/int4/rmsnorm/silu/softmax
│   │   ├── layer.rs                  # one decoder layer: norm->attn->residual->norm->swiglu->residual
│   │   ├── attention.rs              # GQA prefill SDPA + decode GEMV attention over loop-bound KV
│   │   ├── rope.rs                   # theta 7e7 tables, q/k projection-epilogue variant
│   │   ├── looprun.rs                # 2x pass; KV=layer+loop*22; final RMSNorm after each pass
│   │   ├── kv.rs                     # 44-deep KV cache: layout, prefix-cache reuse, int8 KV (gated)
│   │   ├── batchsched.rs             # layer-major continuous batching (§6.7): admission, step planner
│   │   ├── lmhead.rs                 # full GEMV, row-sliced GEMV (§6.10), fused argmax
│   │   ├── sampler.rs                # greedy/temp/top-k/top-p, seeded RNG, mask AND, logprob capture
│   │   └── decode.rs                 # frozen DecodeParams/DecodeOutput contract + AR loop + streaming hooks
│   ├── tasks/                        # THE NLP TASK LAYER (§7): one module per task over one Task trait
│   │   ├── mod.rs                    # Task trait, TaskSpec registry, preset loading, I/O envelopes
│   │   ├── ir.rs / recipe.rs         # bounded TaskIR; public recipes only after built-in equivalence
│   │   ├── extract.rs / ner.rs / resolve.rs / sentiment.rs / classify.rs
│   │   ├── redact.rs / judge.rs / summarize.rs / keyphrases.rs / answer.rs / generate.rs
│   │   ├── presets/                  # focus-area packs (sentiment adjectives, PII types, judge rubrics) as data
│   │   └── mapreduce.rs              # long-doc chunk/map/reduce spine shared by summarize/score/extract
│   └── textutil/                     # non-LLM utilities: sentence split, chunking, unicode normalize, counts
├── tests/                            # conformance_harness, native_engine_e2e (model-gated), robot_contract,
│   └── fixtures/                     #   tokenizer/template/grammar corpora + frozen reference outputs
├── benches/                          # prefill, decode_token, batch_throughput, task gauntlets
├── scripts/                          # fetch_model.sh/.ps1, gen_reference_fixtures.py (pinned oracle), check.sh
├── docs/                             # DISCREPANCIES.md, NEGATIVE_EVIDENCE.md, PERF_LEDGER.md, truth-pack/
└── .github/workflows/                # ci.yml (check.sh), dist.yml (5-target release matrix)
```

**Why this shape:** identical rationale to franken_ocr §4.1 — the single-crate + two-thin-shim-binaries layout with a self-contained `native_engine/` is proven; `tasks/` and `grammar/` are the new top-level citizens because they are this project's product, not accessories.

**Subsystem codenames** (used by the README and progress reporting; they name regions of this tree rather than adding structure): **Foundry** = converter + `.fnlpq` + pull (§5) · **Lexicon** = `tokenizer/` + `template/` · **Ouroboros** = `native_engine/`, the loop-scheduled model core (the snake that eats its own tail runs its own layers twice) · **Conveyor** = `batchsched.rs` + the `kv.rs` prefix cache + batch/jobs (§6.7, §8.4) · **Stencil** = `grammar/` and its exact execution compiler (§7.3) · **Atelier** = `tasks/` + bounded `TaskIR` (§7) · **Assay** = conformance, qualification, and gauntlet (§9).

### 4.2 In-crate subsystems that franken_ocr did not need (build-fresh list)

- **SentencePiece tokenizer (`tokenizer/`)** — parse `tokenizer.model` (a protobuf; we write a minimal, dependency-free reader for the pieces/scores/type fields — same discipline as franken_ocr's in-crate HF-JSON BPE), implement the observed SentencePiece **BPE** merge/scoring semantics, byte-fallback behavior, special-token injection, and exact decode (including whitespace/replacement conventions). Token-id-exact vs the slow reference tokenizer is gate L0 (§9.2); OQ-6 retains edge semantics, not the model-family choice.
- **Chat-template builder (`template/`)** — a typed Rust builder (`Conversation { system, turns, tools, enable_thinking, preserve_thinking, tool_call_format }` → token ids) that reproduces `apply_chat_template` byte-exactly across the mode matrix (OQ-7 fixtures). No Jinja interpreter — the template is one fixed program; we implement *it*, not a template language.
- **Grammar/constrained-decode engine (`grammar/`)** — §7.3; JSON-Schema plus optional source languages → bounded automaton → exact per-step execution choice over the 166,144-token vocab.
- **Task layer (`tasks/`)** — §7; built-ins first compile to a bounded, non-executable `TaskIR`; the public recipe format stays closed until equivalence, resource, and no-code/no-network gates pass.
- **Batch scheduler (`batchsched.rs`)** — §6.7.
- **Durable corpus jobs (`jobs.rs`)** — §8.4; opt-in fsqlite journal + content-addressed semantic keys + checksummed spool/materialization, with no input/output persistence by default.

### 4.3 Pipeline stages (composable, budgeted, cancellable)

```
Resolve model → Load/mmap .fnlpq (census-checked) →
  Tokenize (template build | raw) → [grammar compile, cached] →
  Prefill (batched, prefix-cache aware) → Decode loop (constrained | free) →
  Detokenize / parse → Task postprocess (validate, offsets, calibrate) → Emit (json | ndjson | text)
```

Each stage has a `FNLP_STAGE_BUDGET_<STAGE>_MS` override and a cancellation checkpoint. The two cost centers are **prefill** (compute-bound; dominates scoring/classification workloads where outputs are tiny) and **decode** (memory-bound; dominates generation/summarization). They are profiled and optimized separately, per regime (§10.1) — the cost-center split is the same discipline as franken_ocr's vision-vs-decode split, with prefill playing vision's role.

### 4.4 Op → frankentorch map (exists vs must-build)

| Op | Where used | frankentorch status | Plan |
|----|-----------|---------------------|------|
| int8 dynamic linear (SDOT/VNNI/scalar, bit-identical) | all decoder GEMMs, lm_head | **EXISTS** | reuse as baseline + correctness oracle |
| Tiled register-blocked int8 GEMM (SMMLA/VNNI/AVX2, packed B) | prefill + batched decode | **BUILD** (named, not shipped) | our wedge; §6.4–§6.6 |
| int4 group-quant storage + in-register unpack → int8 MAC | decoder weights | **BUILD** | §6.4, Phase 5 |
| f32 SDPA (masked, GQA) | prefill attention | **EXISTS** | reuse for v1 parity; int8 later, gated |
| Decode GEMV attention over 44-deep KV | decode | **BUILD** | §6.8 |
| RMSNorm f32 | everywhere | **EXISTS** | reuse; fuse variants ours |
| RoPE θ=7e7 hd=128 | attention | **BUILD** | split-half/f32-table contract; OQ-4 evidence first |
| SiLU / softmax / argmax | MLP, sampler | **EXISTS** | reuse; vectorized exp gated §6.12 |
| Embedding lookup (bf16 table → f32 row) | embed | **BUILD** thin | index_select pattern |
| bf16 bulk dequant | load path | **EXISTS** through pinned ft-core/ft-serialize paths | reuse; no direct `half` dependency |
| safetensors BF16 load | converter | **EXISTS** (`ft-serialize`) | reuse in `fnlp convert` |
| Loop scheduler / KV binding / prefix cache / batch planner | engine | **BUILD** | §6.7–§6.9 |
| Samplers + mask AND + seeded RNG | decode | **BUILD** | §2.9 |
| SentencePiece tokenizer / template / grammar / tasks | product | **BUILD** | §4.2, §7 |

---

## 5. Weight transformation pipeline

The mandate: **lawfully obtained, immutable HF snapshot → reference-parity load → custom quantized form → deterministic round trip → conditional authorized distribution → verified local activation**. There are two deliberately separate download paths:

1. **Sovereign/source path:** a maintainer or user explicitly downloads the pinned 8.34 GB upstream checkpoint, verifies it, and runs `fnlp convert` locally. This path is complete even when redistribution is blocked.
2. **Release/install path:** after §5.7 clears, maintainers publish the already-converted canonical `.fnlpq` as immutable GitHub Release chunks; `fnlp pull` reassembles and activates it on the user's machine. The installer invokes that same Rust pull path—it never downloads or converts the upstream bf16 shards itself.

### 5.1 Stages

```
[1] ACQUIRE PINNED SOURCE (explicit, out-of-band, `scripts/fetch_model.sh` / `.ps1`)
    `Nanbeige/Nanbeige4.2-3B` @ f56ec5a9650268aa098496734743c25ea778bd2d:
    2 safetensors shards + index + config + tokenizer.model + tokenizer.json +
    tokenizer_config.json + added_tokens.json + special_tokens_map.json +
    generation_config.json. Download into a revision-scoped source directory,
    verify the Phase −1 conversion-source manifest, then atomically expose each file.
    This is never the end-user installer path and never runs at inference time.

[2] REFERENCE-PARITY LOAD (`fnlp convert`)
    ft-serialize::load_safetensors_from_bytes -> BF16->f32 in process
    WeightsManifest census: expected (name, shape) for every tensor from the §2.6 census
    MISSING / SHAPE-MISMATCH / EXTRA -> LOUD named diff, refuse to proceed
    (EXTRA is also the OQ-1 tripwire: any mHC/ngram/depth tensor name aborts with a design-assumption error)

[3] QUANTIZE (staged recipe, each stage its own parity gate + ledger entry)
    int8 (Phase 2): per-output-channel symmetric (quantize_per_output_channel_i8), zero-point 0
      (2a) MLP gate/up/down (the bulk: 2.18 B params) -> parity gate
      (2b) attention q/k/v/o behind FNLP_INT8_ATTN kill-switch -> parity gate
      (2c) lm_head behind FNLP_INT8_LMHEAD kill-switch -> parity gate
    int4 (Phase 5): uniform per-group baselines first; g/tier candidates and any mixed map selected
      only by AA-Q1 complete-artifact evaluation (§10.5)
    KEEP HIGH PRECISION (BF16 verbatim) BY DEFAULT: embed_tokens, all norms; lm_head/embeddings quant are
      measured, kill-switched stages, never silent defaults

[4] PRE-PACK (arch-specific, offline)
    Candidate interleaves for aarch64 SDOT/I8MM, x86 VNNI-256/VNNI-512, exact AVX2, and
    Generic row-major; exact tile ids come from the measured dispatch table, not this prose

[5] WRITE .fnlpq (§5.2) — self-describing, versioned; embed exact approved license materials when authorized

[6] IF redistribution authority is green (§5.7), PACKAGE for GitHub Releases (§5.6):
    fixed 1,957,046,720-byte parts (tail shorter), per-part + whole SHA-256, canonical
    embedded manifest, license bundle, source/conversion receipt, and reconstruction record.
    Upload exact named files to a draft model release; never wildcard or `--clobber`.

[7] REMOTE REPLAY + INSTALL:
    re-download every published part, verify the release inventory, run a clean-cache
    `fnlp pull`, derive the native packing, pass census/selftest, and reproduce a frozen
    inference fixture before the release is eligible to publish.

[8] ROUND-TRIP / DETERMINISM GATE (§5.4)
```

#### 5.1.1 Pinned conversion-source download contract

`scripts/fetch_model.sh` (Unix) and `scripts/fetch_model.ps1` (Windows) are human-run provisioning tools, patterned on FrankenOCR's Baidu fetcher. Their default destination is:

```text
Unix:    ${HOME}/.cache/franken_nlp/source/Nanbeige4.2-3B/f56ec5a9650268aa098496734743c25ea778bd2d/
Windows: %LOCALAPPDATA%\franken_nlp\source\Nanbeige4.2-3B\f56ec5a9650268aa098496734743c25ea778bd2d\
```

Each constructs every URL as `https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/<immutable-revision>/<canonical-filename>`; follows redirects; honors `HTTPS_PROXY`/`HTTP_PROXY`; retries transient failures; resumes only into a uniquely named `.partial`; verifies exact length and SHA-256; syncs; then same-directory renames. `--check-only` fully rehashes the existing closure. “File exists” or “size matches” alone is never a cache hit. A non-default revision requires an explicit flag, is labeled untrusted until a new truth-pack revision is reviewed, and cannot reuse the default recipe/catalog identity.

The Phase −1 **conversion-source manifest** pins every file needed by `fnlp convert`, not merely the two large shards: the shards and index, config, tokenizer model/JSON/config/token maps, and generation config. At the currently inspected revision, the HF LFS API reports these load-bearing identities **[OBSERVED@pin; promote to EVIDENCED in Phase −1]**:

| file | exact bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 4,973,547,960 | `09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94` |
| `model-00002-of-00002.safetensors` | 3,366,076,760 | `31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389` |
| `tokenizer.json` | 18,450,979 | `1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f` |
| `tokenizer.model` | 2,782,298 | `fb41d04798b714520a9b075727b0226538b7330254299062742c50ec8374bc36` |

A separate **truth-pack research manifest** pins the model/configuration source, card/API metadata, report, and negative LICENSE/NOTICE census. Those evidence files inform architecture and redistribution review but do not become converter or runtime prerequisites. Preflight computes free space from the conversion-source manifest plus conversion-output upper bound and safety margin; it does not hard-code “10 GB should be enough.” Invalid old files are quarantined with their observed digest, not silently overwritten. Each script's final instructions use the revision directory as one unit:

```bash
# Windows uses scripts/fetch_model.ps1 with equivalent arguments.
scripts/fetch_model.sh --dest /path/to/nanbeige-source
fnlp convert --source /path/to/nanbeige-source \
  --source-manifest docs/truth-pack/nanbeige4.2-3b.source.json \
  --recipe nanbeige42-int8-v1 --arch generic \
  -o nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
```

`fnlp convert` refuses a missing/extra/wrong-digest conversion input before parsing tensors, writes through a same-directory staging file, and emits a machine-readable conversion receipt containing the ordered source-root digest, 201-tensor census digest, converter commit, recipe, rounding/packing ids, output length, output SHA-256, and license state. It then reloads the staged `.fnlpq`, reconstructs every logical tensor, runs structural/census/round-trip and kernel selftests, syncs, and atomically renames. If pinned reference fixtures are locally available, conversion also runs the applicable L1–L4 smoke checks; release certification always runs the complete named L1–L4 and task-quality gates. “Optimal” means the best recipe that has cleared the locked parity, task-quality, footprint, and host-regime performance gates; the word is never inferred from bit width alone.

### 5.2 Custom on-disk format `.fnlpq`

Adapt the `.focrq` container pattern, with a newly specified strict envelope rather than ABI-by-analogy:

```
magic: b"FNLPQ\0\0\1"                         # eight bytes; version is also explicit below
format_version: u32 little-endian
header_len / payload_len / tensor_count: u64  # checked arithmetic; hard caps before allocation
arch_target: enum { Generic, Aarch64Sdot, Aarch64I8mm,
                    X86Vnni256, X86Vnni512, X86Avx2 }
source_files: [{name, len, sha256}]           # ordered provenance; no ambiguous concatenation hash
recipe_id / converter_commit / semantic_digest
license_bundle: bytes                         # exact upstream materials; mandatory only when §5.7 authorizes it
model_config: json                          # frozen relevant config.json fields + census hash
tokenizer_blob / template_blob              # embedded tokenizer.model + chat template source: ONE artifact
                                            #   is the whole runnable model (no sidecar-drift class of bugs)
header_json: { tensors: { name: { dtype, shape, offset, len, scales*, group_size?, tier? } },
               bit_allocation_table?, packing_manifest, provenance }
payload: <raw bytes>
```

- The binary envelope has one canonical little-endian encoding; JSON is descriptive metadata, not the authority for byte offsets. Every offset/length/alignment computation uses checked `u64`, is range-checked against the actual file before slicing, and is capped for tensor count, rank, dimension, header size, tokenizer size, and total mapped bytes. Unknown required flags/enums reject; no “best effort” parsing of a newer format.
- Quantized tensors carry inline scales (per-out-channel int8; per-group int4 + tier metadata), with exact expected scale counts derived from shape and group size. NaN/Inf/non-positive scales, overlapping ranges, duplicate names, misalignment, and non-canonical nibble padding reject.
- High-precision tensors stored **BF16 verbatim** (byte-identical round-trip; the bf16-not-f16 rule inherited — f16 narrowing is lossy and only ever a ledgered divergence).
- Reader = checked byte-range index over one owned buffer (default) or the opt-in `FNLP_MMAP=1` mapping (audited island); only the approved serde/JSON metadata layer sits above the binary envelope. Census plus semantic digest runs on load so wrong/stale weights fail loudly before any kernel sees a pointer.
- An arch-specific artifact never silently “falls back” to a different packing. A `Generic` artifact may be deterministically derived into a separate arch cache; an incompatible arch target is a hard error naming the required derivation.

### 5.3 Tensor remapping (HF dotted paths → internal layout)

| HF path | Internal | Quant plan |
|---------|----------|-----------|
| `model.embed_tokens.weight` | `embed` | bf16 verbatim (int8 embed = late, measured, kill-switched experiment) |
| `model.layers.{0..21}.input_layernorm.weight` | `layer[i].norm1` | f32/bf16, never quantized |
| `model.layers.{0..21}.self_attn.{q,k,v,o}_proj.weight` | `layer[i].attn.{q,k,v,o}` | int8 stage 2b; int4 baseline/AA-Q1 |
| `model.layers.{0..21}.post_attention_layernorm.weight` | `layer[i].norm2` | never quantized |
| `model.layers.{0..21}.mlp.{gate,up,down}_proj.weight` | `layer[i].mlp.{gate,up,down}` | int8 stage 2a; int4 baseline/AA-Q1 (no predeclared “sensitive” tier) |
| `model.norm.weight` | `final_norm` | never quantized |
| `lm_head.weight` | `lm_head` | int8 stage 2c (kill-switched); row-sliced access §6.10 |

(No biases exist — `attention_bias`/`mlp_bias` false. Exact HF names census-confirmed in Phase −1; any deviation is a loud converter error.)

### 5.4 Determinism & round-trip story

- **Bit-exact round-trip** for BF16 tensors (convert→load→re-serialize asserts byte equality).
- **Deterministic quant:** pure function of the ordered source-file bytes + versioned recipe. Rounding mode (round-to-nearest, ties-to-even), zero-row behavior, scale precision, clamp domain, int4 nibble order, group traversal, and padding bytes are specification, not implementation details. Same inputs produce the same Generic artifact on every host. If an eval/calibration set enters AA-Q1, its license, digest, order, preprocessing, and split ids join the recipe.
- **Canonical algebra:** logical activations and weights are signed i8. Per-channel int8 computes `y[o] = bias? + sx * sw[o] * Σk(qx[k]·qw[o,k])` (there are no model biases; the notation reserves the contract). VNNI consumes `u8×s8`, so x86 forms `u = qx XOR 0x80` and exactly subtracts `128·Σk qw[o,k]`, with row sums stored in the packing metadata. Per-group int4 computes and dequantizes a separate i32 accumulator for each group, then adds groups in a specified increasing-group f32 order; scales may not be pulled outside that sum. Every SIMD path preserves this logical reduction order.
- **Overflow proof (recomputed for this model and every materialized intermediate):** at K=10752, full-domain S8×S8 ≤ 176,160,768; raw U8×S8 ≤ 350,945,280; `|128·Σw|` ≤ 176,160,768; even the conservative independent raw-plus-correction bound is 527,106,048, inside i32. Int4 group accumulators have their own much smaller bound. Compile-time formula tests cover every model K; runtime kernel tests use exhaustive small-K pairs plus adversarial full-K vectors and compare every tier to an i64 oracle.
- **One canonical logical artifact, many derived packings:** `fnlp convert --arch {generic,aarch64-sdot,aarch64-i8mm,x86-vnni256,x86-vnni512,x86-avx2}`; packing ids also include tile-table version. CI verifies every packing reconstructs identical logical weights.

### 5.5 Quantization recipe rationale (what's validated vs what's ours to prove)

Unlike franken_ocr (which inherited a community-validated recipe), the only prior art here is the fork's GGUF K-quant ladder **[REPORTED]** — evidence that Q4_K_M-class works for chat, but with no published per-tensor sensitivity for *this* architecture and no task-level accuracy data. So the recipe is **ours to establish by measurement**:

- **int8-everything-quantizable first** (Phase 2, staged 2a/2b/2c) as the near-lossless correctness oracle; expected ≈ 4.7 GB with bf16 embed (§13).
- **int4 by allocation, not uniformity** (Phase 5): begin with uniform deterministic recipes as baselines, then let the offline AA-Q1 allocator (§10.5) choose per-tensor tiers under a footprint budget against **held-out task metrics**, not perplexity alone. Hypotheses informed by llama.cpp `_M` recipes—such as higher precision for `down_proj` or early/late layers—remain hypotheses. The loop reuses each quantized tensor in both passes, so sensitivity interactions are joint: one-at-a-time tensor curves may seed search, but only a complete candidate artifact evaluated end-to-end may be promoted.
- Embeddings/lm_head int8 and KV int8 are separate, kill-switched, measured stages (§6.9).

### 5.6 GitHub-asset distribution, `fnlp pull`, and installer handoff

This section deliberately copies FrankenOCR's **proven invariants**, while correcting its historical rough edges: versioned immutable filenames, a release-bound embedded manifest, exact recipe compatibility before any multi-GB request, streamed part/whole hashing, cache discovery that recognizes the installed basename, per-artifact locking, same-directory staging/rename, and clean-cache inference certification.

#### 5.6.1 Canonical artifact and independent model version

- **Only after §5.7 is green**, the canonical arch-neutral Generic `.fnlpq` may be published. Binary SemVer and model-artifact SemVer are independent: e.g. several `fnlp` `v0.x.y` binaries may consume immutable model release `models-nanbeige42-fnlpq-v1`. A patch binary release must not rename or republish identical 4+ GB bytes.
- The logical filename is artifact-version/quant/packing-bearing: `nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq`; the exact recipe id is authoritative in the manifest and container header. A different source revision, format, recipe, tokenizer/template closure, or logical bytes gets a new artifact version/name/tag; bytes under an existing tag/name are never replaced.
- Public hosting stays 1× per quant tier. The released artifact is Generic; `fnlp pull` derives the host's measured-default packing locally (M4/M5 SDOT/I8MM candidate, Zen 3 AVX2, Zen 4/5 VNNI-256/512 candidate) and caches it by `(whole_artifact_sha256, packing_id, tile_table_version)`. Capability alone never decides the winner. Generic remains the reconstruction/provenance root; the derived cache is disposable and byte-differential-tested against its logical tensors.

#### 5.6.2 Deterministic chunking and release inventory

- The release packager splits the exact `.fnlpq` byte stream into **1,957,046,720-byte** chunks, with only the final chunk shorter—the exact safely-under-2-GiB size proven in FrankenOCR. Names append zero-padded ordered suffixes `.part00`, `.part01`, …; the manifest caps the count (64 in v1) so two digits are sufficient and unambiguous. Concatenation in manifest order is the original file; there is no archive or compression layer.
- A canonical schema-versioned manifest records: model/artifact ids; immutable release tag; logical filename/length/SHA-256; exact ordered part name/length/SHA-256/HTTPS mirror URLs; ordered source-file digests and source-root digest; recipe/converter/format compatibility; tokenizer/template/census/semantic digests; license-bundle digest; packing policy; revocation/supersession state. The canonical manifest bytes and SHA-256 are embedded in every compatible binary and also attached to the model release for audit. Default pull never fetches manifest identity from `main`, `latest`, or another mutable branch.
- The staging directory also contains `MODEL_ASSET_RECEIPT.json`, `SHA256SUMS`, `RECONSTRUCTION.txt`, and the exact approved license/NOTICE/modification bundle. The receipt binds the conversion receipt, split command/tool version, ordered reconstruction, whole result, parity/quality receipts, and intended release inventory. These small records are committed or attached; the multi-GB `.fnlpq` and parts remain gitignored.
- Publication uses a **draft** GitHub Release and exact explicit upload paths—never a wildcard, never `gh release upload --clobber`. The runbook first hashes the retained staging set, reconstructs it, and compares the whole bytes with the converter output. After upload, it queries the remote inventory, downloads each asset through its public URL into a clean directory, repeats part/whole verification, runs a clean-cache `fnlp pull`, derives the native packing, runs `robot selftest`, and reproduces a frozen real-model inference fixture. Only that receipt plus LG-1 authorizes publishing the draft. A bad release is superseded/revoked by a new catalog; immutable old bytes are not rewritten.

#### 5.6.3 One Rust artifact manager

`fnlp pull [--quant int8|int4] [--arch auto|...] [--model-dir ...]` and the installer share exactly one implementation: the installed `fnlp` binary's Rust artifact manager over asupersync HTTP/TLS. `install.sh`/`install.ps1` never parse the model manifest, concatenate chunks, or maintain their own hash/cache rules. This prevents the binary and installer from disagreeing about recipe, filename, cache layout, or integrity.

- Default manifest = release-bound embedded bytes. `--manifest <local-path-or-https-url>` is the sovereign/private-mirror escape hatch and **must** be paired with `--expected-sha256`; `FNLP_MANIFEST_URL` is not a hidden library read. HTTPS authenticates transport; the expected digest authenticates the manifest; its part/whole digests authenticate content. Redirects are host-allowlisted and credentials are never forwarded cross-origin.
- Before any artifact request, the parser enforces exact schema/recipe/format compatibility, portable single-component filenames, unique case-folded names, canonical part ids/order, HTTPS URLs, checked part-size sum, count/size/string caps, and required license/census digests. An incompatible historical recipe fails before cache creation or network I/O.
- The manager streams response frames directly into one same-filesystem staging file while updating per-part and whole SHA-256; a part is never buffered in RAM. Each mirror attempt rolls back to the last verified part boundary. A small resume journal binds the manifest digest, final path, committed length, and verified part index. On restart, the manager rehashes every committed part before trusting the prefix. Intra-part Range resume is used only after an asupersync fixture proves `206` plus exact `Content-Range`; otherwise the unverified tail is truncated and that part restarts.
- A per-content-address install lock closes cache-check/install races. Existing files count as cached only after exact length + whole SHA-256. After all parts verify, the manager syncs and fully parses/census-checks the staged `.fnlpq`, verifies its embedded source/license/semantic identity, derives and differentially checks the selected native packing, runs required selftests, then atomically installs and switches the small active-artifact record. Failure, cancellation, disk-full, crash, or a malicious mirror leaves the previously active artifact untouched. Concurrent pulls converge on the same content address.

#### 5.6.4 Local placement and discovery

The installed canonical file lives at:

```text
Unix:    ${HOME}/.cache/franken_nlp/models/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
Windows: %LOCALAPPDATA%\franken_nlp\models\nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq
         (fallback: %USERPROFILE%\.cache\franken_nlp\models\...)
```

`--model-dir` wins; otherwise an explicitly configured `FNLP_MODEL_DIR` wins; otherwise the platform default above. Derived packings, the embedded-manifest audit copy, license bundle, and activation receipt sit under content-addressed subdirectories of the same model root. Successfully installed remote part files are not retained: the verified staging stream becomes the one canonical local `.fnlpq`, avoiding a second 4+ GB copy. Orphan/live staging is distinguishable by its lock and `fnlp doctor` can quarantine stale state. `fnlp models` reports installed/active digests, recipe, format, source revision, packing, license path, bytes, and whether the artifact is public-catalog, private-manifest, or locally converted. Default `NlpEngine` discovery must include the exact versioned basename that `fnlp pull` installs; a clean install cannot require `--model`.

#### 5.6.5 Installer behavior

Phase 6 `install.sh` and `install.ps1` first install and SHA-256-verify the target binary, execute its exact version check, and optionally run `fnlp robot selftest`. Then:

- when a public authorized catalog is embedded, an interactive TTY gets a clear `y/N` offer stating model version, quant, download bytes, peak required free bytes, and destination; acceptance runs the installed binary by absolute path: `"<install-dir>/fnlp" pull`;
- `--with-model` / `-WithModel` explicitly opts non-interactive automation into the same pull; `--no-pull` / `-NoPull` suppresses it; quiet/non-interactive mode never silently starts a multi-GB transfer;
- a pull failure does not roll back a successfully verified binary and does not destroy an older working model; the final summary prints the exact retry command, cache path, artifact state, and uninstall/cache-removal instructions;
- before LG-1 clears, installers publish binaries only, do not offer a nonexistent public model, and instead print the pinned-source `fetch_model.sh` / `.ps1` + `fnlp convert` route and the private-manifest form. The default `fnlp pull` returns typed `PUBLIC_ARTIFACT_UNAVAILABLE` mapped to the existing exit `3` (“model not found/unavailable”) rather than pretending that card metadata authorized assets.

#### 5.6.6 Required tests and receipts

Tiny deterministic multi-part fixtures exercise manifest caps, mirror fallback, streamed assembly, part/whole mismatch, overlong/short bodies, malicious names/redirects, resume at every boundary, cache-hit rehash, symlink/device targets, concurrent pulls, cancellation/disk-full/crash, old-active preservation, native-pack derivation, and `pull`-basename discovery. Installer E2E drives the real Unix and PowerShell scripts against a fake release into a fresh HOME/LOCALAPPDATA, including interactive accept/decline, `--with-model`, `--no-pull`, quiet mode, reinstall, and failure injection. The 8.34 GB source and real release assets remain model-gated; release certification must additionally prove:

1. pinned HF closure download/check-only;
2. deterministic local conversion twice on independent clean directories;
3. package/reassemble byte identity;
4. remote asset inventory and re-download identity;
5. fresh-machine installer → `fnlp pull` → no-flag model discovery → frozen real inference;
6. second pull returns an exact cache hit and inference opens no network.

`fnlp convert` remains the sovereign path (G3: public pull is convenience, never a requirement). `fnlp robot health` prints recipe, catalog/manifest/artifact/source/packing/license digests and source class. **Until §5.7 is green, all public-release language above is a technical design and test target, not permission to publish weights. Local conversion and user-supplied/private pinned manifests are the complete distribution story.**

### 5.7 Redistribution-authority gate (fail closed)

The source repository and transformed weights are separate licensing surfaces. Before the first public model asset, commit `docs/truth-pack/LICENSE_PROVENANCE.md` with:

1. the immutable upstream revision and a complete file/API census showing the license declaration;
2. the exact license text, copyright attribution, upstream NOTICE (including an explicit “none supplied” statement if confirmed by the rights holder), and our modification notice;
3. a short written determination by the project owner that the evidence authorizes redistribution of **weights and transformed derivatives**, not merely code;
4. a clean-room provenance statement for presets derived from `swiss_army_llama` (the repository currently exposes no license file at the inspected pin; copy no code/data until ownership or permission is recorded);
5. a release test that extracts the bundle from `.fnlpq` and byte-compares it with the approved truth-pack files.

Current state: the HF card declares Apache-2.0, but the pinned repository has no license or NOTICE file **[BLOCKED]**. Therefore Phase 2 may build and test local artifacts, but its public-asset exit gate cannot pass yet. This is a legal/provenance gate, not a technical inference.

---

## 6. Model-specific CPU kernel strategy

### 6.1 The hot ops (profile-anchored hypothesis — to be REPLACED by measured profiles before any kernel lands)

| Op | Regime | Bound | Priority |
|----|--------|-------|----------|
| MLP GEMV ×3 ×22 ×2 loops (2.18 B unique weights; **4.36 B weight-MACs/token**) | decode | memory | **HIGHEST** |
| Attention q/k/v/o GEMV ×22 ×2 (0.97 B unique; **1.94 B weight-MACs/token**) | decode | memory | **HIGH** |
| lm_head GEMV (510 M, or **sliced ~kB-scale** §6.10) | decode / scoring | memory | **HIGH** (or ~free when sliced) |
| Decode attention over 44-deep KV | decode, grows with ctx | memory | **MED-HIGH** |
| Prefill GEMMs (same weights, token-parallel) | prefill / scoring | compute | **HIGH** (scoring throughput = this) |
| Prefill SDPA (48:8 heads, causal) | prefill, O(T²) | compute | **MED** (short-doc regime), **HIGH** ≥ 8K |
| RMSNorm / RoPE / SiLU / softmax / residual | both | mem/compute | **LOW** (autovectorize; fuse) |
| Embedding row gather | both | memory | **LOW** |

Cost model per §2.6: a decoded token streams ≈ 3.7 GB (int4+int8 recipe) — decode is overwhelmingly weight-bandwidth-bound at batch 1, which is exactly why §6.7's batching is a structural, not incremental, lever.

### 6.2 The optimization doctrine (inherited, binding)

1. **Parity first** — no kernel lands without its bit-exact (integer) or tolerance-proven (float) gate.
2. **Native int8 MAC intrinsics for GEMM/GEMV; LLVM autovectorization for glue** (the 5×-slower hand-SIMD lesson).
3. **Register/cache blocking + offline pre-packing are mandatory** for any matrix-engine path (the un-blocked-SMMLA trap).
4. **Measured-faster wins, capability is not routing** — Apple autovec-vs-SDOT/SMMLA decided per shape by benchmark, recorded in the dispatch table.
5. **One lever at a time** with the 5-pass keep/revert loop and the NEGATIVE_EVIDENCE ledger (§10.2).

### 6.3 Per-arch SIMD dispatch catalog

One `int8_gemm` / `int8_gemv` / `int4_gemv` entry set, capability-detected once but **selected per fixed shape and regime from a measured dispatch table**. All tiers implement the same canonical integer algebra; AVX2 is not allowed a different answer. `fnlp robot backends` reports detected features, selected kernel id/tile, benchmark-table provenance, and the reason a wider tier lost:

| Tier | ISA gate | GEMM (prefill/batch) | GEMV (decode) | Notes |
|------|----------|----------------------|---------------|-------|
| **A1** | aarch64 `i8mm` actually exposed by the OS | SMMLA `vmmlaq_s32`, register/cache-tiled, packed K-panels | SMMLA or SDOT, whichever wins the shape | Candidate for any M4/M5/Arm host that advertises FEAT_I8MM; never infer it from marketing name |
| **A2** | aarch64 `dotprod` | blocked SDOT | SDOT `vdotq_s32` | Primary explicit-SIMD Apple candidate; runtime detect |
| **A3** | aarch64 autovec | LLVM-vectorized i32 MAC | same | *shipped-faster on M-series for some shapes in franken_ocr — first-class citizen, not fallback* |
| **X1a** | OS exposes `avx512f` + `avx512vnni`; 512-bit path wins sustained bench | `VPDPBUSD` zmm tiles + exact offset correction | same | Zen 4/5 Threadripper/EPYC and supporting Xeon; family-specific tiles |
| **X1b** | X1a features + `avx512vl`; 256-bit EVEX-VNNI path wins | VNNI ymm tiles + exact offset correction | same | Keeps VNNI semantics/register file while avoiding a losing 512-bit width on a given shape/host |
| **X2** | `avxvnni` | `_mm256_dpbusd_epi32` + exact offset correction | same | Supporting Intel/AMD clients; feature-detect, do not vendor-name route |
| **X3a** | `avx2` | exact low-7/high-bit `vpmaddubsw` decomposition (§6.5) | same | Primary Zen 3 candidate |
| **X3b** | `avx2` | exact sign-extend-to-i16 + `vpmaddwd` (§6.5) | same | Independent exact candidate; measurement chooses per shape |
| **S** | always | scalar i32 MAC | scalar | cross-compile floor, bit-exact oracle |

int4 storage unpacks to int8 in-register (no CPU int4 MAC exists) and feeds the same MAC paths — the win is bandwidth/footprint, exactly the property the loop doubles (§2.4).

**AVX-512 is a full optimization campaign, not one checkbox.**

- **Zen 4, Zen 5, and Intel are separate benchmark keys.** Zen 4 commonly executes 512-bit work through 256-bit datapaths; Zen 5 Threadripper 9000 advertises a full 512-bit datapath; Intel frequency/power behavior varies by generation. None of that alone chooses X1a. For every hot shape, compare zmm-VNNI, ymm-VNNI where available, AVX2, and scalar after warmup over a sustained forward—not a 5 µs microbench. Record clocks (where observable), joules/doc where available, p50/p95/p99, and thermal steady state. A narrower tier may be the default on a wider-capable CPU.
- **Width and semantics are orthogonal.** AVX-512-VNNI supplies the exact U8×S8→i32 primitive plus 32 registers and masks. AVX-512BW without VNNI is only an optional widened version of an exact AVX2-style decomposition; build it if an actual supported host and profile justify the verification surface. AVX512-BF16 is an optional f32-reference-prefill candidate, not part of the int8 correctness story.
- **Tails still exist.** K/N are favorable fixed dimensions, but batch/prefill row count M, sequence length T, vocabulary candidate count, and grammar work are dynamic. Masked loads/stores may remove scalar cleanup for M/N tails, while K padding must be converter-defined zero padding with digest coverage. Every masked-tail case has tests at 0,1,W−1,W,W+1 and page boundaries.
- **OS/firmware exposure is authoritative.** AVX-512 can be unavailable despite CPU family support (firmware/VM policy); dispatch uses runtime feature detection and executes a guarded smoke instruction in `robot selftest`, never a model-name table. Forced-tier requests that are unsupported fail clearly rather than faulting.
- **AMX is out of v1.** It could be valuable for Sapphire Rapids+ prefill, but adds tile-state/OS enablement and a new packing/proof surface outside the primary Apple/AMD target. Reconsider only from a measured X1 bottleneck.

**Local tuning cannot turn a transient benchmark into a permanent machine fact.** Phase-6 `fnlp tune` starts with fixed-shape, bit-identical kernel choices and conservative thread caps only. Each promoted row is keyed by binary/kernel ABI, artifact recipe, CPU/ISA, OS/firmware exposure, topology/affinity, and a declared workload validity domain; batch size, mmap-vs-owned loading, NUMA placement, and page-cache-sensitive choices are not host-wide constants. Promotion requires repeated randomized A/B trials after warmup, thermal/load preflight, a confidence interval wholly beyond a practical minimum effect, and hysteresis. Noise, unsupported counters, a stale key, or disagreement between microkernel and representative end-to-end results retains the shipped default. `robot backends` explains the selected row and why any local profile was ignored.

**One canonical quantization algebra (cross-ISA bit identity is a theorem first, then a test).** Weights and logical activations are s8, zero-point 0. SDOT/SMMLA consume that representation directly. U8×S8 paths reinterpret `u = qx XOR 0x80`, accumulate `Σu·w`, then subtract the offline-checked `128·Σw`; this yields exactly `Σqx·w`. Row sums are packing metadata covered by the semantic digest and recomputed on load/selftest samples. Integer identity ends before dequantization; float scale application and int4 group summation have a separately fixed operation order.

### 6.4 Weight packing & the tiled GEMM (the build)

- Offline per-arch interleave (§5.1[4]); per-layer contiguity so `layer[i]`'s full weight set streams as one sequential range per loop pass (prefetch-friendly; NUMA-placeable §6.13).
- Tile geometry is specialized around K ∈ {3072, 6144, 10752} and fixed N values. Converter-defined zero padding is permitted only where a chosen K/N tile needs it; dynamic M/T/candidate tails remain real and are handled by proved masks or cleanup kernels.
- Register/cache blocking is selected from a bounded offline candidate set per `(ISA, shape, regime)`. Initial panel sizes come from L1/L2 capacity arithmetic, but only measured winners enter the dispatch table; there is no inherited “256 KiB is right everywhere” constant.
- The same micro-kernel serves prefill GEMM and **batched decode GEMM** (§6.7) — decode at batch M is a skinny GEMM (M×K·K×N), which is precisely where the tiled kernel beats M independent GEMVs.

### 6.5 AVX2 first-class proof (because the reference AMD box is Zen 3)

The `vpmaddubsw` hazard is **intra-instruction**: it saturates an adjacent-pair sum at i16, so no downstream accumulation cadence can restore lost bits. Raw U8×S8 pairs can reach 65,280 over the full i8 domain. Two exact AVX2 candidates are required:

1. **X3a low-7/high-bit decomposition.** For offset activation `u∈[0,255]`, form `lo=u&0x7f` and `hi=u>>7`. Compute `dot(lo,w)` with `vpmaddubsw`: each pair is bounded by `2·127·128=32,512`, so it cannot saturate. Compute `dot(hi,w)` the same way (pair bound 256), widen to i32, shift by 7, add, then apply `−128·Σw`. This reconstructs the canonical signed dot exactly while retaining byte-dot throughput.
2. **X3b widened signed route.** Sign-extend activation and weight bytes to i16 and use `vpmaddwd` into i32. It uses more unpack/half-lane work but has a short proof and is an independent oracle-quality SIMD path.

Measurement chooses X3a or X3b per shape; both must equal scalar/i64 on exhaustive small vectors, randomized property tests, all-extreme full K=10752 vectors, alternating signs, offset-correction extremes, and every tail length. Raw saturating `vpmaddubsw` is not a shippable “fast approximate” variant because the failure is data-dependent and silent. Zen 3 gains primarily from many-core batch GEMM, cache/NUMA placement, and exact prepacking—not from pretending it has VNNI.

### 6.6 The loop scheduler (`looprun.rs`)

- Straight-line plan: `for loop_idx in 0..2 { for layer in 0..22 { layer_forward(layer, kv[layer + loop_idx*22]) } hidden = final_rmsnorm(hidden); }` — no embedding re-injection or boundary projection. This observed rule is implemented once, truth-pack line-backed, and property-tested (including a named loop-boundary fixture).
- Per-`(loop, layer)` KV binding resolved at engine build; no hash-map lookups in the hot loop.
- **Loop-aware prefetch candidate:** the schedule knows loop-2 layer-0 follows loop-1 final norm, so a bounded prefetch distance is testable. Hardware prefetch/cache pollution may make explicit prefetch slower; it enters only after a counter-backed benchmark and may be rejected per CPU family.

### 6.7 Batched layer-major execution (`batchsched.rs`) — the structural throughput lever

The NLP workloads (§7) are corpus-shaped: thousands of short documents × one shared task prompt. Sequential batch-1 decode streams 3.7 GB per token per document — absurd waste when 64 documents could share each weight stream. The design:

- **Admission:** the batch daemon (§8.4) admits sequences only when the worst-case next-step KV/pages, activation rows, logits/candidate rows, grammar state, output buffering, and one safety margin fit the configured engine memory budget. `M` is the minimum of that certificate and a measured throughput cap—not a naked 16–128 guess. Per-sequence state is explicit and cancellation releases its pages deterministically.
- **Step plan:** `for loop in 0..2 { for layer in 0..22 { GEMM over compatible rows } final_norm(rows) }`. Each physical layer streams **once per loop for all M rows**—two shared streams total, versus 2M independent streams. Idealized weight bytes per sequence-token approach `3.7 GB/M`, but attention, lm_head, packing, and incomplete rows prevent perfect amortization; the measured crossover is a deliverable, not “M=32 fixes it.”
- **Mixed prefill/decode steps:** group compatible rows by `(model recipe, numerics profile, loop, layer, phase, attention shape)`; chunk long prefills into bounded morsels so one 200K-token document cannot monopolize a step. Decode latency has an explicit maximum-wait budget and can preempt further prefill admission between morsels. “Continuous batching” never means combining rows whose attention semantics differ.
- **Prefix cache (`kv.rs`):** prefill immutable task prefixes once and fork page references copy-on-write. The key is the exact prefix token ids plus artifact/recipe digest, tokenizer/template/prompt hashes, thinking/tool mode, RoPE/numerics/KV dtype, and kernel-semantic version. Default cache eligibility is restricted to shipped task/system prefixes—never arbitrary user document text—so cross-request reuse cannot leak private content. User-prefix caching is explicit opt-in and namespace-isolated. Correctness gate: forked prefix equals cold prefill. A byte-budgeted eviction policy starts as deterministic LRU; S3-FIFO becomes a candidate only after traces show scan pollution (§10.5).
- **Determinism:** batch composition must not change any sequence's tokens (row order in a GEMM changes no arithmetic per row; reduction orders are per-row). A determinism gate asserts batch-M output == batch-1 output token-exactly per sequence; any deviation is a bug, not a tolerance. The mechanism, stated precisely: the integer paths (int8/int4 GEMM/GEMV, i32 accumulation) are exact regardless of blocking, hence batch-invariant by construction; the f32 paths (norms, softmax, attention accumulation, dequant-scale application) must keep **per-row reduction order identical between the M=1 and M>1 kernel variants** — a kernel-selection rule the gate enforces, not an accident we hope for.
- **Sequential fallback:** `--batch 1` (and the library default) preserves the simple latency path; the daemon is the opt-in throughput surface.

### 6.8 Attention kernels

- **Prefill:** an explicit 48:8 GQA adapter around frankentorch's dense f32 SDPA is acceptable as an L1/reference implementation; it must not materialize 6× KV in the production claim path. Build the native GQA kernel next. Post-parity candidates: online-softmax tiling after profile crossover (not a guessed fixed 8K), then int8 QKᵀ/scores·V behind a numerics kill switch.
- **Decode:** per step, per (loop, layer): q GEMV → RoPE → scores vs KV (8 KV heads × 128; block six query heads per KV head) → online softmax → weighted V sum → o GEMV. Logical addressing is `[loop-layer][head][position][128]`; the physical paged order below is selected by benchmark while preserving sequential position scans and 64-byte alignment.
- **KV pages:** fixed token pages (initial completed-page candidate 16 tokens; measured) hold all 44 slots for a sequence range, reference-counted for prefix forks and allocated lazily from a byte-capped pool. One bf16 16-token page is about 2.75 MiB, so a fork may not blindly clone a partially filled full-size page. The design evaluates immutable sealed pages plus smaller fork-tail slabs (down to one token), versus explicit tail copy/recompute; admission prices the actual parent-fill state and fan-out before branching. Page tables, not a giant contiguous reservation, make cancellation and COW bounded. Layout alternatives `[page][loop-layer][head][token][dim]` vs head-major are benchmarked separately for prefill writes, decode scans, and fork-tail amplification.
- **KV int8 (gated):** per-head-per-token scales halve the 176 KiB/token bf16 census; enable only after long-context parity/task budgets and the additional scale bandwidth are measured.

### 6.9 Memory, allocator, layout

- Owned weight buffer by default; `FNLP_MMAP=1` opt-in island (trusted immutable artifacts; the franken_ocr posture). `madvise(WILLNEED)` on the active ranges; huge pages where available (multi-GB streamed blob → TLB relief), measured not assumed. (These are platform-conditional, feature-gated levers — Linux-first — each with a portable no-op fallback; none of them touches the G3 "no foreign ML runtime" claim, which §1.1 defines precisely.)
- Activation rails, per-step scratch, mask words, and scheduler metadata are reserved to the admitted batch envelope. KV pages are allocated only at admission/page boundaries from a pre-reserved pool; the inner token/layer loops perform no general allocator calls (allocation-count test). **Default per-sequence context cap is 8192, but memory is governed by an independent `--memory-budget`**. One bf16 sequence at 8192 needs about 1.38 GiB of KV; 64 such sequences would need about 88 GiB before activations, so the engine must queue/reject before promising that combination. Up to 262144 positions is an observed model limit, not a practicality promise; `robot plan --ctx --batch --quant` prints exact committed/peak bytes and the admission result without allocating.
- 64-byte alignment for all packed weights/activations; scales in SoA.
- Allocator: system allocator only in the release design. An alternative allocator requires a separately approved FrankenSuite surface and plan revision; benchmark fairness still records allocator identity.

### 6.10 lm_head strategies (`lmhead.rs`)

- **Generation:** full 3072→166144 GEMV (int8, gated) fused with argmax for greedy (no materialized softmax over 166K). Top-k/top-p **still evaluate the full vocabulary** — that cost is irreducible for exact sampling; the saving is skipping any materialized full softmax/sort via streaming threshold/partial selection in one pass.
- **Finite-candidate scoring:** when a task needs a known candidate set, compute only those lm_head rows: C-row sliced GEMV instead of all 166144 rows. Single-token candidates need one projection. Multi-token candidates use teacher-forced continuation scoring under an explicitly named sum/mean/terminal-token rule.
- **Exact continuation trie:** compile multi-token candidates into a token trie and evaluate each unique prefix state once; terminal/EOS is an explicit scored edge when one label prefixes another. This is dynamic programming over the *complete* candidate language, not beam search: no candidate may be pruned. Before allocation, the compiler estimates unique-node work **and actual KV fork-tail bytes** from trie breadth/depth, page fill, and the 44-slot geometry, then chooses among exact breadth/frontier batching, depth-first snapshot/restore with bounded tail state, or naïve per-candidate scoring. A compact canonical short-ID encoding is a separate quality/performance baseline, not assumed equivalent to semantic labels. The trie ships only when scores/rankings equal the naïve implementation under the same full-vocabulary or trie-conditional normalization, candidate order is irrelevant, admission is certified, and the measured reuse win exceeds traversal/KV cost. Little prefix sharing or excessive fork memory selects the naïve fallback.
- **Stencil sparse projection:** for constrained greedy or grammar-conditioned sampling, `ProjectLegal(rows)` evaluates **every** legal row and no illegal row when the compiled legal set is below a measured threshold. That is exact for the conditioned distribution because the omitted rows are impossible under the declared grammar. Dense states use `FullProjection(mask)`. No heuristic may omit a legal row.
- **Forced runs:** when the grammar/tokenizer product has exactly one legal **token id** and per-token logprob telemetry was not requested, token selection needs no lm_head. A uniquely determined byte suffix is **not** sufficient: multiple tokenizations of the same bytes lead to different hidden/KV states, so byte-level jump-forward, canonical retokenization, and token healing are excluded from this exact path. A maximal bounded run of uniquely forced token ids can be teacher-fed through a causal micro-prefill to update all 44 KV slots. The transformer work is not skipped. The micro-prefill path must match sequential feeding at every token/KV point under the designated numerics contract; otherwise the fallback is one-token feeding. If logprobs were requested, the required projection is still performed. OQ-19 measures whether exact-token runs occur often enough to justify retaining the optimization; rarity is an acceptable negative result.
- **Normalization authority:** candidate-conditional/trie-conditional normalization is cheap. A true full-vocabulary denominator—or legal/illegal mass or margin—requires the full lm_head at **that hidden state**. It can be batched across rows but cannot be cached across different hidden states or described as free. Every result and telemetry field names its normalization and reports unavailable quantities as `not_computed`.

### 6.11 Fusion catalog (bit-exact; each gated by the isomorphism proof)

| Fused unit | Replaces | Win |
|-----------|----------|-----|
| Gate/up projections share one quantized input; pairwise SiLU×up writes the canonical intermediate scratch | duplicate input quantization + separate activation/multiply buffers | one input quantization and one intermediate, while preserving the global down-proj activation-scale contract |
| RMSNorm output → one shared activation quantization feeding q/k/v (and separately gate/up at norm2) | repeated quantization per projection | norm scratch/max scan once, one quantized row reused |
| q/k GEMV epilogue → RoPE (rotate in-register before the KV write; RoPE applies to *projected* q/k, so this is the projection's epilogue, not a norm fusion) | extra pass over q/k | rotated q/k never round-trip through memory |
| residual add while accumulating next RMSNorm sum-of-squares, followed by the required normalization pass | standalone residual write + later norm read/reduction | removes one read; does not pretend RMSNorm is single-pass |
| lm_head GEMV → argmax (greedy) | full softmax + scan | argmax-only over 166,144 |
| grammar mask AND → sampler | separate mask pass | mask applied during logit scan |

### 6.12 Vectorized transcendentals — the measured exception

LLVM generally cannot turn scalar libm calls into the exact vector transcendental sequence we want, so a range-reduced minimax `exp`/sigmoid is a candidate for softmax/SiLU. It is **not known-fast here until profiled**, changes numerics, and cannot be described as SIMD≡scalar bit identity. It starts default-off, carries max-ULP/domain/monotonicity/NaN tests plus L1–L5 impact, and becomes default only per architecture/shape if the ledger wins. The reference libm path always remains available.

### 6.13 Many-core & NUMA scaling (Threadripper/EPYC + Apple P/E)

- **Two parallelism axes, never blindly mixed:** latency (one sequence: parallelize within the forward — row-blocks of each GEMM/GEMV across cores) vs throughput (the §6.7 batch: parallelism lives inside the batched GEMMs; NEVER N concurrent independent forwards oversubscribing the pool).
- **Decode at batch 1 stops scaling at bandwidth/coordination saturation:** a reproducible 1…physical-core sweep directly selects the smallest thread count within the confidence interval of peak throughput/latency. A USL fit may summarize the curve and flag contention, but it never extrapolates a dispatch cap beyond measured points. No guessed “8–16” default enters code.
- **NUMA (multi-CCD/multi-socket TR/EPYC):** compare three explicit policies: bind workers+KV to one node and accept fewer cores; interleave shared weights; replicate read-only packed weights per node when the memory budget permits. “Local” without worker restriction causes remote reads, and “replicate” can double multi-GB footprint; admission includes the chosen placement. Per-node batch shards avoid cross-node writable state and merge only ordered results.
- **Apple P/E:** macOS offers **no hard thread-affinity API**, so "pin to P-cores" is not a portable guarantee — the honest lever is **QoS hints** (kernel pool at user-interactive QoS, orchestration at utility), treated as a measured optional experiment with a no-affinity fallback, with `robot backends` reporting the *observed effective* placement. Per-shape autovec-vs-SDOT/SMMLA dispatch per §6.2(4); the big-SLC effect on the double weight stream (loop!) is measured, not assumed.

### 6.14 Build-time optimization (measured release recipes, not “free”)

Create a small matrix—thin vs fat LTO, codegen-units 1 vs a practical parallel value, PGO off/on—and measure clean build time, binary size, R1–R4 p50/p95/p99, and instruction-cache counters. Promote one reproducible profile per target only if its end-to-end gain clears the keep threshold; PGO training corpus ids/order and compiler commit are hashed. BOLT is Linux-only research after PGO and requires a separately replayed profile. `panic=abort` applies to shipped binaries only when the error/cancellation contract proves no unwinding dependency; an embedding application's final Cargo profile controls its library linkage. Portable target-feature baselines plus runtime dispatch are mandatory; never ship `target-cpu=native`.

---

## 7. The NLP task layer (the product)

> Everything before this section makes one model run fast. This section is why anyone installs the binary. Design mandate: **every task is (a) valid-by-construction (G8), (b) deterministic by default, (c) batch-first, (d) honest about quality** (task evals with named datasets, §9.6). The portfolio was generated and winnowed via the idea-wizard method: ~30 candidates scored on usefulness × exploitability-of-our-engine × honesty (can a 3B LLM actually do this well?) × implementation cost; the survivors below. POS/dependency/lemma pipelines were cut as non-goals (§1.2) — the honest answer there is "use a finite-state tagger."

### 7.0 The `Task` architecture

```rust
trait Task {
    type Request;  // serde types; CLI + library + NDJSON share them
    type Response; // includes provenance: model recipe id, prompt hash, mode, token counts, timings
    fn spec(&self) -> TaskSpec;          // name, version, JSON Schemas (request/response), presets
    fn plan(&self, req: &Request) -> TaskPlan;   // prompt build + decode strategy + grammar + budget
    fn parse(&self, raw: DecodeOutput) -> Result<Response>; // validate against own schema (G8)
}
```

- **TaskPlan decode strategies:** `PrefillOnly { candidates }` (logit-sliced/trie-scored, §6.10), `ConstrainedJson { schema }`, `ConstrainedPattern { grammar }`, `FreeText { stops, budget }`, `Distribution { scale }` (§7.5). Tasks compose these; the engine doesn't know task semantics.
- **Envelope:** every response carries `schema_version`, the task spec version, the **prompt-template hash** (prompt changes are versioned, diffable events — a quality regression must be attributable to either weights, kernel, or prompt, so all three are hashed), token/timing counts, and the thinking mode used.
- **Presets are data, not code** (`tasks/presets/*.json`): only repo-authored or provenance-cleared packs may ship/embed; every file carries origin/license metadata and a content hash. Users may supply `--preset-file`.
- **Bounded `TaskIR`, not a plugin system:** built-in task plans compile before model load into exact prompt-token segments, grammar/execution program, optional continuation trie, deterministic finite postconditions, resource bounds, and a stage dependency scope: `ItemLocal`, `PartitionReduce`, or `CorpusGlobal`. The scope controls cache/reuse authority—adding one item may preserve local map results but invalidates any reduce/global result whose child set changed. The runtime executes this IR; it never interprets arbitrary code. A later public `.fnlptask.json` recipe may expose only a frozen subset: typed prompt segments/placeholders, supported decode strategies, budgets, source constraints, finite postconditions, calibration references, and dependency scope that cannot be weakened by the caller. It may not execute code/tools, access the network, read undeclared files, embed Jinja/shell/Python/WASM, retry unconstrained, or add neural operators. Unknown keys forward-reject.
- **Prompt-segment ABI:** prompts preserve typed segments (`global policy`, `task instruction`, `document`, `answer scaffold`) rather than becoming an opaque string too early. That permits content-addressed common-prefix discovery without forcing an instructions-last layout. Corpus-major sharing is the default proven layout. Document-major `analyze` packs are eligible only per task after instructions-before vs instructions-after scorecards show no quality regression and cold-prefill equivalence passes; ineligible tasks run independently. No speculative “3.5×” arithmetic becomes a product claim before real token layouts and end-to-end measurements.
- **Typed trust boundary:** only trusted template code may inject role, thinking, or tool-control token ids. An `UntrustedDocument` segment is encoded through a constrained tokenizer path that excludes those ids while preserving the document's decoded bytes (using ordinary/byte pieces); if exact byte preservation without a forbidden id is impossible, preflight rejects instead of silently weakening the boundary. Literal marker text remains ordinary untrusted content. This prevents control-token/role-boundary smuggling, **not semantic prompt injection**: instruction-shaped prose can still steer the model and is addressed by per-task matched clean/attack scorecards. Every output envelope marks fields derived from untrusted content; downstream agents must continue to treat them as data.

### 7.1 `fnlp extract` — schema-constrained structured extraction (flagship)

- **Contract:** user supplies the documented finite JSON-Schema subset (§7.3) + text. A successful result is schema-valid by construction and independently checked. String fields may opt into `"x-fnlp-source":"verbatim"`; then a successful value is a byte-exact source substring by construction. Normalized dates/amounts remain ordinary semantic fields. Repeated occurrences return all compatible intervals or a separately constrained occurrence choice; string membership alone never fabricates a unique offset.
- **Why flagship:** covers a broad set of bounded object/array extraction schemas and demonstrates G8. Unsupported JSON-Schema features reject by keyword; this is not “any schema.”
- Modes: single-shot; `--ndjson` batch; map-reduce for over-context docs (§7.10). Phase 5 may add explicit `--verify-semantic`: each non-verbatim field is rendered by a versioned, schema-aware claim template and checked with §7.6 faithfulness. This is a same-model, correlated second read—not an independent proof or correctness certificate. It defaults off until a locked task eval shows incremental error-catching value after false alarms and cost; results are `{entailed, contradicted, unsupported, not_checked}` with evidence and may feed a policy only after separate calibration. An eligible shared-document prefix may reduce the cost; a cold fallback remains correct and no “10% overhead” is promised before measurement.

### 7.2 `fnlp ner` + `fnlp resolve` — entities, offsets, canonicalization

- **`ner`:** typed spans over a configurable type set. Output uses half-open `{byte_start, byte_end, scalar_start, scalar_end}`; “scalar” means Unicode scalar-value index, not grapheme cluster or UTF-16 code unit. After OQ-17 passes, surface forms use the exact source-language constraint by default. Repeated occurrences remain `ambiguous` with candidate intervals unless a separately constrained occurrence selector resolves them; ordinary non-grounded recipe fields may still be `unanchored`, but are never silently relocated. Every accepted span satisfies `source[byte_start..byte_end] == text` on UTF-8 boundaries. `confidence` is a separately calibrated mention-correctness estimate—not a raw average of whichever tokens encoded the span.
- **`resolve`:** second stage over anchored NER output: lexical blocking → LLM pair scores only within blocks → deterministic graph clustering. Corpus results are independent of arrival/completion order: at flush, mentions sort by `(document_id, byte_start, type, surface)`, candidate pairs sort canonically, scores are frozen, and a named clustering rule runs once. Mention extraction and unchanged pair scores may be `ItemLocal` cache entries; final clustering is `CorpusGlobal`, bound to the complete corpus-snapshot digest, and reruns whenever that child set changes. Incremental heuristic clusters may be emitted only as explicitly provisional events; final ids never depend on scheduler timing and never pretend to be stable across different snapshots. Optional cross-snapshot `unchanged/new/retired/merge/split/ambiguous_match` lineage is Phase-7 work.
- This is the headline "what people used SpaCy for" surface, minus the parts a 3B LLM shouldn't do (§1.2).

### 7.3 The constrained-decoding engine (`grammar/`) — how G8 is real

- **Normative v1 schema subset:** root and nested `type` in `{object,array,string,number,integer,boolean,null}`; `enum`/`const` over JSON scalars; object `properties`, `required`, and **mandatory** `additionalProperties:false`; arrays with `items` and a finite effective `maxItems` (schema value or engine cap); finite string/output byte caps; and the namespaced string annotation `"x-fnlp-source":"verbatim"` after OQ-17. Reject `$ref`, recursion, `oneOf`/`anyOf`/`allOf`/`not`, `patternProperties`, arbitrary `additionalProperties`, `uniqueItems`, `contains`, semantic numeric/string keywords (`minimum`, `multipleOf`, `format`, general `pattern`), and `verbatim-normalized` until each has a valid-by-construction implementation. Canonical object-key order and JSON number spelling are part of the output schema contract. Unsupported keywords fail before model load.
- **Compile:** the accepted schema becomes a bounded typed-JSON automaton: ordered keys with optional/required transitions, enum tries, RFC-8259-compatible string escapes/UTF-8 and canonical number lexers. State/transition counts and estimated mask-cache bytes are computed with checked arithmetic and compared to per-request limits before allocation.
- **The hard part is tokenization alignment** (a token may cross several grammar states). The mask oracle consumes **detokenized byte transitions** through a vocab trie built when the approved tokenizer asset loads; it is cached by `(tokenizer_digest, schema_digest, grammar_version)`. Per-state masks are lazy and byte-budgeted, so an attacker cannot force `states×vocab` memory. SentencePiece word-boundary and byte-fallback behavior belong to the detokenization transducer; raw piece text is never treated as emitted bytes.
- **Source-language product:** a bounded substring automaton over the original source bytes is intersected with the typed-JSON automaton and tokenizer detokenization transducer. The JSON lexer owns escaping while the source automaton sees logical unescaped UTF-8 bytes. Start-anywhere/finish/empty/min-length transitions, repeated occurrences, byte-fallback pieces, and offset recovery are specification—not implementation folklore. Preflight accounts for source-index bytes and rejects over budget. Unsatisfiable required grounded fields return a typed task no-result (nullable fields may emit `null` only if the schema permits it); there is no silent unconstrained fallback.
- **Execution compiler:** each product state emits one exact primitive: `ProjectLegal(all_legal_rows)`, `FeedForced(tokens)`, `CopyFromSource(state)`, or universal `FullProjection(mask)`. Thresholds select between equivalent paths by measured cost, never by dropping candidates. `fnlp schema check` compiles/resources-checks without model load; `schema sample` walks only the supported grammar for DX/fuzz reuse. Model-proposed `schema infer` is Phase-7 experimental because “compilable” is not “correct for the user's data.”
- **Runtime:** execute the compiled primitive; a dense state may still AND the 20.3 KiB legal-token mask into the full sampler scan. Greedy-under-mask is default. Thinking, if explicitly enabled, has its own token/time cap and stays outside the JSON region until the exact pinned close delimiter. Missing close delimiter, timeout, cancellation, unsatisfiable required source field, or output budget exhaustion returns a typed **no-result error**; it never emits a truncated object as successful and never retries unconstrained.
- **Diagnostics, not truth:** optional full-projection audits may report pre-mask argmax legality, legal mass, or best-legal minus best-illegal margin. Sparse projection cannot know illegal logits, so those fields are `not_computed`; candidate-conditional legal probabilities are labeled separately. These signals may reveal model/constraint tension but are **not a fabrication detector, correctness probability, or calibrated confidence** and cannot alone authorize acceptance/escalation.
- **Guarantees:** emitted successful bytes parse and validate against an independent validator; grounded values pass an independent source-membership/offset verifier; EOS is legal only in accepting states; control/template tokens are illegal inside JSON; every reachable nonaccepting state has a path to acceptance within the configured remaining output bound or compilation rejects that request. Fuzzing covers random supported schemas, every token crossing class, byte-fallback/non-ASCII strings, JSON escaping, repeated substrings, empty-language/depth/length boundaries, sparse-vs-full projection equality, forced-vs-sequential KV equality, dead-state search, and cancellation at every emitted token. Separate Lexicon properties prove an untrusted segment cannot emit a control id and decodes to the original bytes; matched injection fixtures measure content steering without mislabeling that empirical result a firewall.

### 7.4 `fnlp classify` — zero-shot classification (logit-sliced)

- Labels + descriptions + text → single- or multi-label, with different probability semantics. **Single-label:** mutually exclusive label continuations, candidate-conditional softmax, frozen length/prior correction for multi-token labels. **Multi-label:** independent yes/no (or present/absent) continuation per label, calibrated and thresholded separately—never a softmax that forces labels to compete. Multi-token candidate sets compile to the exact continuation trie in §6.10 when its measured reuse wins; full-vocabulary teacher-forced and trie-conditional modes remain distinct. Label-query suffixes batch together and may fork a shared document/task prefix; the realized cost scales with unique trie states/label queries and is reported.
- Cost target: one shared document prefill plus batched label suffix/continuation scoring where the cache boundary permits. Compare accuracy, calibration, latency, and memory against a conventional classifier and the fork; do not market “one prefill” when multi-label suffixes add work.
- Presets: topic packs, intent packs, moderation/toxicity dimensions, spam, language-id (the LLM route; `textutil` also ships a non-LLM n-gram detector for wire-speed cases — honesty: use the cheap one when it suffices).

### 7.5 `fnlp sentiment` — dimension scoring (the swiss_army_llama descendant)

- **Model:** newly authored or provenance-cleared focus-area presets (for example reviews, earnings calls, support tickets), each defining adjective dimensions, audience/context, and a `[-100,+100]` interpretation with 0=indeterminate. Do not port ancestor prompt/preset text until §5.7 records authority.
- **Mode A — `--mode distribution` (candidate default after quality proof):** one shared-prefix-cached prefill per `(doc, dimension)` and a closed, census-verified bucket vocabulary. The fast path computes only bucket rows and reports a **candidate-conditional** distribution; it must not report “captured mass,” because that requires the full 166144-row denominator. A full-vocabulary audit path computes true captured mass on the calibration/test sample and establishes whether the prompt reliably puts probability on the scale. Presets include an explicit `UNSURE/OFF_SCALE` bucket. Single-token opaque labels are preferred; multi-token alternatives use a frozen teacher-forced sequence-scoring/length-normalization recipe and require separate calibration. Report normalization mode, E[score], dispersion, entropy, calibration version, and abstention/off-scale decision.
- **Mode B — `--mode sampled --justify`:** independently implemented N-seed generations with grammar-locked score plus free-text justification and preregistered aggregation/interval recipe. It is inspired by the prior project's behavior, not copied code/data.
- **Quality gate (§9.6):** both modes are judged independently against held-out human labels/rankings and reliability targets. Cross-mode agreement is a diagnostic, not ground truth and not a reason to force two genuinely different estimators to coincide. Distribution mode becomes default only if it meets the task metric, calibration, full-vocab captured-mass, and throughput gates; otherwise sampled or abstaining mode remains default.

### 7.6 `fnlp judge` — rubric scoring, faithfulness, pairwise preference

- **Rubric mode:** score texts against provenance-cleared rubric presets; distribution is only a candidate default after the same held-out gates as sentiment.
- **Faithfulness mode:** claim (or answer) + source → `{entailed | contradicted | unsupported}` + evidence spans — the local RAG-verification loop (batchable over a whole retrieval log). NLI-style, classify-path cost.
- **Internal second-reader use:** `extract --verify-semantic` may reuse this exact versioned task, but its same-model correlation and per-field claim rendering remain visible in the receipt. Verification is a diagnostic/decision-policy input whose uplift must be measured per extraction domain, never a certificate attached merely because a second forward ran.
- **Pairwise mode:** A vs B under a criterion → preference + margin (from the two candidates' logprobs, order-debiased by running both orders and averaging — 2 prefills, still cheap). Feeds eval pipelines and rerank-by-judge.

### 7.7 `fnlp redact` + the rest of the portfolio (compact contracts)

- **`redact`:** union source-grounded/anchored NER spans with deterministic rule detectors, then resolve overlaps by a versioned recall-first interval policy. Unanchored model guesses become warnings, never destructive offsets. Actions `mask|placeholder|pseudonymize`; deterministic pseudonyms require a caller-supplied secret/namespace and are explicitly **not anonymization**. `--map-out` uses exclusive create + atomic write, enforces owner-only permissions/ACL or refuses unless the user explicitly accepts an insecure target, never overwrites, and never enters telemetry. `--verify` reruns the full detector union over the redacted output and returns a nonzero result plus leak report on residual findings; it is a useful clean-pass receipt, not a compliance certificate. Redaction remains defense-in-depth.
- **`summarize`:** length/style presets (`--words`, `--bullets`, `--tldr`), map-reduce over §7.10 for long docs, optional `--grounded` (every bullet carries source-span evidence, verified like §7.1).
- **`keyphrases`:** ranked phrases, constrained list output, exact-match anchoring, `--max N`.
- **`answer`:** context-grounded QA: question + provided passages → answer + span citations + abstain (`answerable: false`) calibrated per §7.8 — the RAG companion (retrieval itself is frankensearch's job; we consume its output NDJSON shape).
- **`generate` / `chat`:** completion/chat with the native template, bounded thinking, declared XML/JSON tools, seeded sampling, and streaming NDJSON. `fnlp` parses/emits tool-call data but **never executes tools**.
- **`textutil` (non-LLM, wire-speed):** `fnlp tokens` (count/encode/decode — agents constantly need exact token counts for *this* tokenizer), `fnlp split` (sentences/chunks with byte+token budgets — the map-reduce feeder), unicode/whitespace normalize, the n-gram language detector. Zero-model-load fast paths (tokenizer-only).

### 7.8 Confidence & calibration (cross-cutting)

Raw LLM logprobs are not confidence claims. Each calibrated task has disjoint development, calibration, and locked test ids. Temperature/isotonic parameters fit only calibration data; reliability/ECE/Brier/selective-risk is reported only on the locked test. Conformal abstention/prediction sets are used only where the exchangeability/unit-of-analysis assumptions are written and checked; coverage is stated for the named dataset/population and finite sample, never universally. Recipe, prompt, label set, thinking/numerics mode, calibration digest, and validity date key the artifact. Distribution shift invalidates the coverage claim and falls back to raw scores labeled `uncalibrated` or conservative abstention—never silent extrapolation.

**Selective automation is static and user-owned first.** After a task has a valid calibration artifact, a versioned `DecisionPolicy` may map task-valid signals to `{accept, explicit thinking retry, explicit N-seed aggregate, abstain, review_spill}` under a declared loss table. The initial policy is an offline-fit deterministic table; all retries/discards count in cost and every accepted result records the authorizing signals/policy. Constraint diagnostics from §7.3 are advisory unless independently calibrated on the same task. Corrections imported into an explicitly named local suite may support a later `eval/calibrate/policy fit`; there is no hidden online learning, prompt mutation, cloud escalation, or automatic incorporation into project scorecards. Ordinary task output/abstention is the immediate fallback.

Review-spill corrections are **selection-biased toward cases the current policy already considered hard**; they cannot estimate error among confidently accepted production results or silently extend a qualification's validity. Only an independently sampled audit with known inclusion probabilities and human-authorized grades (AA-A1) may update that accepted-population evidence. Qualification/policy artifacts name population assumptions, validity/expiry conditions, and shift indicators; invalidation yields `uncalibrated`, increased review/abstention, or a new qualification—not hidden online refitting.

### 7.9 Thinking-mode policy

Conservative pre-evaluation default is **off for every structured/batch task**, including judge and answer; `generate/chat` honors the model-template default. Each task exposes a bounded explicit override. Thinking tokens never enter the task JSON, prompt caches are keyed by mode, and a missing close delimiter fails as described in §7.3. Phase 5 may promote thinking per task only when the locked eval's accuracy/selective-risk gain justifies p95 latency, energy, and output-token cost.

### 7.10 Long-document map-reduce (`mapreduce.rs`)

Shared spine for summarize/extract/score-over-long-docs: deterministic token-budgeted chunking → batched map → task-specific deterministic reduce, with every output linked to chunk ids/byte ranges. Because 44-deep KV makes even 32K expensive and full attention is quadratic in prefill, map-reduce is a primary operating mode, not an edge case hidden behind the reported 256K maximum. Conflict/dedup rules, score weights, reduce fan-in, and lossy-summary warnings are versioned; hierarchical reduction never claims exact equivalence to single-context inference.

---

## 8. The `fnlp` CLI design

### 8.1 Binaries & entrypoint

Two `[[bin]]` — `fnlp` (short, what humans/agents type) + `franken_nlp` (long) — each a one-line shim over `franken_nlp::cli_main()` (the franken_ocr doctrine-#9 pattern verbatim, including the never-two-targets-one-file rule). `fn main()` synchronous; ShutdownController installed before dispatch.

### 8.2 Subcommand surface

| Subcommand | Purpose |
|-----------|---------|
| `fnlp extract --schema s.json [doc.txt \| -]` | flagship structured extraction |
| `fnlp ner [--types ...] / resolve` | entities + canonicalization |
| `fnlp sentiment --focus-area X [--mode distribution\|sampled]` | dimension scoring |
| `fnlp classify --labels a,b,c [--multi]` | zero-shot classification |
| `fnlp judge --rubric r \| --faithfulness --source s \| --pair a b` | scoring/judging |
| `fnlp redact [--policy pii-default] [--map-out map.json]` | PII redaction |
| `fnlp summarize / keyphrases / answer` | the rest of the portfolio |
| `fnlp generate/chat [--think/--no-think] [--seed N]` | generation, streaming |
| `fnlp batch --task <t> [--task-args f.json]` | **NDJSON daemon: docs in on stdin, results out on stdout (§8.4)** |
| `fnlp job start/status/resume/verify/materialize` | opt-in durable corpus execution with digest-bound receipts (§8.4) |
| `fnlp job partition/merge` | Phase-7 portable snapshot shards after single-host scope semantics are proven |
| `fnlp schema check/sample` | model-free supported-subset validation and valid-instance sampling |
| `fnlp recipe check/explain/sample/run` | bounded data-only `TaskIR` recipes (Phase 5; no executable/plugin surface) |
| `fnlp eval/calibrate/qualify` | user-owned scorecards, calibration, and candidate-vs-active qualification |
| `fnlp audit plan/grade` | Phase-7 human-authorized acceptance sample for one frozen owned job; never a universal certificate |
| `fnlp tokens / split` | non-LLM utilities (no model load) |
| `fnlp pull [--quant int8\|int4]` | fetch + verify + install release artifacts (§5.6) |
| `fnlp convert <shards> -o m.fnlpq [--quant] [--arch]` | offline weight transformation |
| `fnlp models [activate/rollback]` | installed artifacts, compatibility, explicit digest-bound activation |
| `fnlp tune [--quick\|--full]` | Phase-6 local selection among already-proved bit-identical paths only |
| `fnlp resident start/status/stop` | AA-R1 Phase-7 owner-only local IPC experiment; never a routable service |
| `fnlp robot schema/health/backends/selftest` | agent surface: contract, diagnostics, ISA tiers, **bit-exact kernel self-proof on THIS cpu** |
| `fnlp runs / sync export-jsonl` | fsqlite run history / audit export |
| `fnlp doctor` | idempotent self-check/repair (artifact resolution, format versions, cache perms) |

Conventions: stdout is data, stderr diagnostics; `--json` on every task; `-o file` infers format; bare `fnlp` prints help (never a TUI); `NO_COLOR`/`CI`/`TERM=dumb` honored.

### 8.3 Robot / NDJSON contract

Versioned NDJSON events (`schema_version` on every line): `run_start`, `stage`, `doc` (per-document result in batch), `token` (streaming generate), `run_complete`, `run_error`. `robot schema` self-describes; a frozen-schema contract test guards it. Deterministic under greedy: same input+args+artifact → byte-identical output (a gate, §9.2).

### 8.4 The batch daemon (`fnlp batch`)

Long-running process; NDJSON requests on stdin (`{id, text, task_args?}`), bounded by line/document/token/schema/output sizes before admission. Default emits completion-order results with ids; canonical byte replay requires `--ordered`, a declared maximum reorder window, and telemetry-free output. Backpressure pauses reads; duplicate ids are rejected within the live window; one malformed document emits one `doc_error` without corrupting framing. EOF requests a graceful drain; first SIGINT cancels admission and drains within a budget, second follows the documented immediate-cancel path. `resolve` collects mentions and emits deterministic final clusters only on a numbered flush/EOF transaction. Weights load once.

**Durable jobs are a separate opt-in contract, not magic attached to an arbitrary pipe.** `fnlp job start` canonicalizes a replayable manifest and freezes a corpus-snapshot digest plus semantic execution key over ordered input ids/bytes/normalization, artifact and packing, tokenizer/template/prompt/grammar/TaskIR including dependency scopes, task args, calibration/policy, numerics/KV/thinking/sampling, output schema, and engine semantic version. Cache entries name their stage and scope: `ItemLocal` may reuse an exact item key; `PartitionReduce` must rerun its deterministic reduce when the complete child-set digest changes; `CorpusGlobal` authority is valid only for the exact snapshot. fsqlite transactionally records `pending → admitted → running → result_committed → materialized`; interrupted pre-commit states are retryable. A result becomes complete only after its bytes/digest are durable. `resume` refuses any key mismatch with a field-level diff; `verify` checks journal/spool/materialization consistency; `materialize --ordered` writes one canonical record per item.

The authority boundary is explicit: when the job owns a checksummed spool and output materialization, it can promise one canonical committed record per item. Raw stdout to an arbitrary consumer can promise only stable ids and documented at-least-once replay. Default journal tables contain ids, digests, state, attempts, metrics, and errors—**not input text or output bytes**—so resume requires the original manifest. `--spool-input`, `--spool-results`, and incremental result caching are explicit privacy/storage choices with owner-only permissions, byte/retention budgets, and inspect/purge commands. Kill/disk-full/corrupt-tail injection at every transition must show uninterrupted ≡ resumed semantic output in ordered deterministic mode.

Phase-7 acceptance sampling, if AA-A1 survives review, operates only on a frozen, verified owned job population. `audit plan` deterministically selects ids from preregistered strata using SHA-256 ranking over `(job_digest, audit_seed, stratum, item_id)`; it does not materialize text/results unless the operator explicitly requests a protected review pack. `audit grade` accepts human-authorized labels and reports only the finite-population/stratified claim its frozen design supports. Model self-grading cannot authorize acceptance. The receipt binds the population, strata, estimator, risk thresholds, sample ids, grader provenance, and missing/invalid grades. Post-hoc stratum or threshold changes create a new audit, never rewrite the old claim.

### 8.5 Exit codes & env

`0` ok · `1` generic · `2` usage · `3` model not found · `4` input decode/parse error · `5` budget/timeout · `6` cancelled · `7` artifact integrity/format/version mismatch · `8` schema-compile error · `9` admission/resource limit. The CLI snapshots `FNLP_*` once into `NlpEngineBuilder`; the library never reads environment variables behind its caller's back. Capability detection may be process-global, but thread/batch/context/memory/numerics/NUMA/mmap choices are engine configuration. Forced ISA runs selftest first and fail on absent capability.

### 8.6 Threat, privacy, and resource model

| Boundary | Failure/attack | Required control and proof |
|---|---|---|
| `.fnlpq` / manifest | integer overflow, overlapping ranges, path traversal, digest confusion, decompression bomb | no compression in v1; checked/capped parser; canonical part ids; whole/part/source digests; fuzz + malicious corpus |
| Schema/grammar | exponential states or mask-cache exhaustion | normative subset; preflight state/byte estimate; per-request and global cache caps; cancellation; adversarial-schema fuzz |
| NDJSON/text | giant lines, invalid UTF-8, duplicate ids, output amplification | byte/token/depth/output caps before allocation; streaming parser; bounded reorder window; typed per-doc errors |
| Untrusted document segment | role/control-token smuggling; instruction-shaped content steering | forbidden-control-id tokenizer with exact-byte decode-or-reject; typed segment provenance; matched clean/attack task scorecards; never market an injection firewall |
| Prefix/KV cache | cross-user content reuse or use-after-cancel | shipped-prefix-only default; namespace/digest-complete keys; COW page ownership model; hostile interleaving tests |
| Pull/network | manifest substitution, redirect abuse, overlong body, resume confusion, partial install | release-bound embedded manifest; explicit private-manifest digest; HTTPS + redirect/credential policy; declared-length caps while streaming; revalidated part-boundary journal; per-part/whole/source/license hashes; staged census/native-pack selftest + atomic activation |
| Local persistence/jobs | accidental prompt/result/PII retention; torn or mixed-config resume | disabled by default; metadata field allowlist; explicit owner-only spools; exact semantic key; transactional states; kill/disk-full/corrupt-tail tests; reversible maps never enter telemetry |
| Model output | instruction-shaped text, unsafe tool text, PII miss | output is untrusted data; segment provenance preserved; `fnlp` executes no tools; redaction is explicitly not a compliance guarantee; evidence/abstention surfaced |
| Audit/qualification | model grades itself; cherry-picked/post-hoc sample; privacy leak in review pack | human authority; frozen job/risk/strata/seed; exact population binding; explicit protected materialization; scoped claim or `INSUFFICIENT_DATA` |
| CPU/memory | denial by long context/batch/thinking | admission certificate, lazy KV pages, per-stage deadline/cancellation, bounded thinking/output, fair prefill morsels |

---

## 9. Verification & conformance methodology

Apply `/porting-to-rust` (spec-first), `/testing-conformance-harnesses`, `/testing-golden-artifacts`, `/running-the-gauntlet-on-your-rust-port`. **Parity gate FIRST, perf second.**

### 9.1 The reference oracle

- **Primary:** pinned HF environment (`transformers==4.51.0`-line per generation_config — exact pin decided in Phase −1; `torch` CPU build) running `NanbeigeForCausalLM` with `trust_remote_code` — **a text-only dense model, so a CPU oracle is expected to run as-is** (unlike franken_ocr's CUDA-oriented `infer()`); Phase −1 proves it and records the oracle's own nondeterminism floor (two runs × two thread counts; tolerance derived from *measured* variance, never inherited).
- **Secondary oracles:** the `nanbeige42` llama.cpp fork (token streams at matched quant; also the perf baseline) and, opportunistically, `rlx-nanbeige` — three-way triangulation on loop semantics.
- `scripts/gen_reference_fixtures.py`: per-layer/per-loop hidden-state dumps (post-embed; per (loop, layer); pre-lm_head; logits), greedy token streams for the golden prompt corpus, template renderings across the mode matrix, tokenizer id sequences over the multilingual/code/special corpus, sampled streams at fixed seeds where reproducible.

### 9.2 Parity ladder (gates L0–L5)

| Gate | Granularity | Tolerance |
|------|-------------|-----------|
| **L0 tokenizer + template** | token ids; rendered prompts (all thinking/tools modes) | **exact** |
| **L1 per-op** | each kernel vs scalar/f32 oracle activation | integer exact; float max-abs/max-rel/ULP/cosine vector with thresholds derived per op from the pinned oracle floor—never cosine alone |
| **L2 per-(loop, layer)** | all 44 layer outputs **plus both post-loop final-RMSNorm states** | the full L1 metric vector; loop-1 post-norm/loop-2 input is a named equality fixture |
| **L3 logits** | pre-sampling logits | within *measured* quant tolerance; **argmax exact** where oracle deterministic |
| **L4 tokens** | greedy decoded stream | **exact** over the oracle-reproducible prefix |
| **L5 task outputs** | end-to-end task JSON on the golden corpus | exact where deterministic (constrained greedy); score drift within documented band |

Determinism is scoped: semantic greedy outputs are exact under one named numerics profile; canonical byte replay additionally fixes ordering and excludes timings/run ids. Batch-M≡batch-1 and prefix-fork≡cold are exact only for kernels that preserve the canonical reduction order; approximate numerics modes carry separately named tolerance/quality contracts and cannot silently become the deterministic default.

### 9.3 Differential / metamorphic / golden / fuzz

- **Differential:** ours vs oracle(s) per-op and end-to-end; ours-int8/int4 vs ours-f32; every integer SIMD tier vs canonical scalar/i64; every packed layout vs logical weights.
- **True metamorphic invariants:** batch composition and prefix fork equivalence; Generic→arch packing preserves logical tensors; ordered NDJSON scheduling preserves per-id semantic results; grammar success validates independently; grounded values occur in source under the named byte contract; untrusted-segment token ids exclude every control id while decoding to the same document bytes; sparse legal-row projection equals full projection after masking; forced causal micro-prefill equals sequential feeding; every continuation-trie execution mode equals naïve candidate scores; killed-and-resumed owned jobs equal uninterrupted jobs; an item-only snapshot change preserves exact unaffected `ItemLocal` entries but invalidates/recomputes every changed `PartitionReduce`/`CorpusGlobal` authority; anchored NER offsets survive deterministic coordinate conversion; artifact splitting/reassembly preserves the whole digest. Label-order sensitivity, content-injection response, negation response, prompt-segment order, and document-concatenation effects are **quality probes**, not invariants—the model is allowed to be context/order sensitive, so those results are reported rather than asserted mathematically.
- **Golden** (`/testing-golden-artifacts`): frozen task outputs (exact for constrained greedy; scrubbed for timings/ids), CLI help, robot schema — `UPDATE_GOLDENS=1` with mandatory diff review.
- **Fuzz** (`/testing-fuzzing`): tokenizer (arbitrary input → encode/decode never panic; the correctness bar is **parity with the reference tokenizer's own canonical behavior** — SP normalization means `decode(encode(x)) == x` is *not* a general property, so byte-losslessness is asserted only on paths where the reference itself is lossless, e.g. byte-fallback; the separately declared untrusted-segment path must byte-round-trip or reject and must never emit a control id), schema/TaskIR compiler (arbitrary input → compile-or-clean-error, never dead/unbounded or executable), grammar/source products (random walks parse; grounded fields independently occur; adversarial Unicode/escapes/repetition/empty languages), `.fnlpq` reader (malformed headers rejected loudly), NDJSON daemon (malformed lines → doc_error, stream survives), job journal (kill/disk-full/torn-tail recovery), and audit sampling/math (population/order invariance, golden exact enumeration on small finite populations, missing-grade fail-closed behavior).
- **Model-gated e2e:** ordinary CI reports an explicit `SKIPPED_NO_MODEL` result rather than a counterfeit pass. Release certification requires the armed job and records artifact digest + proof that the native path ran (all fallback locations pointed at `/nonexistent`).

### 9.4 Ledgers & scorecard

`docs/DISCREPANCIES.md` (every accepted divergence: reference behavior, ours, measured impact, kill-switch, review date) · `docs/NEGATIVE_EVIDENCE.md` + `docs/PERF_LEDGER.md` (artifact-graph format: claim_id, evidence, fixture hashes, CPU feature string, exact command+env, disposition) · `docs/FEATURE_PARITY.md` (FeatureUniverse/SurfaceMatrix: every subcommand, robot event, exit code, env var, task, preset — `present|partial|missing|n/a`; partial never rounds up) — all inherited formats from the shipped franken_ocr practice.

### 9.5 The gauntlet — three-pillar release certification

Three independent pillars: **(a) Performance** (per-stage/per-regime distributions and fair baselines), **(b) Conformance** (L0–L5, property/model-check/fuzz, task evals), and **(c) Surface parity** (FeatureUniverse). Load-bearing deterministic claims use the right authority: checked arithmetic/formal bounds for accumulators and capacity; exhaustive/property differential tests for SIMD; bounded-state model checking plus hostile interleaving replay for scheduler/cancellation; independent validation + grammar fuzz for JSON; exact fixtures for batch/prefix determinism. Statistical task metrics use preregistered point estimates and confidence intervals on locked data. There is no dimensionally meaningless “Beta-posterior × conformal” aggregate and no statistical process standing in for a proof. Release requires every hard gate green and ≥2 consecutive clean full runs after the last load-bearing change; the broader review workflow still targets ≥10 adversarial rounds.

### 9.6 Task-quality evaluation (the layer franken_ocr didn't need)

Kernel parity proves faithful execution, not useful tasks. Every dataset first enters `docs/eval/DATASETS.md` with immutable source, exact license text/terms, allowed use/redistribution, preprocessing digest, unit of analysis, development/calibration/test ids, and known contamination/leakage risks. A script may download only when the terms permit automated access; otherwise the harness accepts a user-supplied file and verifies its digest. “Fetch by hash” is not a substitute for permission. Only repo-authored fixtures or explicitly redistributable data are committed.

- `ner`: candidate corpora include WNUT-17 and Few-NERD **only after their exact upstream licenses are archived**, plus a repo-authored modern-web/CJK set. CoNLL/OntoNotes/Reuters/LDC-class candidates remain user-supplied when terms require it. Report span-exact/type-relaxed F1, anchoring status, and performance by language/type.
- `extract`: schema-diverse fixture corpus (field-F1, 100% validity by construction — validity is asserted, accuracy measured).
- `classify`: candidate sentiment/topic/intent corpora (SST-2/AG-News/Banking77-class) after the same license gate, plus owned multilingual smoke; accuracy/macro-F1/Brier/ECE/selective risk.
- `sentiment`: disjoint calibration/test splits with human labels/rankings, rank correlation/MAE/reliability/selective risk, full-vocab captured-mass audit, and cross-mode agreement as a diagnostic. Directional negation probes remain sanity reports, not universal linguistic laws.
- `judge`-faithfulness: licensed/owned entailment labels; `answer`: answerability + citation/abstention sets; `summarize`: length/coverage plus human-annotated factuality/citation subset (any external LLM judge is disclosed and never the sole gate); `redact`: synthetic and licensed realistic PII with recall-first reporting, precision and unresolved-anchor rate.
- Every document-consuming task includes matched clean/adversarial-content fixtures across role-marker text, refusal steering, false-negative steering, value steering, and tool-call bait. The structural control-id property is an invariant; **content-injection resistance is only an empirical per-task attack-success metric** and may block a consequential preset without supporting a universal “prompt-injection safe” claim.
- Every scorecard states: model recipe id, thinking mode, prompt hash, dataset version. Regressions gate releases at the same rank as parity.

Assay is also a user surface, but a user scorecard is never promoted into a universal project claim. `fnlp eval` consumes versioned task-shaped NDJSON and emits the same digest-bound metrics; `calibrate` requires a disjoint calibration split; `qualify` performs paired candidate-vs-active replay under caller-declared gates and writes a host/dataset/config-scoped receipt. Dataset overlap/duplicate detection, tiny-sample `INSUFFICIENT_DATA`, stale-digest substitution, and activation rollback are tested. `models activate --require-qualification` accepts only an exact matching receipt and leaves the previous content-addressed configuration available; installation and activation are distinct. User corrections enter only an explicitly named local suite.

---

## 10. Performance methodology

Apply `/profiling-software-performance` then `/extreme-software-optimization`; `/alien-graveyard` + `/alien-artifact-coding` for the advanced families. Profile-first (no hotspot → no change), one lever at a time, keep-gate + revert discipline, `.bench-history` ratchet.

### 10.1 Regimes, rooflines, and honest ceilings

Profiles and ledgers are kept **per regime** because the cost center moves: **(R1) latency generate** (batch 1, decode-bound); **(R2) corpus scoring/classification** (prefill-dominated, prefix-cached, batch-M); **(R3) corpus generation** (summarize/extract batch, mixed); **(R4) long-context** (≥32K, attention/KV-bound). Per `(host fingerprint, artifact recipe, kernel table, regime)`, measure effective bandwidth and int8 throughput with the same access shapes, then compute memory/compute lower-bound models. STREAM peak and vendor TOPS are context, not attainable denominators. A roofline gap becomes actionable only when counters/profile samples identify its cause and its EV score clears §10.4. Illustrative bandwidth-only ceilings below use a hypothetical 3.7 GB/token recipe and nominal memory bandwidth; they are not predictions:

| Machine | ~BW | decode ceiling |
|---|---|---|
| M4 Pro | ~273 GB/s | ~70 tok/s |
| M4 Max | ~410 GB/s | ~111 tok/s |
| TR 5995WX (Zen 3, 8ch DDR4) | ~205 GB/s | ~55 tok/s |
| TR 7995WX (Zen 4, 8ch DDR5) | ~333 GB/s | ~90 tok/s |

Batch-M shifts R2/R3 toward compute floors (the §6.7 amortization) — the measured docs/min table per machine is the headline artifact of Phase 4.

### 10.2 The optimization loop (mandatory)

The mandatory loop is: (1) freeze claim, code/artifact SHA, host fingerprint, workload, counters, warmup and baseline distribution; (2) change **one causal lever**; (3) prove the lever's equivalence/tolerance contract; (4) rerun randomized A/B trials through thermal steady state and report median, p95, p99, confidence interval, energy where available, and build/size cost; (5) keep only when the confidence interval and practical threshold both win, otherwise restore the baseline implementation while retaining the patch/commands/results in `NEGATIVE_EVIDENCE`. Coefficient of variation is diagnostic, not a universal 5% law. Criterion may be a dev-only harness if justified; the benchmark protocol, not a framework name, is authoritative.

### 10.3 Head-to-head gauntlet vs the real baselines

`benches/gauntlet` drives the pinned fork and CPU HF-if-proven with thread/allocator/precision/prompt/output parity. Trials are randomized paired A/B after warmup and thermal stabilization; report distributions and confidence intervals, never best-of-N. R2 runs identical prompts/settings through the fork's completion surface, then attributes grammar/prefix/task-layer savings separately. Rows name quant classes rather than pretending Q8_0 and our int8 are numerically identical. Slower rows remain. R4 context points are selected from admitted memory on the host, not hard-coded to 32K when it cannot fit.

### 10.4 Profile-gated lever queue and EV rule

Greenfield rankings are hypotheses, not profiles. The initial queue is: layer-major/morsel scheduling and prefix reuse (R2/R3); int8→int4 traffic reduction (R1); fixed-shape prepacked GEMM/GEMV (R1/R2); candidate-row lm_head (closed-set tasks); GQA attention/KV paging (R4); then fusion/transcendentals, NUMA/thread placement, and build recipes. A card may enter implementation only when a measured hotspot maps to it and

`EV = (Impact × Confidence × Reuse) / (Effort × AdoptionFriction) ≥ 2`.

Scores are integers 1…5 with one-sentence evidence for each factor. Reuse matters: a kernel/scheduler lever that benefits four regimes outranks an exotic controller for one preset. One card, one causal change, one evidence ledger entry. If the hotspot disappears, the card returns to the queue.

### 10.5 Alien-artifact recommendation cards (evidence before exotic machinery)

These ideas come from the alien-artifact/graveyard review, but the mathematics is admitted only where it compresses a real decision. No subsystem composes more than three mathematical families without a written compatibility/overhead review; redundant tail methods are forbidden. Runtime adaptation begins only after a deterministic static policy is green and follows `offline replay → shadow → canary → explicit default`, with an immediate deterministic fallback.

| Card | State/action/loss and why it fits | Evidence and promotion gate | Deterministic fallback |
|---|---|---|---|
| **AA-K1 Certified fixed-shape kernel search** (high reuse) | Offline state = model shape + ISA + cache fingerprint; actions = bounded tile/unroll/packing/fusion candidates; loss = weighted p50/p95/p99 + code size, subject to exact/tolerance constraints. Normalize→enumerate/rewrite→verify, not a runtime optimizer. | Profile names GEMM/GEMV hotspot; search manifest, candidate graph, generated-code digest, scalar/i64 differential certificate, compiler/host pin, and randomized A/B gauntlet. Generated source is human-reviewed and reproducible; no regex mass edit. | Current hand-written scalar/prepacked kernel and measured dispatch table |
| **AA-Q1 Joint rate-distortion bit allocation** | Offline multiple-choice budget: each tensor chooses `{bf16,int8,int4-g32,int4-g16}`; objective combines bytes/latency with held-out task distortion. Per-tensor curves seed search, but loop reuse/cross-tensor interactions make separable water-filling only a heuristic. | Uniform recipes first; development-set search, then complete candidate artifacts on disjoint calibration/test; exact footprint; mean + CVaR₀.₁ error; repeated-seed stability; allocation table/provenance in `.fnlpq`. No claim of global optimality without a valid dual bound. | Best uniform recipe inside the same byte budget |
| **AA-S1 Communication-avoiding morsel scheduler** | State = bounded queues, sequence phase/position/pages/deadline; actions = admit, form compatible row group, choose prefill morsel, decode step, cancel/drain; loss = p99 latency + throughput + memory subject to fairness/capacity invariants. Layer-major execution reduces weight communication; queueing models describe load, not authorize unsafe admission. | Arrival/service traces, Little's-Law audit, byte-exact capacity certificate, bounded-state automaton/model checking, hostile cancellation/interleaving replay, and R2/R3 tail benchmarks. Static table first; at most one later adaptive controller at this layer. | Fixed batch cap + fixed morsel + FIFO admission |
| **AA-W1 Cross-loop physical-layer wavefront** (research only) | Different ready sequences at physical layer `i` but different semantic loops may share identical q/k/v/o/MLP weights for linear work; KV destinations and dependencies remain loop-tagged. Actions coalesce only already-ready compatible rows—never parallelize the two loops of one sequence or delay latency work to manufacture a pair. | **First** trace loop-stage occupancy/partial weight sweeps. If fragmentation exists and EV≥2: model-check the readiness DAG, split linear/stateful compatibility, prove every sequence equals loop-major, and show fewer partial sweeps plus R2/R3 p50/p95/p99/energy wins. Dense lockstep batches imply zero opportunity and graveyard the card. | Ordinary loop-major compatibility key including `loop` |
| **AA-C1 Scan-resistant prefix/grammar cache** | Candidate S3-FIFO admission/eviction only if LRU traces show one-hit scan pollution; loss = miss cost + cache bytes + p99 lock time. | Replayed key trace, hit/byte-hit/eviction/lock metrics, privacy namespace proof, workload-shift test. Do not add because the algorithm is fashionable. | Byte-bounded deterministic LRU |
| **AA-T1 Tail-aware quality gate** | Use **CVaR only** as the optimization tail objective for per-document quality loss; do not stack EVT/large-deviation terminology on the same quantity. | Preregister α, loss, grouping, bootstrap interval, minimum subgroup sizes, and paired candidate/baseline test on locked data. Mean metrics remain visible so tail wins cannot hide broad regressions. | Uniform recipe or promote offending tensors one tier |
| **AA-D1 Exact loop-draft research** (high risk) | Draft proposes several tokens from loop-1; target two-loop model verifies them with an exact speculative-decoding acceptance rule. State/action includes draft length and fallback; loss includes target calls, wasted draft work, p99, and **exact token-distribution preservation**. Confidence-threshold acceptance is explicitly approximate and cannot use this name. | First prove algorithm algebraically and against a tiny enumerated model; then token-stream/distribution differential, acceptance/waste profile, full L4/L5. Loop-1 `lm_head` cost is included. If no exact verifier or net win, record the graveyard result. | Ordinary two-loop decoding |
| **AA-U1 Task uncertainty** | Calibration/abstention from §7.8; this is product semantics, not kernel control. Conformal methods only under stated exchangeability; no e-process ornament. | Disjoint splits, coverage/selective-risk/reliability with confidence intervals and shift invalidation. | Raw scores labeled uncalibrated or conservative abstention |
| **AA-A1 Frozen-corpus acceptance/decay audit** (human authority) | State = one verified owned-job population + preregistered strata/inclusion probabilities/risk contract + human grades; actions = sample, accept, reject, expire/invalidate qualification, or no-claim; loss matrix prices review effort, false acceptance, false rejection, and stale qualification. SHA-256 ranks unique ids without replacement using an externally supplied seed committed after population freeze (ties break by canonical id); model-derived signals may stratify but never grade. | Independent derivation/golden enumeration of the exact finite-population or preregistered weighted/stratified interval; simulations for design power only; every accepted item has known nonzero inclusion probability when the design claims accepted-population risk; population/design/seed/sample/grades immutable in the receipt; seed provenance/retries visible; missing/invalid grades and post-hoc design changes fail closed. Claim is an error-rate decision for that job/population or longitudinal validity window, never correctness of each item or a universal certificate. | No audit/continuing-qualification claim; caller reviews more items, reviews all, increases future audit coverage, or declines acceptance |
| **AA-R1 Local resident rendezvous** (research only) | State = locally observed independent `fnlp` clients, loaded-artifact copies, pool contention, endpoint/lifecycle state; action = keep per-process engines or multiplex the existing framed robot contract through one owner-only OS-local IPC endpoint. Loss includes RSS/cold-start/p99/fairness plus lifecycle/security complexity. No routable listener or remote credentials. | First trace multi-process artifact loads, memory, core oversubscription, and latency on agent-heavy hosts. If EV≥2: owner/peer permissions, namespace-complete cache keys, per-client admission/cancellation, protocol negotiation, drain/restart receipts, hostile disconnect tests, and result=batch-1 equivalence. Cross-platform local IPC must use approved suite/std surfaces or the card stays research. | Current in-process library/CLI and cooperating `fnlp batch` pipe |

Every implemented card gets a recommendation record with hotspot evidence, EV factors, equations/units, assumptions, p50/p95/p99, artifact/license provenance, interaction matrix, rollback command/config, and the exact claim it is allowed to support. Adaptive cards also name state space, actions, loss matrix, calibration signal, shadow/canary thresholds, and interference tests. “Alien” means unusually rigorous evidence and composition—not unusually elaborate vocabulary.

### 10.6 Idea-wizard disposition (what survived adversarial review)

`WIZARD_IDEAS_COD.md` and `WIZARD_IDEAS_CC.md` are preserved as non-normative ideation provenance. The plan, not those files, controls. Review applied the graveyard rule that a clever idea is valuable even when the correct disposition is “rewrite,” “defer,” or “measure and kill.”

| Proposal | Disposition in this plan | Why / authority boundary |
|---|---|---|
| Stencil execution compiler: all-legal row projection + forced causal runs | **ACCEPT, Phase 4** | Exact primitives with full/sequential fallback; separate attribution and state/KV differentials |
| Byte-exact source-grounded fields | **ACCEPT behind OQ-17, Phase 4** | Stronger product guarantee; v1 is `verbatim` only, repeated occurrences stay explicit, no normalized-offset fiction |
| Exact continuation-trie scoring | **ACCEPT, Phase 4** | Complete finite-language DP; naïve scorer fallback; no pruning or normalization relabeling |
| Durable/incremental corpus jobs | **ACCEPT, Phase 4** | High-certainty operational value; owned-spool authority is distinct from arbitrary stdout; persistence is opt-in |
| User `eval/calibrate/qualify` + safe activation | **ACCEPT, Phases 5–6** | Reuses required Assay machinery and scopes every receipt to exact data/config/host |
| Bounded public task recipes | **DESIGN NOW; OPEN IN PHASE 5** | Internal `TaskIR` first; public data-only format waits for built-in equivalence, caps, and no-code/no-network proof |
| Document-major multi-task packs | **EMPIRICAL ELIGIBILITY; PHASE 7** | Prompt order can change quality; typed segment ABI is adopted, instructions-last and speedup numbers are not |
| Selective automation/correction flywheel | **STATIC SUBSET ONLY, PHASE 5** | Offline deterministic policy + explicit review spill; no self-training, cloud call, or compound adaptive controller |
| `schema check/sample` | **ACCEPT, PHASE 4** | Deterministic compiler/fuzzer reuse; zero model required |
| `schema infer` | **DEFER, PHASE 7 EXPERIMENT** | A schema can be compilable yet semantically wrong; holdout utility must be measured |
| Per-install `fnlp tune` | **ACCEPT LATE, PHASE 6** | May select only already-proved bit-identical paths; stale/mismatched profile ignored; shipped defaults remain |
| Cross-loop wavefront coalescing | **AA-W1 RESEARCH CARD ONLY** | Benefit exists only under measured loop-stage fragmentation; otherwise negative evidence is success |
| Semantic-field second reader | **EXPERIMENT, PHASE 5** | Same-model faithfulness check may catch errors but is correlated and task-dependent; never a certificate or default without incremental locked-eval value |
| Typed untrusted-document boundary | **ACCEPT STRUCTURAL CORE, PHASE 1; EVAL PHASE 4** | Control-id smuggling can be made impossible with byte-preserving encode-or-reject; prose steering remains empirical and explicitly unproved |
| Frozen-job acceptance sampling | **AA-A1, PHASE 7** | Human grades + preregistered finite-population claim can support scoped sign-off; model self-grading/post-hoc sampling cannot |
| Scope-correct corpus snapshots/portable partitions | **TYPE SCOPE NOW; PARTITION/MERGE PHASE 7** | Item-local cache authority cannot be reused as corpus-global authority; portable receipts are useful only after single-host jobs prove exact snapshot semantics |
| Multi-client local resident | **AA-R1 RESEARCH CARD, PHASE 7** | Agent hosts may otherwise duplicate multi-GB weights/pools, but local IPC/lifecycle/security is a new product surface and requires traces first |
| “Constraint pressure = fabrication detector” / “margin is free” | **REJECT / REWRITE** | Neither implication is valid; illegal logits/full mass cost a full projection, and tension is not correctness confidence |
| Automatic think→self-consistency escalation | **DEFER** | Compounds calibration, latency, and controller interactions; explicit user actions/static policy come first |
| Prompt-boundary token healing as parity fix | **REJECT AS DEFAULT** | It deliberately changes the pinned model's token context; it may be a separately evaluated recipe choice, never L0 parity |
| Raw floating-logit hashes as portable selftest | **REJECT** | Reduction/compiler/quant modes make bit hashes the wrong authority; use integer differentials and tolerance-scoped golden canaries |

---

## 11. Phased roadmap

Each phase: goals · key tasks · exit gates. Correctness before speed throughout; a gate cannot pass while it depends on an unresolved §14 item. Empirical questions are resolved in the phase that can actually measure them—Phase −1 must not fabricate answers to Phase 5 experiments.

### Phase −1 — Source/Oracle Truth Pack (FIRST; no kernel work until green)
- Pin + SHA-256 every source: HF repo revision (config, both safetensors, index, tokenizer files, modeling/configuration source, template/generation config, report PDF, card/API metadata); record the **absence** of LICENSE/NOTICE files as evidence rather than inventing them; pin the `nanbeige42` fork and any community GGUF used only as a cross-check. Commit separate conversion-source and research-evidence manifests; make `scripts/fetch_model.sh --check-only` and `scripts/fetch_model.ps1 -CheckOnly` reproduce every conversion-source length/digest from revision-scoped clean directories.
- Generate the **machine-readable census** (§2.6): every tensor name/shape/byte + KV formulas + score-bucket single-token verification (§7.5) + context/buffer tables; CI-guarded.
- Promote the source-answerable §14 observations needed by Phases 1–2 into line-backed **[EVIDENCED]** records; leave empirical OQ-11–15 and OQ-17–24 open behind their own later gates.
- Stand up the CPU oracle; prove it runs; measure its nondeterminism floor; capture smoke fixtures (greedy streams, per-(loop,layer) dumps, template renderings, tokenizer corpora).
- **Exit:** OQ-1…9 and OQ-16 are evidenced to the extent required for tokenizer/f32 forward; OQ-14's oracle floor is measured; census green; fixtures committed; license state explicitly blocked/cleared; claim→source-line index replay passes.

### Phase 0 — Scaffold
- The template repo shape (§4.1): crate, two shim bins, immutable git+rev suite deps with checked `Cargo.lock`/`SUITE.lock` agreement, deny-by-default unsafe policy, CI (`check.sh`: validators, fmt, locked check/clippy/test, bounded UBS), process-shared runtime/kernel resources with engine leases, robot skeleton/schema, optional metadata-only fsqlite store, error/resource map, model-gated harness, ledgers, and fixture scripts. Governance: public `main` repo, project source license copied exactly from the approved sibling template **without falsely relicensing third-party weights**, `.gitignore`, CHANGELOG, and checked suite-pin manifest.
- **Exit:** builds on all 5 targets; `robot schema|health|backends` emit valid versioned JSON; empty-pipeline e2e skips-green without weights.

### Phase 1 — Tokenizer + template + f32 parity (correctness before speed)
- SentencePiece tokenizer (OQ-6 spec-first) → **L0 exact**; template builder over the OQ-7 matrix → L0 exact. Specify the separate `UntrustedDocument` encoder: exact decoded bytes with every role/think/tool-control id forbidden, or typed rejection; fuzz the full control-id set.
- Pure-f32 forward from the extracted spec: embed → 22 layers → final RMSNorm → same 22 layers → final RMSNorm → lm_head → greedy, with `layer+loop×22` KV.
- **Exit:** L0–L4 green in f32 (44 layer outputs + two post-loop norms; greedy exact over oracle-reproducible prefixes); determinism gate green; census loader green.

### Phase 2 — int8 + `.fnlpq` + distribution
- Converter + format + census loader; staged quant 2a/2b/2c each with its own parity gate; deterministic Generic artifact; fixed-size release packager + asset receipt/reconstruction runbook; release-bound embedded manifest; resumable streamed `fnlp pull`; content-addressed native-pack derivation; `fnlp models`; `robot selftest`.
- **Exit:** int8 L3/L4 within measured tolerance (argmax agreement ledgered, L4 exact only where the quantized path actually preserves it); two-clean-directory conversion identity; package/reassemble identity; malicious-format/manifest/pull/resume/concurrency/install tests green; a private/fake-release clean-cache pull resolves with no `--model`. Publish public weight assets **only if LG-1 is cleared and §5.6's remote replay + real-inference receipt is green**; otherwise ship local conversion + private/pinned-manifest support and record the publication blocker.

### Phase 3 — SIMD kernels + latency perf
- Dispatch catalog (§6.3), both exact AVX2 candidates, sustained 256/512-bit VNNI comparisons, fixed-shape GEMM/GEMV, native GQA/decode attention, then profile-gated fusion/NUMA/build recipes. Gauntlet R1 begins.
- **Exit:** every integer SIMD kernel equals scalar/i64, floating approximations meet their named tolerance and stay distinct; R1 is measured fairly on accessible M4/M5 and Zen 3/4/5 hosts (no hardware, no performance claim); every winner/loser ledgered.

### Phase 4 — Batch engine + task layer v1
- `batchsched` + prefix cache + batch daemon (invariance gates); opt-in durable jobs; grammar/source execution compiler (`FullProjection`, all-legal sparse projection, forced-run fallback); exact continuation trie; `schema check/sample`; tasks: extract, ner, classify, sentiment (both modes), redact/verify, generate/chat, tokens/split; presets v1; task evals (§9.6) stood up. Typed prompt-segment/trust ABI lands, but document-major sharing remains per-task empirical; matched content-injection suites measure residual steering.
- **Exit:** G8 and grounded-field property/fuzz proofs green; untrusted segments exclude control ids or reject; sparse/full, forced/sequential, all trie traversal/naïve, uninterrupted/resumed, and dependency-scope mutation invariants green; scheduler automaton/interleaving, batch, prefix/fork-tail capacity, privacy, and cancellation gates green; per-task injection attack-success rows are published without a firewall claim; R2 gauntlet recorded; task scorecards published.

### Phase 5 — int4 + full portfolio + calibration
- int4 uniform baselines, then AA-Q1/AA-T1 if their EV gates survive; remaining tasks; AA-U1 calibration; user `eval/calibrate/qualify`; offline static decision policies/review spill; experiment with `extract --verify-semantic` against per-domain locked labels; open the bounded public recipe compiler only after built-in equivalence and security gates; measured thinking defaults; AA-C1 only if cache traces justify it; public artifacts remain LG-1-conditional.
- **Exit:** complete int4 candidates meet mean/tail/footprint gates on locked data; full portfolio surfaced and evaluated; calibration validity/coverage measured and scoped; semantic verification is either promoted per task with incremental-error/cost evidence or remains off; qualification receipts and stale-digest rejection proven; public TaskIR recipes either pass all gates or remain internal.

### Phase 6 — Hardening + cross-platform release
- CLI/robot contract freeze; digest-gated activate/rollback; optional `fnlp tune` restricted to proved bit-identical choices; doctor; installer (`install.sh`/`.ps1` per the sibling pattern) that invokes the installed binary's one artifact manager; interactive/`--with-model`/`--no-pull` policy; fresh-HOME/LOCALAPPDATA fake-release E2E; 5-target dist matrix; the full three-pillar gauntlet to convergence; agent-ergonomics audit; README/docs truth pass (`/reality-check-for-project` + `/de-slopify`).
- **Exit:** all targets build+smoke; scorecard all-green; installers verify exact binary versions and the model handoff on Unix/Windows; LG-1-blocked installers truthfully omit the public-model offer; an authorized release proves installer → pull → no-flag discovery → real inference → offline second run; release certified.

### Phase 7 — Stretch (explicitly optional, in EV order)
- Per-task-qualified document-major `analyze` packs · `schema infer` holdout experiment · AA-A1 human-graded frozen-job acceptance/qualification-decay audits · scope-correct portable snapshot `partition/merge` + entity lineage · AA-R1 local resident only if multi-process traces justify it · AA-W1 only if occupancy traces prove fragmentation · `ft-kernel-metal` prefill experiment (CPU stays product/reference) · remote/routable `fnlp serve` only as a separate later decision · AA-D1 exact loop-draft only if its research gate survives · translation after multilingual evals · int8-embedding/KV-int4 refinements.

---

## 12. Risks & mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Observed loop/minimal-surface semantics are transcribed incorrectly** — plausible text, silently wrong model | **HIGH** | Phase −1 promotes pinned lines/index to evidence; 44 layer outputs + two loop-norm fixtures; three-oracle triangulation; no implementation based only on this prose |
| **Upstream checkpoint/source drift** | HIGH | immutable pins, per-file digests, machine census, fail on extra/missing tensors; suite/source upgrades are explicit plan revisions |
| **Tokenizer/template drift** (SP subtleties, thinking/tool markup) | MED-HIGH | L0-exact gates over adversarial corpora (multilingual/code/specials); fast-vs-slow reconciliation (OQ-6); template mode-matrix fixtures (OQ-7) |
| **Quant sensitivity/interactions unknown** | MED-HIGH | staged int8; uniform int4 baselines; AA-Q1 joint complete-artifact evaluation; AA-T1 tail + mean gates; kill switches; no separability/optimality fiction |
| **Constrained decode correctness** (mask bugs → invalid JSON or distorted outputs) | MED | property-fuzz (never-invalid, never-dead-state); L5 constrained-greedy exactness; independent-parser verification in CI |
| **Execution compiler silently changes semantics** (sparse rows, forced runs, trie sharing, source product) | HIGH | complete-language/no-pruning rule; sparse=full, forced=sequential KV, trie=naïve, source-membership differentials; universal fallbacks; each lever benchmarked separately |
| **Batch/prefix/KV paging breaks parity, privacy, or memory bounds** | HIGH | exact invariants, namespace-complete cache keys, lazy-page ownership model, checked admission, automaton/interleaving replay, sequential/cold fallback |
| **Durable job overclaims exactly-once or retains sensitive data** | HIGH | exactly-once scoped to owned spool/materialization; raw stdout explicitly at-least-once; metadata-only default; opt-in owner-only spools; kill/disk-full/torn-tail proof |
| **AVX2 intra-pair saturation or offset-correction error** | HIGH | X3a mathematical pair bounds + X3b independent decomposition; full-domain/i64 tests; raw saturating shortcut forbidden |
| **AVX-512 is available but slower/thermally unstable** | MED | sustained zmm/ymm/AVX2 A/B per host and shape; frequency/energy/tail ledger; narrower measured fallback |
| **Perf parity with a years-tuned llama.cpp fork is hard** | MED | honest bar per §1.1 G2 (decode + batch-throughput at matched quant, per-regime ledger); our structural edges (prefix-cache, logit-slicing, batch layer-major, task fusion) are levers the generic fork lacks; slower rows stay published |
| **Deadlock/oversubscription regressions** | MED | the architectural fix (one pool, sequential-or-batched, never nested) + `many_docs_without_deadlock` watchdog |
| **3B task-quality overclaim** | HIGH | disjoint development/calibration/test; named scorecards; non-goals enforced; AA-U1 scoped abstention; a task may remain experimental or be cut |
| **Recipe/controller scope becomes a generic framework or confidence theater** | MED-HIGH | bounded data-only TaskIR; no code/network/tools; static policy first; task-valid calibrated signals only; public surface may stay closed; document-major/infer/adaptation remain empirical stretch |
| **Incremental cache returns globally stale corpus results** | HIGH | TaskIR stage scope is mandatory; snapshot/child-set digests; local maps may reuse but reduce/global authority reruns; resolve cluster ids are snapshot-qualified; mutation invariant |
| **Continuation-trie forks explode 44-deep KV memory** | HIGH | admission uses actual page-fill/fan-out bytes; sealed pages + small tail slabs/recompute; breadth/depth-first/naïve/short-ID alternatives; exact-score equality |
| **Resident mode expands into an insecure server** | MED | AA-R1 traces/EV first; owner-only local IPC, no routable listener/credentials, namespace caps, in-process/batch fallback; omit if approved cross-platform surface is unavailable |
| **Control-token containment is mistaken for prompt-injection immunity** | HIGH | typed untrusted encoder proves only forbidden-id exclusion/byte preservation; matched adversarial-content scorecards measure steering; outputs remain untrusted and no firewall claim ships |
| **Same-model verification launders correlated agreement into proof** | HIGH | `verify-semantic` off by default; report four-state result and same-model provenance; promote only on incremental locked-eval value with false-alarm/cost rows |
| **Acceptance sampling overclaims a corpus certificate** | MED-HIGH | AA-A1 human authority, frozen population/design/seed, independently checked finite-population math, protected explicit review pack, scoped error-rate claim or no-claim |
| **Weights/tokenizer/preset redistribution authority missing** | **HIGH / PUBLICATION BLOCKER** | §5.7 fail-closed provenance; no public derivative weights or embedded tokenizer until LG-1; no swiss_army_llama copying without recorded authority |
| **Release chunks or installer disagree with the runtime catalog** | HIGH | one release-bound embedded manifest; binary/model versions independent; installer delegates to installed `fnlp pull`; exact inventory/part/whole/source/license hashes; remote clean-cache replay; never clobber immutable assets |
| **Untrusted input exhausts CPU/RAM or parser state** | HIGH | §8.6 caps before allocation, lazy KV, grammar preflight, deadlines/fair morsels, malicious corpora |
| **Scope creep** (model zoo, server, GPU, POS tagging…) | MED | §1.2 non-goals enforced at review; Phase 7 quarantines stretch; one model, one product |

---

## 13. Success metrics

**Correctness (G1/G8, gating):** L0 exact; all 44 layer outputs + two loop-final-norm states meet per-op measured error vectors; L4 greedy exact on the f32 oracle-reproducible prefixes; quantized argmax/task deltas stay within preregistered budgets; every successful structured result independently validates and every declared grounded field independently occurs in source; untrusted segment encoding byte-preserves or rejects while excluding all control ids; sparse/full, forced/sequential, trie/naïve, determinism/batch/prefix/scheduler/capacity gates green.

**Performance (G2 target):** R1 decode/token and R2 docs/min meet or beat the pinned fork at matched quant on each host for which the project makes a claim; p50/p95/p99, energy where available, memory peak, prompt/output lengths, and losing rows are published. No available M4/M5/Zen host means “not measured,” never extrapolated certification.

**Footprint & portability (G3/G4):** converter/loader report exact per-section artifact bytes and peak/RSS/KV admission; no speculative 20 MB binary or 2.4 GB artifact number becomes a promise before builds. Five target binaries build/smoke; inference opens no network and needs no foreign ML runtime/GPU. Crate roots use `deny`, audited islands are enumerated, and every island has scalar differential + policy-scan evidence.

**Product (G5/G6/G7):** tasks graduate individually only with scorecards; batch daemon sustains bounded corpus workloads under soak/cancellation; owned durable jobs survive kill/disk-full and emit verified receipts; user qualification is digest-scoped; robot contract is frozen/self-describing; every divergence/rejection is ledgered. The sovereign path proves pinned source download → deterministic conversion → native packing → inference. When LG-1 permits it, the release path additionally proves deterministic split → remote inventory/re-download → fresh-machine installer → `fnlp pull` → no-flag discovery → identical inference, with a byte-perfect cache hit and no inference network on the second run.

---

## 14. Research-decision register

`[OBSERVED@pin]` rows have answers from this review but still block implementation until Phase −1 archives line-backed evidence. `[PARTIAL]` and `[OPEN]` rows retain only their unresolved portion. Empirical rows are not source-reading chores.

| ID | Current answer / remaining question | State | Blocks / evidence authority |
|----|-------------------------------------|-------|-----------------------------|
| **OQ-1** | Pinned index has 201 tensors and only embed, 22×(2 norms+7 matrices), final norm, lm_head; no mHC/ngram/depth/LoopSplit parameters. | **OBSERVED@pin** | Phase 1 until index census is committed |
| **OQ-2** | KV index is `layer + loop×22`; same logical positions/cache positions, 44 independent slots. Confirm fork agrees and fixture prefill/decode append semantics. | **PARTIAL** | loop/KV; modeling lines + fork + trace |
| **OQ-3** | Each 22-layer pass is followed by final RMSNorm; loop 2 consumes loop-1 normalized state; no reinjection/projection. | **OBSERVED@pin** | forward/L2 until line-backed fixture |
| **OQ-4** | RoPE is split-half rotation; f32 frequency matmul/cos/sin, duplicated frequencies, cast to q/k dtype; θ=7e7. Pin exact position/index behavior. | **PARTIAL** | RoPE L1 |
| **OQ-5** | attention scale `1/√head_dim`, additive mask, f32 softmax then cast, zero inference dropout; no observed softcap. Pin mask shapes/GQA repeat behavior. | **PARTIAL** | attention |
| **OQ-6** | SentencePiece BPE, identity normalizer, byte pieces, BOS true/EOS false, legacy false. Resolve byte-fallback flag semantics, added-token precedence, fast-vs-slow corpus equality, whitespace/invalid-byte decode. | **PARTIAL** | tokenizer L0 |
| **OQ-7** | `<|im_start|>/<|im_end|>`, think tags, thinking/preserve and XML/JSON tool branches observed. Archive exact bytes for every role/mode/generation suffix. | **PARTIAL** | template L0/tasks |
| **OQ-8** | Determine effective HF `generate()` processors/defaults beyond the small generation config and define which are parity surface vs explicit fnlp options. | **OPEN** | sampler L4 |
| **OQ-9** | Template delimiters are observed; resolve malformed/missing think close, EOS, preserve/strip rules and bounded task behavior against reference/fork parsers. | **PARTIAL** | structured tasks/chat |
| **OQ-10** | Pin exact Nanbeige tool-call parser conventions and malformed-call behavior; `fnlp` never executes tools. | **OPEN** | `generate --tools` only |
| **OQ-11** | Design/census the opaque single-token bucket set, `UNSURE`, multi-token fallback recipe, and full-vocab mass audit. | **OPEN / empirical** | sentiment distribution |
| **OQ-12** | Audit pinned fork GGUF tensor mapping and quant tiers as an independent census/prior—not as authority for our recipe. | **OPEN** | quant prior, not f32 forward |
| **OQ-13** | Measure joint loop-aware quant sensitivity on complete artifacts; one-tensor curves are only search hints. | **OPEN / Phase 2–5 empirical** | AA-Q1/int4 |
| **OQ-14** | Measure HF CPU nondeterminism across repeated runs/thread counts/builds before L3/L4 thresholds. | **OPEN / Phase −1 empirical** | all numeric tolerances |
| **OQ-15** | Config/plain RoPE advertises 262144; verify oracle/fork behavior and publish time/KV/RSS practicality. Never conflate acceptance with usability. | **PARTIAL / R4 empirical** | >8K claims |
| **OQ-16** | BOS=true, EOS=false and card path uses templated text with `add_special_tokens=false`; archive pad/BOS/EOS matrices and off-by-one fixtures. | **PARTIAL** | L0/L4 |
| **OQ-17** | Specify and prove the exact `verbatim` source-language product: start/finish/empty transitions, JSON unescaping, byte-fallback/UTF-8, repeated-occurrence offsets, caps, and unsatisfiable-field semantics. `verbatim-normalized` remains out. | **OPEN / Phase 4 design+proof** | grounded extraction/NER defaults |
| **OQ-18** | For each task, compare prompt-segment layouts on locked quality and exact cold/fork behavior; only individually eligible tasks may enter document-major packs. | **OPEN / Phase 4–7 empirical** | `analyze` sharing, not ordinary tasks |
| **OQ-19** | Measure legal-set/forced-run/trie distributions and exact crossover thresholds; prove sparse=full, forced=sequential, trie=naïve before enabling each optimized primitive. | **OPEN / Phase 4 empirical** | execution-compiler optimized routes |
| **OQ-20** | Census every role/think/tool-control id and specify a forbidden-id `UntrustedDocument` tokenization that decodes to the original bytes or rejects; separately measure per-task content-steering attack success on matched clean/attack fixtures. | **OPEN / Phase 1+4** | typed document boundary and consequential task presets |
| **OQ-21** | On each extraction domain, measure whether same-model faithfulness verification catches additional semantic-field errors after false alarms, correlated misses, latency, and prefix eligibility are counted. | **OPEN / Phase 5 empirical** | `extract --verify-semantic` default/policy use |
| **OQ-22** | Specify AA-A1's exact finite-population/weighted-stratified estimator, known inclusion probabilities, post-freeze seed authority/non-grinding evidence, longitudinal invalidation rule, grader authority, missing-grade behavior, privacy surface, and independently checked golden cases before exposing `audit grade`. | **OPEN / Phase 7 research** | any acceptance/continuing-qualification audit claim |
| **OQ-23** | Specify TaskIR stage-scope propagation, snapshot/child-set digests, mutation invalidation, portable partition compatibility classes, merge refusal, and cross-snapshot entity-lineage semantics. | **OPEN / Phase 4 design; Phase 7 distribution** | incremental cache authority; partition/merge/lineage |
| **OQ-24** | Measure duplicate model loads/core pools across independent local clients; only if AA-R1 clears EV, specify owner-only Unix/Windows local IPC, identity/caps/cancellation, lifecycle, protocol negotiation, and namespace isolation. | **OPEN / Phase 7 research** | `fnlp resident` |
| **LG-1** | Card metadata says Apache-2.0; pinned repo supplies no LICENSE/NOTICE. Obtain immutable license materials or rights-holder confirmation and owner review. | **BLOCKED** | public weights, embedded tokenizer/presets |

---

## 15. Skills, methodology & the path to beads

### 15.1 The named skills and where each governs

| Skill | Role | Sections |
|-------|------|----------|
| `asupersync-mega-skill` | sync API over one process-shared resource set, budgets, cancellation, no nesting/re-entrant block | §3.3, §8.4 |
| `/porting-to-rust` | spec-first: `EXISTING_NANBEIGE_STRUCTURE.md` before code; resolves §14 | §9, §11 P1 |
| `/running-the-gauntlet-on-your-rust-port` | three-pillar certification and adversarial convergence; this revision corrects its statistical metaphors where deterministic proof is required | §9.5, §11 P6 |
| `/testing-conformance-harnesses` + `/testing-golden-artifacts` + `/testing-metamorphic` + `/testing-fuzzing` | the L0–L5 ladder + golden/metamorphic/fuzz mechanics | §9.2–§9.3 |
| `/profiling-software-performance` + `/extreme-software-optimization` | profile-first + the keep/revert loop + per-regime ledgers | §10.1–§10.2 |
| `$alien-graveyard` + `$alien-artifact-coding` | AA cards, composition budget, evidence artifacts, negative-results graveyard, deterministic fallbacks | §10.5–§10.6 |
| `/extreme-software-optimization` LLM guidance + `focr` skill | the sibling's measured lessons (dispatch, autovec-vs-SDOT, ledger formats) | §3.2, §6 |
| `/agent-ergonomics-…-cli-tools` + `/world-class-doctor-mode…` | the robot surface + doctor (Phase 6) | §8 |
| `/cross-project-pattern-extraction` + `/installer-workmanship` + `/release-preparations` + `/gh-actions` | FrankenOCR-derived source/package/pull/installer invariants and the Phase 6 release train | §5.1, §5.6, §11 P2/P6 |
| `/beads-br` + `/beads-workflow` + `bv --robot-*` | the work graph (below) | §15.2 |
| `/cass` | mine the franken_ocr/frankensearch kernel sessions before each perf bead | §10 |

### 15.2 The path to beads (after plan review)

Per the planning workflow: this plan first survives **≥ 4 external review rounds** (GPT Pro Extended Reasoning per the standing prompt; multi-model blend optional) to steady-state, then `/beads-br` + `/beads-workflow` convert it:

- **Epics = §11 phases** + a spec-extraction epic + a gauntlet epic.
- **Every unresolved/partial §14 item becomes a research bead blocking only its dependent surface; observed rows become truth-pack evidence beads. LG-1 is a publication blocker, not a kernel blocker.**
- **Every kernel/subsystem gets test + bench + doc dependencies**; every AA card starts as a spike carrying hotspot evidence, EV, proof obligation, interaction check, and fallback.
- Polish beads 4–6 rounds in plan-space (never oversimplify, never lose features); `bv --robot-insights | jq '.Cycles'` must be empty before implementation.

---

## 16. Primary-source index

These links are discovery aids; the truth pack must fetch immutable bytes, record SHA-256 and source spans, and remain the implementation authority.

- Nanbeige checkpoint at the inspected revision: [tree](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/f56ec5a9650268aa098496734743c25ea778bd2d), [config](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/f56ec5a9650268aa098496734743c25ea778bd2d/config.json), [modeling source](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/f56ec5a9650268aa098496734743c25ea778bd2d/modeling_nanbeige.py), [tensor index](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/f56ec5a9650268aa098496734743c25ea778bd2d/model.safetensors.index.json), and [tokenizer config/template](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/f56ec5a9650268aa098496734743c25ea778bd2d/tokenizer_config.json).
- Authors' llama.cpp fork: [inspected commit](https://github.com/Nanbeige/llama.cpp/commit/c6640a1c0cf7b38df342b67021a3900b04d092e7).
- Foundation commits: [frankentorch](https://github.com/Dicklesworthstone/frankentorch/commit/523aaf827faf538aa541126ee222fcd7af348410), [asupersync](https://github.com/Dicklesworthstone/asupersync/commit/8eb48575889c81b65f7556db4b26d47a8bc03197), and [frankensqlite](https://github.com/Dicklesworthstone/frankensqlite/commit/5676cb97486a62c4f0a19c053184e0ff3cfb2852).
- Sentiment lineage candidate: [swiss_army_llama at the inspected commit](https://github.com/Dicklesworthstone/swiss_army_llama/blob/7bd155410ff2cdf71b4ddf4ccd5a626a600690b3/sentiment_score_generation.py); behavior may inform a clean implementation, but no code/preset copying until §5.7 provenance clears.
- ISA/platform references: [Apple Silicon CPU Optimization Guide](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide), [Arm FEAT_I8MM toolchain table](https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain), [AMD Zen 4 AVX-512 architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/products/epyc/4th-gen-amd-epyc-processor-architecture-whitepaper.pdf), [AMD Zen 5 Threadripper 9000 full-width AVX-512 announcement](https://www.amd.com/en/blogs/2025/designed-to-create-built-to-inspire-amd-introduces-new.html), [AMD runtime-dispatch/BIOS guidance](https://docs.amd.com/r/en-US/57404-AOCL-user-guide/12.4.1.-Dynamic-Dispatch), and [Intel VNNI `vpdpbusd` overview](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html).
- Distribution/language rules: [GitHub release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) and [Rust lint-level semantics](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes).

---

*End of plan. Living document: resolve §14 as sources are read; append every accepted divergence to `DISCREPANCIES.md`, every rejected lever to `NEGATIVE_EVIDENCE.md`; re-state the parity receipt on every perf commit. The loop-aware kernels (§6) and the task layer (§7) are the product; the three-pillar gauntlet (§9.5) is the conscience.*
