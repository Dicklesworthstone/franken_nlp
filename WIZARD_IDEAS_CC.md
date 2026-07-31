# WIZARD_IDEAS_CC.md — Top-8 Improvement Ideas for franken_nlp

**Status:** idea-wizard output, candidate amendments/additions to `COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md` — subject to the plan's own review cycle (§15.2) before any of it becomes doctrine.
**Method:** 30 candidates were generated across five axes (new capability, performance, robustness, developer experience, trust), each thought through for mechanism, user perception, implementation cost, and doctrine fit, then winnowed to the 8 below. The 22 cut/absorbed candidates are in the appendix with verdicts, so the winnowing is auditable.
**Ordering:** Ideas are ranked **best (1) to worst (8)**.

> **Post-review disposition (2026-07-30):** preserved as ideation provenance, not specification. Plan v3 §10.6 controls and corrects several claims below. “Constraint pressure” is a diagnostic, not a fabrication detector or calibrated confidence; best-legal/best-illegal margin and legal mass are unavailable without full projection; no e-process replaces a deterministic invariant; instructions-last is a per-task quality experiment; source normalization is outside v1; raw floating-logit hashes are not a portable selftest; automatic think/self-consistency escalation is deferred behind static, calibrated, user-owned policies; and comparative novelty/magnitude statements below are hypotheses rather than project claims.

**Winnowing criteria** (applied in this order):

1. **Deepens the core differentiator instead of widening scope.** The moat is valid-by-construction output + the loop-aware batch fabric + enforced honesty. Ideas that make those *more* true beat ideas that add surface area.
2. **Leverage.** Prefer ideas that are mostly composition of machinery the plan already commits to building (Stencil, Conveyor, AF-4 calibration, fsqlite run store, the §9.6 eval harness).
3. **Honest measurability.** Every claimed benefit must be measurable with the project's own instruments; no idea whose value can only be asserted.
4. **Time-to-perceived-value.** The user should feel the idea in the first hour (DX) or the first overnight run (ops).
5. **Doctrine compatibility.** Closed dependency universe, no network at inference, determinism contracts, deterministic fallbacks for anything adaptive, ledger entries for anything numeric.

---

## Idea 1 — Grounded-by-construction decoding (copy-constrained fields)

**Type:** AMENDMENT — §7.3 (Stencil), §7.1/§7.2/§7.7 (task contracts), §9.3/§9.5 (new invariant) · **Phase:** 4 · **Confidence: high**

### What it is

G8 guarantees *syntax*: the JSON always parses and matches the schema. The residual failure mode — the one users actually fear — is *semantic*: the model can fill a perfectly valid schema with a fabricated invoice number, a misquoted citation, an entity surface form that isn't in the text. For extraction-shaped fields the structural fix is available to us and to almost nobody else: **constrain string-value states so they can only emit byte-exact substrings of the source document**, by running the grammar automaton in product with a suffix automaton built over the document's bytes. "Valid by construction" becomes **"grounded by construction"**: a hallucinated value is not detected and flagged — it is *unrepresentable*, exactly the way invalid JSON already is.

### Why it makes the project obviously better

- It attacks the **single biggest trust gap** of LLM extraction with the same class of mechanism that is already the project's flagship differentiator. The pitch sentence writes itself: *"the extracted value cannot be invalid, and for grounded fields it cannot be invented."*
- It **collapses the anchoring machinery**. `ner` surface forms, `redact` spans, `answer` citations, `keyphrases`, and `extract --evidence` quotes all currently emit text that must be re-anchored post-hoc, with `anchor_status ∈ {exact, relocated, unanchored}` covering the failure cases. Copy-constrained decoding makes `exact` **guaranteed by construction** for those fields — the `relocated`/`unanchored` classes vanish wherever the constraint is on, and offsets stop being a heuristic search problem.
- No local-LLM tool ships this end-to-end. Lexically-constrained and pointer-style decoding exist in the research literature, so we claim engineering-first, not concept-first — the productized, byte-exact, SentencePiece-transducer-composed version with offset guarantees is ours.

### How it works

- Stencil already consumes **detokenized bytes** through the SP transducer (§7.3(f)) precisely because raw piece text is unsound. That means a byte-level intersection is *architecturally native* — there is no new tokenization-alignment problem to solve.
- At task-plan time, build a **suffix automaton** (DAWG) over the normalized document bytes: linear time and space, a 40-year-old textbook construction, dependency-free. For a typical 8K-token document (~30 KB) this is sub-millisecond and sub-MB; even a 256K-context document (~1 MB) costs ~tens of MB, priced and printed like the KV table (and huge docs already route through map-reduce, §7.10).
- In a copy-constrained string state, the legal next-byte set is (grammar-legal bytes) ∩ (suffix-automaton transitions from the current match state). The mask oracle computes token masks by the same lazy trie walk. Note the cost asymmetry: copy regions admit **few** legal continuations, so lazy masks in copy mode are *cheaper* than in free-string mode; the globally-memoized scaffolding-state masks are untouched (copy-region memos are per-document by nature, which is fine — they're small).
- **Escaping is layered correctly:** the grammar's string lexer owns the JSON encoding (`\"`, `\n`, `\uXXXX`); the suffix automaton consumes the *unescaped value bytes*. The string state machine exposes the logical value stream — clean separation, already implied by the §7.3 design.
- **Surface:** a schema annotation (`"x-fnlp-source": "verbatim" | "verbatim-normalized"`) plus `--ground <fields>` on the CLI. Defaults: ON for `ner` surface forms, `redact` spans, `answer` citations, `keyphrases` (their contracts already promise exact anchoring); OFF per-field in `extract` (values the model legitimately normalizes — dates, amounts — stay unconstrained and honestly labeled).
- **Offsets for free:** the emitted value is a substring by construction, so span resolution reduces to occurrence selection (existing nearest-to-hint convention; all-occurrences mode available) — never "not found."

### Failure modes, proof obligations, fallbacks (doctrine fit)

- A copy constraint can be **unsatisfiable at field entry** (enum/minLength intersects an unlucky document to an empty language). Defined behavior, never silent: emit `null` if the field is nullable, else a typed per-field error with `grounding_status: "unsatisfiable"` — and never a silent fallback to unconstrained decoding.
- New **e-process invariant** (§9.5 list): every copy-constrained emitted value is a byte-exact substring of the normalized source, checked by an independent verifier under fuzz (adversarial unicode, escapes, repeated substrings).
- Constraint-pressure telemetry (Idea 3) detects **forced copying** — the model wanted off-document content but the automaton dragged it along a substring — so the guarantee cannot silently degrade output quality without leaving a measurable trace.
- Determinism, batch-invariance, and greedy-under-mask are untouched (per-sequence state, same mask-AND mechanism).

### Why ranked #1

Highest product-differentiation per unit of engineering risk in the whole candidate set. The mechanism is classical, linear-time, and slots into the exact byte-level seam Stencil already committed to; the failure modes are enumerable and get defined semantics; and it upgrades the project's central promise rather than adding a side feature. Confidence is high because every hard sub-problem (byte alignment, lazy masks, liveness, anchoring) is either already solved by the plan's design or made *easier* by the constraint.

---

## Idea 2 — AF-6: the calibrated escalation ladder

**Type:** NEW alien-artifact family + task-layer policy — amends §7.8/§7.9/§8.4/§10.5 · **Phase:** 5 (think-retry tier available in Phase 4) · **Confidence: high (mechanism), medium-high (magnitude)**

### What it is

The plan prices the 3B ceiling honestly ("frontier models will beat it on raw accuracy") and then leaves the user alone with it. Systematize the response instead: a per-task, calibrated, **measured escalation policy** over compute tiers that all already exist in the design:

```
fast path (no-think)  →  thinking retry  →  self-consistency-N  →  spill / abstain
```

- **Tier 1:** the throughput default (§7.9).
- **Tier 2:** re-run the item with `--think` — the plan already builds the toggle, the per-task measured think-delta artifact, and (with Idea 7) a doc-KV fork that makes the retry pay only the instruction tail.
- **Tier 3:** N seeded samples + **field-level majority vote**, with cross-sample agreement as a calibration-free confidence signal. This is uniquely well-defined here: G8 means every sample parses, so field-wise voting never hits a parse failure; seeded RNG makes it reproducible; the batch fabric makes N-way sampling cheap.
- **Tier 4:** `--spill low_conf.ndjson` writes the full envelope (input ref, output, confidences, pressure telemetry) for downstream handling — a bigger model, a human queue. **fnlp never phones an external model itself** (the no-network doctrine holds); the spill file *is* the escalation interface.

### Why it makes the project obviously better

- It converts the strongest objection to the whole project ("a 3B model will be wrong sometimes") into its smartest behavior: **median cost ≈ fast path; tail accuracy ≈ the best this system can do; the boundary chosen by calibrated measurement, not vibes.** That is a categorically better story than "usable accuracy" alone.
- The published artifact — a per-task **escalation curve** (accuracy vs mean cost, with the operating points marked) — is a marketing-grade honesty instrument no competing tool has, and it's generated by machinery (§9.6 evals + AF-4 calibration) the plan builds anyway.
- For pipeline users, `--spill` makes fnlp the perfect *triage layer*: 90–95% of a corpus handled locally at zero marginal cost, the measured-hard residue explicitly handed off. That is how a 3B appliance coexists with frontier APIs instead of competing with them.

### How it works (the AF contract, satisfied)

- **State space:** per-item signal vector — AF-4 calibrated confidence, captured mass, constraint pressure (Idea 3), per-field agreement (tier 3).
- **Actions:** {accept, think-retry, consistency-N, spill, abstain}.
- **Loss matrix:** from `--quality fast|balanced|max` presets (user-configurable cost of a wrong answer vs an escalation), documented per task.
- **Calibration:** thresholds fit on the §9.6 eval sets at release; refittable on the *user's own data* via Idea 6.
- **Deterministic fallback:** ladder off (always accept tier-1) — the shipped default until the measured curves justify enabling it per task.
- **Evidence ledger:** per-run escalation receipts (per-tier counts, cost delta) in the envelope and the Idea-5 run receipt.
- **Batch integration:** escalated items re-enter the daemon queue with amended task args — the per-doc `task_args` override surface already exists (§8.4).

### Risks

Magnitude is the honest unknown: if thinking or consistency buys little on a given task, the ladder stays off for that task *by evidence* — which is itself the §7.9 discipline, generalized. The design cannot lose; only individual tiers can.

### Why ranked #2

Biggest strategic reframe available for near-zero novel machinery — it is composition, policy, and measurement over parts the plan already commits to. It ranks below Idea 1 only because Idea 1 hardens the core guarantee while this one optimizes above it.

---

## Idea 3 — Constraint-pressure telemetry (the fabrication detector)

**Type:** AMENDMENT — §7.3/§7.8 envelope + robot schema; new §9.5 monitor · **Phase:** 4 (with Stencil) · **Confidence: very high**

### What it is

Constrained decoding has a unique silent failure the plan does not yet instrument: **the grammar can launder model uncertainty into confident-looking, schema-valid output.** When the model has no idea what the invoice total is, the mask still forces *some* digit sequence out, and it arrives dressed exactly like a good answer. Measure the laundering, per step and per field:

- **Override indicator** — was the pre-mask argmax token grammar-illegal at this step?
- **Legal mass** — pre-renormalization probability the model put on the legal set (this generalizes §7.5's captured-mass diagnostic, currently unique to sentiment distribution mode, to *every* constrained decode step).
- **Logit margin** — max legal logit − max illegal logit (a zero-extra-cost proxy when the softmax denominator isn't otherwise needed).

Aggregate per field — the automaton always knows which schema field it is inside — into a `constraint_pressure` block in the response envelope, with per-preset thresholds that set a `fabrication_risk` flag.

### Why it makes the project obviously better

- **It makes G8 trustworthy instead of merely true.** "Schema-valid 100% of the time" is the headline; "…and we tell you, per field, when validity was doing the work instead of the model" is the sentence that makes a skeptical engineer adopt it.
- It is the **routing signal** the rest of this list runs on: Idea 2's ladder escalates on pressure; Idea 1's forced-copy case is detected by it; `--spill` selects on it. Build it first and three other ideas get their nervous system.
- Release-to-release **fabrication-risk rate** on the golden corpora becomes an e-process-monitored quality metric — a genuinely novel gate no other constrained-decoding implementation publishes.

### How it works / cost analysis

- The pre-mask argmax is already computed (the fused GEMV+argmax path, §6.11); the override indicator is a comparison.
- Margin mode costs zero extra vocab passes. Mass mode needs the full softmax denominator — which is *already* computed whenever logprob confidences (§7.8) are on; the incremental cost is an exp-sum over 166,144 floats (~1–2% of a decode step, and decode steps are memory-bound anyway). Ship margin-only as the default, mass on demand.
- Envelope and `robot schema` fields versioned from the first release so downstream contracts never break when thresholds evolve.

### Proof obligations

Fixture suite where pressure *must* spike: adversarial schemas over off-topic documents (extract a flight number from a cookie recipe), enum sets that exclude the true answer, copy-constraints over documents lacking the value. Plus the inverse: on-topic fixtures where pressure must stay low. Both directions gate.

### Why ranked #3

The highest utility-per-line-of-code in the entire candidate set — a handful of comparisons in the sampler and an envelope block, closing the honesty doctrine's one genuine blind spot. It ranks below Ideas 1–2 only on absolute scope: it is an instrument, not a capability. Build order note: this lands *before* Idea 2 (the ladder consumes its signals).

---

## Idea 4 — The schema toolchain: `fnlp schema infer | check | sample`

**Type:** NEW capability — Atelier/Stencil · **Phase:** `check`/`sample` fall out of Phase 4 nearly free; `infer` is Phase 5 · **Confidence: high (check/sample), medium-high (infer quality — but self-measuring)**

### What it is

The flagship task's adoption barrier is the blank page: *"I don't have a JSON Schema."* Close the loop with three subcommands:

- **`fnlp schema infer examples/*.txt -o schema.json`** — read a handful of example documents and emit a proposed extraction schema. The decode is constrained by a **meta-schema grammar** (the grammar of the supported JSON-Schema subset itself), which yields the closure property that makes this idea house-brand elegant: **the inferrer can only emit schemas the engine can compile.** Valid-by-construction, one level up.
- **`fnlp schema check s.json`** — compile-or-fail with the exact unsupported keyword named (the existing exit-8 semantics, §7.3/§8.5, exposed as a first-class command; no model load, wire-speed, CI-friendly).
- **`fnlp schema sample s.json -n 5`** — random-walk the compiled automaton and print valid instances. This is the §9.3 fuzzer's core, shipped as DX: users *see* the shape-space their schema admits before spending a single model token.

### Why it makes the project obviously better

- **First-five-minutes magic:** "point it at 5 invoices, get a working schema, extract 100K" is the demo that sells the flagship. Today's plan assumes the user arrives schema-in-hand; most don't.
- `check` and `sample` make schema development a tight, zero-model-load loop — and give users a way to regression-test their schemas in *their* CI against *our* published grammar subset.
- The meta-circular constrained decode is exactly this project's brand of trick, and it is cheap: the meta-schema of the supported subset is itself expressible in the supported subset, modulo recursion — handled by **depth-unrolling** (D≈5 covers essentially all real-world extraction schemas) with the honest scope note attached.

### How it works

- **Per-example inference → deterministic merge.** The model proposes a schema per example (constrained decode); Rust-side widening merges them: optionality from absence (field missing in some examples → not `required`), numeric type widening, enum discovery by cardinality threshold with repetition evidence, array item unification. The merge is code, not model — deterministic and testable.
- **Optional `--refine`:** one thinking-mode pass reviewing the merged schema against the examples.
- **Self-measuring quality:** `--holdout N` withholds N examples, runs real extraction with the inferred schema against them, and reports field coverage and pressure telemetry (Idea 3). Schema quality is judged by downstream utility, not vibes — the §9.6 spirit applied to a generator.

### Utility vs complexity

`check`/`sample` are nearly free once the Phase 4 grammar engine exists (the automaton walk and the fuzz walker are the same code). `infer` adds one preset, the meta-schema fixture, and the merge module. Worst case, `infer` emits a mediocre draft the user edits — which still beats the blank page, and the holdout loop says so honestly.

### Why ranked #4

The strongest adoption/DX lever in the set, riding almost entirely on committed machinery, with the closure property giving it a correctness story rather than a "best-effort helper" story. Below Ideas 1–3 because it improves *reach* rather than the core guarantees.

---

## Idea 5 — The durable batch contract: journal, resume, receipts

**Type:** AMENDMENT — §8.4 (Conveyor daemon) + §9.5 invariant · **Phase:** 4, alongside the daemon · **Confidence: very high**

### What it is

The project's pitch workload — "score 100K documents overnight on the Threadripper" — currently dies to a 3 a.m. OOM-kill, power blip, or fat-fingered `kill`, and the answer is "re-run everything." Make the daemon crash-proof with boring, proven machinery the dependency universe already contains:

- **Per-doc journal** in fsqlite (already the run store): `(run_id, doc_id, status, attempt, output_offset)`, committed in batches.
- **`--spool`:** content-addressed copy of stdin to disk, so resume never depends on the caller being able to replay a pipe.
- **`--resume <run_id>`:** verifies artifact recipe id + task-args hash match the journaled run (mismatch = refusal with a named diff — the determinism contract forbids silently mixing configurations), then skips completed ids and continues.
- **Heartbeat `status` NDJSON lines** (stderr or `--status-fd`): queue depth, docs/min, ETA, error taxonomy — the observability an overnight operator actually checks.
- **End-of-run receipt** (already sketched as "drain receipts" — promote and specify): per-label counts, score histograms, latency percentiles, error taxonomy, escalation-tier counts (Idea 2), all config/prompt/artifact hashes. The receipt is the run's provenance document.

### Why it makes the project obviously better

- This is the difference between a demo and **infrastructure**. Ops engineers evaluate batch tools on exactly these behaviors before trusting them with a corpus; their absence caps fnlp at "interesting toy" for the audience the Threadripper story targets.
- **The determinism contract makes resume provably safe — for free.** Because batch-M ≡ batch-1 per sequence is already a CI-gated invariant, a resumed run's outputs are *byte-identical* to the uninterrupted run's (modulo completion order). That upgrade — resume as a proven invariant rather than a best-effort feature — costs nothing extra and is a claim no llama.cpp-server-class baseline can make. CI test: `kill -9` injection mid-run, resume, assert corpus equality.
- The receipt closes the audit loop for the compliance-flavored use cases (redaction runs especially): one artifact that says exactly what ran, on what, with which hashes, and what came out.

### Utility vs complexity

Low complexity: every component (fsqlite, bounded queues, envelopes, per-doc isolation) is already planned; this specifies their composition plus ~three flags. Journal write amplification is managed by batched commits — fsqlite is local and cheap at this rate.

### Why ranked #5

Unsexy and obviously right. It ranks mid-list not because it's uncertain — it's the *most* certain idea here — but because it defends expected value (runs that finish) rather than creating new value. It is the idea most likely to be silently thanked at 3 a.m.

---

## Idea 6 — `fnlp eval`: bring-your-own-benchmark

**Type:** NEW capability — productizes §9.6 (Assay) · **Phase:** 5 · **Confidence: high**

### What it is

The honesty doctrine makes *our* claims carry evidence. Extend the same instrument to the user's domain:

```bash
fnlp eval --task ner --dataset my_labeled.ndjson --gold entities
fnlp eval --task classify --dataset tickets.ndjson --gold label \
          --compare presets/promptA.json presets/promptB.json
fnlp eval --task extract --dataset invoices.ndjson --gold fields --assert-min field_f1=0.85
```

Output: the **same scorecard artifact** §9.6 mandates internally — metrics per task type (span-F1 exact+relaxed, accuracy/macro-F1/ECE, field-F1, rank correlation), stamped with recipe id, prompt hash, thinking mode, and dataset hash — plus paired-bootstrap significance for `--compare` (in-house resampling math; no new dependencies) and CI-gate assertions.

### Why it makes the project obviously better

- **"Measure it on your data before you believe us"** is the strongest trust move this project can make, it is unique in the local-LLM tool space, and it is ~10% incremental work on a harness the plan builds anyway for its own release gates.
- It makes customization **safe**: presets and prompts are user-overridable by design (`--preset-file`), but today an edit is a leap of faith. With `eval --compare`, every prompt tweak gets a measured verdict — the §10.2 keep/revert loop, exported to users.
- It closes two loops in this document: Idea 2's escalation thresholds get **refit on the user's corpus** (calibration where it matters), and Idea 4's inferred schemas get scored on held-out data by the same machinery.
- Strategic honesty dividend: a tool that *invites* measurement on arbitrary user data is making a credibility claim no benchmark table can. The scorecards users generate become the project's distributed, unfakeable evidence base.

### How it works / complexity

Dataset adapters (NDJSON with a `gold` field per task shape), the §9.6 metric implementations (being built regardless), a comparison runner over the batch fabric (evals are themselves corpus workloads — Conveyor makes them fast), fsqlite persistence of eval runs, and the assertion/exit-code surface. Low-medium complexity, near-total reuse.

### Why ranked #6

Pound-for-pound the best trust-builder after Idea 3, and the cheapest major feature in the set. It sits below Ideas 4–5 only because its audience (users with labeled data) is a subset of all users — though it's precisely the subset that writes adoption-driving blog posts.

---

## Idea 7 — Document-major multi-task execution

**Type:** AMENDMENT — §6.7 (Conveyor step planner), §7.0 (prompt layout contract), §8.4 (daemon envelope) · **Phase:** layout decision **now** (design stage), machinery in Phase 4 · **Confidence: medium-high**

### What it is

Conveyor's prefix cache shares the *task prompt* across many documents (corpus-major: one task × 100K docs). Document-intelligence pipelines run the transpose: **many tasks over the same document** (extract + ner + redact + summarize per contract, per email, per report). Under the plan's implicit instructions-first layout, that costs K full re-prefills of the same document. Amend the prompt layout contract to **instructions-last**:

```
[task-static preamble]  [document]  [task-dynamic tail + answer scaffold]
```

and the existing copy-on-write fork machinery generalizes into a **fork tree**: fork after the task-static block for corpus-major (exactly today's design), *additionally* fork after the document for doc-major. Surface: `fnlp multi --tasks ner,redact,extract doc.txt`; daemon lines gain an optional `"tasks": [...]` array.

### Why it makes the project obviously better

- **The arithmetic is not speculative.** 4K-token document, ~200-token instruction blocks, 5 tasks: instructions-first pays 5 × 4,200 = 21,000 prefill token-forwards; doc-major pays 4,000 + 5 × ~400 ≈ 6,000 — a **~3.5–5× prefill reduction** on exactly the workload (multi-task document pipelines) that the task portfolio exists to serve. Prefill is the R2 cost center (§6.1); this is a structural lever on it, the same class as the prefix cache itself.
- **Synergy with the ladder (Idea 2):** a think-retry or consistency sample forks from the document KV too — escalation re-pays only the instruction tail, not the document. The ladder gets cheaper the moment this lands.
- Both sharing patterns coexist: an M-doc × K-task grid shares the task-static block globally *and* each document's KV down its task column. The fork machinery is direction-agnostic; the plan just hasn't asked it to be yet.

### Why this must be decided now (the design-stage argument)

Prompt layouts are frozen, hashed, versioned data (§7.0) with eval fixtures built against them. Flipping the layout *after* Phase 4–5 means re-fixturing every task eval and re-running every scorecard. Flipping it *now* costs a paragraph in §7.0. This is the one idea on this list with a genuine now-or-expensive-later structure, which is why it makes the cut despite medium implementation weight.

### Risks & obligations

- **Instruction position can affect task accuracy.** Mitigation: the layout is per-task measured data — Phase 4's eval stand-up runs both layouts per task and ships whichever wins (practice generally favors instructions-last for long documents — recency-biased attention keeps the instructions hot — but the doctrine is *measured wins*, and both layouts stay expressible).
- Step-planner admission must handle forks at heterogeneous depths — but the prefix cache is already a tree (system → task → fork point); this widens it rather than restructuring it. KV memory: task branches append only their small dynamic tails; copy-on-write handles the rest.
- Batch-invariance and prefix-fork ≡ cold-prefill gates extend to fork trees unchanged (they're per-sequence properties).

### Why ranked #7

A real structural throughput lever with a cheap-now/expensive-later decision embedded in it. It ranks below the trust/robustness cluster because its beneficiary set (multi-task pipelines) is narrower than "everyone," and it carries the one open empirical question (layout quality) on the list — albeit with a cheap measured answer.

---

## Idea 8 — `fnlp tune`: per-install measured personalization

**Type:** NEW capability — extends §6.3 dispatch, §6.13/AF-5 pool sizing · **Phase:** 6 (repackages Phase 3–4 harnesses) · **Confidence: high (safety/mechanism), medium (magnitude)**

### What it is

The doctrine is *measured-faster wins* — but the shipped dispatch tables, USL pool caps, and batch-M defaults are measured on the project's reference machines. The install base will not look like the reference machines: M4 Pro vs Max is a 2× bandwidth spread; DDR4 Zen 3 vs DDR5 Zen 4 moves the roofline; core counts and P/E topologies move USL peaks. `fnlp tune [--quick ~2min | --full ~15min]` re-runs the bounded sweeps **on the user's silicon** — autovec-vs-SDOT/SMMLA per shape, USL thread-cap fits per op class, the batch-M throughput curve, mmap-vs-owned load — and writes a `tuned_profile.json` keyed to (artifact hash, binary version, CPU signature). `robot backends` reports provenance (`defaults` vs `tuned@date`); gauntlet and bench output always states which was in effect.

### Why it makes the project obviously better

- It closes the **last mile of the measurement doctrine**: today the doctrine reaches the CI fleet; with `tune` it reaches every machine that runs the product. The franken_ocr lesson that motivated measured dispatch in the first place — autovec beating SMMLA on some shapes on M-series — is precisely the kind of result that varies across the M4 Pro/Max/Ultra spread within one ISA tier.
- The user-visible ritual is honest and satisfying: `tuned: +14% docs/min on this machine (measured, before/after)` — printed from real numbers, or `tuning found no improvement; defaults retained`, which the doctrine is equally proud to print.
- It converts AF-5 from an internal artifact into a product feature at near-zero marginal cost: the sweep harnesses exist as `benches/` by Phase 4; `tune` is those harnesses, bounded, repackaged in-binary, writing a profile instead of a ledger row.

### The hard doctrine line (what makes it safe)

Tuning selects **only among bit-identical execution paths** — speed-only choices. It can never touch numerics: not the quantization algebra, not the AVX2 exact construction, not reduction orders. Consequence: no per-machine parity re-proof is needed (`robot selftest` already proves kernel identity on the user's CPU, and stays a separate, always-available check). Deterministic fallback: shipped defaults, always one flag away (`--no-tuned`), and stale profiles (artifact/binary/CPU mismatch) are ignored with a notice, never silently applied.

### Why ranked #8

Bounded upside (single-digit to ~30% depending on how far the user's machine sits from the reference boxes) and zero product-surface novelty — which is exactly why it's #8 and not higher. It makes the list because it is cheap, safe by construction, perfectly on-doctrine, and improves the *actual* performance users experience rather than the performance the ledger records on machines they don't own.

---

## Appendix — the other 22 candidates (considered and cut/absorbed)

Kept for the same reason `NEGATIVE_EVIDENCE.md` exists: the winnowing should be auditable, and several of these deserve one-paragraph fold-ins rather than oblivion.

| # | Candidate | One-line assessment | Verdict |
|---|-----------|---------------------|---------|
| 9 | Sequence packing with block-diagonal segment masks + per-segment positions | Provably batch-invariant packing is achievable, but layer-major batching + the prefix cache already capture the win; mask complexity unjustified | CUT |
| 10 | `redact --verify` clean-pass receipt | Re-run the full detector union on redacted output; nonzero findings = nonzero exit + leak report; small and genuinely good | **FOLD into §7.7** as a flag (recommended) |
| 11 | Prompt-boundary token healing | Re-tokenize the tail at the free→constrained boundary so merged pieces don't degrade the first constrained tokens; real correctness subtlety | **FOLD into §7.3** (recommended) |
| 12 | Standalone run receipts + `fnlp rerun <receipt>` | Provenance + replay | ABSORBED into Idea 5 |
| 13 | `--consistency N` self-consistency flag | Field-level majority voting over seeded samples | ABSORBED into Idea 2 (tier 3) |
| 14 | Standalone low-confidence spill file | Route the hard residue downstream | ABSORBED into Idea 2 (tier 4) |
| 15 | Logit canary fingerprints | Golden-prompt logit hashes shipped in the artifact, checked at load — catches corrupt/mismatched artifacts beyond the census | **FOLD into `robot selftest`** (small, recommended) |
| 16 | Corpus rollup statistics at drain | Label histograms, entity frequencies | ABSORBED into Idea 5 (receipts) |
| 17 | Standalone grammar dev tools | Compile-check + instance sampling | ABSORBED into Idea 4 |
| 18 | KV-snapshot persistence (disk-cached shared-prompt prefills) | Warm-start CLI invocations; invalidation complexity vs measured recompute cost unknown | DEFER until Phase 4 numbers exist |
| 19 | Loop-2 residual "progressive precision" (int4 base + residual stream on sensitive tensors) | Speculative; OQ-13 + AF-1 tiering likely suffice | DEFER (note beside OQ-13) |
| 20 | Per-task quant routing (int4 for classify, int8 for extract) | Two resident weight sets; memory cost kills it | CUT |
| 21 | Hidden-state document signatures for `resolve` blocking | Collides with the frankensearch boundary (§1.2); lexical blocking suffices | CUT |
| 22 | Directory-watcher ingestion mode | `fswatch \| fnlp batch` already composes; Unix philosophy wins | CUT |
| 23 | HTTP metrics endpoint for the daemon | No server in v1; Idea 5's status lines cover observability | CUT |
| 24 | CSV/Markdown-table output emitters | Trivial serializer sugar over extract; add whenever | CUT (backlog) |
| 25 | WASM playground for schema development | Fun, off-mission for v1 | CUT |
| 26 | Artifact signing (ed25519/minisign-class) | Manifest hashes pinned in the binary are already strong; key-management burden not yet justified | DEFER |
| 27 | Preset marketplace / community pack registry | Social infrastructure; premature | CUT |
| 28 | Dedicated table-extraction task | `extract` with array-of-object schemas covers it | CUT |
| 29 | `diff-extract` (structured diff of two document versions) | Compose two runs + jq | CUT |
| 30 | Multi-turn interactive extraction refinement | Agent territory; `chat` + the library API cover it | CUT |

Two near-winners deserve emphasis: **#10 (`redact --verify`)** and **#11 (token healing)** are both small enough to be single-paragraph plan amendments and strong enough that cutting them entirely would be a mistake — they lost slots only to ideas with bigger ceilings, and both are recommended as fold-ins during the next plan revision.

---

*End of document. Ideas 1–8 are proposals, ordered by conviction; each names its plan-integration points, proof obligations, and fallbacks so it can be adopted (or rejected) through the same review discipline as the plan itself (§15.2).*
