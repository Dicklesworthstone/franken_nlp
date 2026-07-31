# WIZARD_SCORES_CC_ON_COD.md — CC's candid evaluation of WIZARD_IDEAS_COD.md

> **Post-review status (2026-07-30): historical cross-score, not specification.** This file evaluated pre-v3 snapshots; its hashes and plan-section/factual-audit statements intentionally describe those snapshots and are stale for the current files. Plan v3 §10.6 is authoritative. The plan now has AA-* cards and §8.6, correctly blocks public model assets at LG-1, keeps `FeedForced` token-id exact (a unique byte run is insufficient), and rejects token healing as an exact/parity path; the preserved idea record now labels Copy-as-Decode withdrawn. This review's claims that legal/illegal margins are effectively free/always-on or that holdout coverage alone can validate schema inference were also rejected. Preserve the scores as adversarial provenance, not present project state.

**What this is:** an idea-by-idea evaluation of the eight proposals in `WIZARD_IDEAS_COD.md`, scored 0–1000. Each score reflects, jointly: how good/smart the idea is, practical usefulness for humans and AI agents, implementability-correctly, and whether utility justifies added complexity and tech debt.
**Calibration:** ~500 = decent idea with real flaws; ~700 = good idea I'd adopt with changes; ~850+ = excellent, adopt with minor edits; 950+ = reserved for ideas I could find no substantive fault with. Scores are candid, not diplomatic.
**Due diligence performed:** all five external citations in COD's document were live-verified (results in the Factual Accuracy Audit, §B), and every plan-section reference was checked against the actual `COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`.

## Summary scoreboard

| COD # | Idea | Score | One-line verdict |
|---:|---|---:|---|
| 2 | Durable, incremental corpus jobs | **900** | Best idea in the file; a strict superset of my treatment, and better |
| 1 | Stencil as an execution compiler | **875** | Superb frame; one of its three primitives has a specification flaw |
| 5 | User-owned qualification & safe activation | **830** | My `fnlp eval` + genuinely better rigor; mildly over-scoped plumbing |
| 6 | Selective automation + correction flywheel | **820** | My escalation ladder + a principled flywheel; adoption-dependent |
| 7 | Exact continuation-trie scoring | **810** | Novel vs my set; sound, exact, honest; narrower audience |
| 4 | Document-major analyze packs | **800** | We independently converged on the core; their scoping is heavier |
| 3 | Bounded declarative task-recipe compiler | **700** | Right shape, wrong time; marginal utility overstated for v1 |
| 8 | Cross-loop wavefront coalescing | **580** | Most impressive analysis; solves a problem the plan's design mostly doesn't have |

Mean ≈ 789. That is a strong list. The distribution is honest: one idea near the ceiling, one clearly below the line.

---

## Idea 1 — Stencil becomes a grounded execution compiler · **875/1000**

**COD's claim:** compile each grammar state into a decode program choosing among `ProjectLegal(rows)` (sparse lm_head projection over the legal set), `FeedForced(tokens)` (forced-run causal micro-prefill), `CopyFromSource(language)` (source-substring grounding), and `FullProjection(mask)` (the fallback).

### What's excellent

- **The frame is the best architectural sentence in either document:** the grammar should be an execution *planner*, not a filter bolted onto generation. The project already pays the full cost of sound grammar/tokenizer alignment (their citation of the real Park/Zhou/D'Antoni GCD paper is apt — mask soundness is a genuine systems problem); extracting execution decisions from that investment is exactly right for a one-model appliance.
- **`CopyFromSource` is my own #1 idea** (grounded-by-construction decoding), independently derived — which I read as strong convergent evidence it belongs in the plan. Treatment comparison: COD's stance on repeated-occurrence ambiguity ("return all compatible intervals; never fabricate a unique offset") is more principled than my nearest-to-hint default, and I'd adopt it. My treatment is deeper on mechanism — the lazy-mask cost *asymmetry* (copy regions are cheaper to mask, not dearer), per-document memoization consequences, per-task defaults (NER/redact/citations ON), the unsatisfiable-at-field-entry semantics, and the new e-process invariant. Merge both; neither is complete alone.
- The sparse-projection exactness argument is correct and carefully stated: under masked greedy, illegal rows cannot win; under conditioned sampling the denominator cancels. The refusal to let telemetry "pretend a candidate-conditional value is full-vocabulary mass" is properly honest.

### Candid problems

1. **`FeedForced` as specified has a trigger-rarity flaw.** COD forces only when "exactly one token ID is legal" at the grammar/tokenizer product state. But under a byte-trie mask over a 166,144-piece SP vocab, a forced *byte* run (`"vendor": `) almost always admits **multiple** legal tokens simultaneously — `"`, `"v`, `"vendor`, … are all prefix-compatible continuations. Exactly-one-legal-token states are therefore rare, and the primitive as written will seldom fire. The valuable case is **byte-level** jump-forward (force the unique byte continuation, then choose a canonical tokenization) — but that breaks the exact-state-equivalence promise COD makes: byte output is identical, yet token boundaries, KV, and hidden states differ from masked sequential decode, so downstream free-text regions can drift, and the run-exit boundary needs token-healing machinery COD never specs. So the primitive is either (a) sound but nearly inert, or (b) potent but unable to meet its own promotion gate as written. Fixable — SGLang-style jump-forward plus healing plus a byte-equivalence (not state-equivalence) gate for L5-relevant surfaces — but the headline throughput claim is materially softer than the document implies. Credit: their own measurement plan ("report forced-run length distributions") would expose this quickly.
2. **The sparse-projection win is modest and they don't size it.** The full lm_head GEMV is ~13–14% of the per-token decode weight stream (≈0.5 GB of ≈3.7 GB at the int4+int8 recipe). Skipping it on structural tokens is real money but not transformative; the scoring-path slicing that delivers the 10–100× is *already in the plan* (§6.10). The idea's perf framing borrows some glow from machinery that exists.
3. **It quietly forecloses always-on constraint telemetry.** My #3 idea (fabrication detection: pre-mask argmax override rate, legal-set mass) requires the full-vocab computation that `ProjectLegal` exists to skip. COD acknowledges the tension and resolves it as "audit-only" telemetry. I think that's the wrong default: the margin/override signals are the cheapest honest quality instrument the constrained path can have, and constrained decode's failure mode (uncertainty laundered into confident valid JSON) is precisely where this product needs standing instrumentation. A sampled-audit compromise (full projection every k-th step, plus always-on during evals) deserves explicit design, not a footnote.

**Utility vs complexity:** high utility even after discounting `FeedForced`; complexity honestly labeled medium-high; proof obligations well-enumerated. **Overlap verdict:** grounding — merge, jointly better than either alone; the compiler frame — theirs, and it's a genuine contribution; the perf primitives — theirs but overweighted.

---

## Idea 2 — Durable, incremental corpus jobs · **900/1000**

**COD's claim:** a first-class `fnlp job` surface (start/status/resume/verify/materialize) over a content-addressed **semantic execution key**, transactional item state in fsqlite, explicit exactly-once scoping, privacy-default journals, and cross-run incremental recomputation.

### What's excellent

This overlaps my #5 (durable batch contract) and is **better than my treatment** in three specific ways I'll concede plainly:

1. **The semantic execution key + cross-run result cache.** My version journaled within a run; theirs makes results reusable *across* runs under exact-key match ("re-running after changing five documents recomputes five documents"). For recurring corpus scans — compliance sweeps, nightly re-classification — this is the single most accretive operational feature in either document, and the exact-match conservatism is what makes it trustworthy rather than spooky.
2. **Privacy-default journals.** Default state stores IDs, digests, status, metrics — never text; spooling is an explicit, named, permissioned, purgeable choice. My `--spool` was opt-in but I never specified the no-text default for the journal itself. For a product whose pitch includes "redact before text leaves the machine," their default is correct and mine was underspecified.
3. **Exactly-once honesty.** The distinction between what a materialized checksummed output can promise (one canonical record per item) and what raw stdout can promise (stable IDs, at-least-once, documented) preempts "a common infrastructure lie." That paragraph alone is worth adopting verbatim.

The proof-obligation list (kill injection at every transition boundary, fsync-order races, disk-full, corrupt-tail WAL, per-component key perturbation, no-text-column schema test) is the most complete in either document.

### Candid problems

- **Surface-area growth.** `job start/status/resume/verify/materialize` plus retention policies plus a purge command is a second product surface next to `batch`. Defensible (pipe vs job is a real distinction) but it's five subcommands, a state machine, and a docs burden that must be frozen and contract-tested; my flags-on-batch version delivers ~80% of the value at ~half the frozen surface. A reviewer should consciously choose the bigger surface, not inherit it.
- Minor: the incremental cache invites scope creep toward "semantic similarity" reuse; COD explicitly forbids it, which is right, but the boundary will need active defense in review.

**Utility vs complexity:** the highest-certainty value in the file; every component is boring, proven, and testable; complexity is ordinary. **Overlap verdict:** theirs wins. I'd adopt COD's Idea 2 over my own #5 and fold my provably-safe-resume framing (batch-invariance ⇒ resumed ≡ uninterrupted, as an *invariant*, which they also reach via their equivalence obligation) into it.

---

## Idea 3 — Bounded declarative task-recipe compiler (TaskIR) · **700/1000**

**COD's claim:** a declarative `.fnlptask.json` recipe format compiled to a frozen `TaskIR`; built-ins dogfood the same IR; strong negative capabilities (no code, no network, no template language); recipe tooling (`check/explain/sample`).

### What's excellent

- The **negative-capabilities list** is the best part — no code execution, no network, no Jinja/WASM/expression language, no retry-after-failure, no uncalibrated confidence claims. That list is publishable as-is and should constrain *any* future extensibility work regardless of this idea's fate.
- The instinct (extensibility without plugins; recipes are auditable data, not installable code) is the right resolution of a real tension, and `recipe explain` (token counts, lm_head strategy, memory bounds, required calibration — before any model load) is genuinely good agent ergonomics.
- Dogfooding built-ins through the IR, with a policy test against bypass, correctly names the "neglected second-class surface" failure mode.

### Candid problems

1. **The marginal utility at v1 is substantially overstated.** Walk the motivating examples against the *existing* plan: a support-ticket routing policy → `classify --labels` + presets (already arbitrary); a legal-clause schema → `extract --schema` (already arbitrary, the flagship); a custom faithfulness rubric → judge rubric presets + `--preset-file` (already planned); a house PII policy → redact policy packs (already data). The plan's §7.0 already makes presets data with user overrides. What recipes add is *recombination* — novel prompt structures, custom candidate sets with declared normalization, postcondition programs. Real, but a fraction of the pitched "twelve tasks → task appliance" delta, because the twelve tasks are already parameterized at their high-value joints.
2. **Inner-platform pressure is under-priced.** Forcing built-ins through a declarative IR means every future built-in need either fits the format or grows it. COD gestures at the escape hatch ("Rust modules only where genuinely specialized") but that boundary is exactly where these systems rot: the format accretes features to chase the built-ins, and you've built a small framework inside the appliance — the thing §3.1's generality-tax doctrine exists to forbid.
3. **A public recipe format is frozen contract debt taken on at the design stage.** Schema versioning, canonical cross-host compilation, migration policy, fuzzing, docs — for a project that hasn't yet shipped Phase −1. The plan's own discipline (freeze contracts in Phase 6, after the surfaces stabilize) argues against minting a new public contract in Phase 4.

**The right version of this idea** — and it's close — is: build `TaskIR` as the *internal* compilation target now (genuinely good architecture; it consolidates TaskSpec/TaskPlan/presets/hashes and makes the twelve built-ins uniform), ship recipe tooling for the *existing* parameterization surfaces (`check/explain/sample` over schemas/labels/rubrics/presets — which absorbs my #4's check/sample exactly as COD notes), and defer the *public* recipe format to v2 once twelve built-ins have stress-tested the IR's expressiveness. That sequencing captures most of the architectural value at near-zero contract debt.

**Utility vs complexity:** good idea, honest about mechanism, but utility-at-v1 does not justify a public format's debt; as internal architecture it's a clear yes. Scored as written: 700.

---

## Idea 4 — Document-major analyze packs · **800/1000**

**COD's claim:** first-class multi-task-over-one-document execution (`fnlp analyze --pack`), an explicit four-segment prompt ABI decided *now*, a content-addressed KV fork graph, a document×task matrix scheduler with hybrid tiles, and atomic bundle semantics.

### What's excellent

- **We independently converged on the core** — this is my #7 (document-major multi-task execution), including the identical load-bearing insight: *the prompt-segment layout must be decided now, while prompts and scorecards are still free to change*. Two models deriving the same now-or-expensive-later argument from the same plan is the strongest adoption signal in this whole exercise.
- Their additions over mine are real: per-task layout **eligibility decided by locked scorecards** (a task is never made shareable by lowering its quality gate — exactly the right doctrine phrasing); atomic bundle semantics with `blocked_by` for dependent branches; the observation that tokenization, coordinate maps, source-substring indexes (Idea 1c/my #1), and rule-detector runs are also built once per document. The vLLM prefix-caching citation is apt precedent for content-hashed KV blocks.

### Candid problems

1. **Scope heaviness for Phase 4.** Packs-as-DAGs (typed inter-task dependencies), a matrix scheduler with corpus-major/document-major/hybrid-tile modes, and a content-addressed KV block graph is a lot of machinery on top of the plan's snapshot-and-fork prefix cache. My leaner v1 (fork tree + `--tasks` list; independent branches only; no DAG, no tiles) delivers the structural prefill win (~3.5–5× on the workloads we both computed) with a fraction of the scheduler surface. Dependencies (NER → redact) and hybrid tiling are v1.5 material.
2. **The KV block graph imports vLLM-shaped generality** (reference counts, cache salts, block hashing) that a single-process, single-model appliance with a tree-shaped fork structure may simply not need. The plan's simpler snapshot/fork design should be exhausted first; block graphs are a measured upgrade, not a starting point.
3. Privacy salts and timing-leak tests across namespaces are listed without a threat model; in a local single-user process this is likely ceremony.

**Overlap verdict:** core idea — tie, independently derived; the ABI-now argument — tie (both made it, load-bearing in both); scoping — mine is the better v1, theirs the better v2 product vision (packs are a genuinely nice surface). Merge: my scoping, their eligibility-by-scorecard rule and bundle semantics, packs later.

---

## Idea 5 — User-owned qualification, calibration, safe activation · **830/1000**

**COD's claim:** productize Assay as `fnlp eval` / `calibrate` / `qualify` / digest-gated `models activate` + `rollback`, with dev/calibration/locked-test split manifests and leakage refusal.

### What's excellent

- Overlaps my #6 (`fnlp eval`) and extends it. The standout addition — the best single upgrade to any of my ideas found in COD's document — is the **split manifest with enforced leakage discipline**: the engine *refuses* to fit calibration on locked-test IDs and records overlaps. That imports real ML-evaluation rigor at trivial cost and protects users from the most common way people fool themselves with local evals. I'd adopt it verbatim into my #6.
- The qualification receipt ("a reproducibility receipt, not a universal certificate," scoped to named data and host) is exactly the house's honesty register. `INSUFFICIENT_DATA` over unstable confidence intervals on tiny sets: correct and easy to forget.
- Digest-bound activation composes beautifully with Idea 2's semantic keys: a stale qualification cannot authorize a different prompt/calibration/packing.

### Candid problems

1. **Activation/rollback plumbing is mildly over-scoped for a one-model appliance.** v1 has one vendor (us) shipping int8 and int4 of one model. An "active model" state layer with gated activation and instant rollback is infrastructure whose main customers are prompt/recipe churn (real) and multi-artifact fleets (mostly hypothetical at v1). The eval/calibrate/qualify core is Phase 5; activation gating can trail in Phase 6 without loss.
2. Paired-interval and paired-under-resume obligations are right but nontrivial to implement correctly (pairing across unordered execution needs care); the document waves at this faster than the work deserves.
3. Energy metrics "where available" — on the primary targets, credible energy measurement is its own project; should be cut from v1 scope rather than half-promised.

**Overlap verdict:** theirs is better overall — my leaner eval+compare+assert is the right core, their split-manifest rigor is a clear improvement, their activation layer is the part I'd trim. Merged version scores higher than either.

---

## Idea 6 — Selective automation + correction flywheel · **820/1000**

**COD's claim:** a versioned `DecisionPolicy` (signals → actions with an explicit loss table) for accept/think-retry/consistency-N/abstain/spill, plus a human-review correction loop: spill records → schema-validated corrections → named local suites → explicit `calibrate`/`policy fit` refits → digest-bound new policy.

### What's excellent

- The policy core **is my #2 (AF-6 escalation ladder)** — same signals, same actions, same never-call-the-cloud boundary, same loss-matrix framing, same "presets name frozen threshold artifacts, not adjectives." Convergence again; this belongs in the plan.
- **The correction flywheel is their genuine addition and it's good.** The properties are exactly right: no hidden online learning, no silent prompt mutation, corrections act only through explicit refit, everything digest-bound and reversible, correction data firewalled from project release scorecards. This converts the spill stream from a dead end into the user's own calibration data — operationally accretive in precisely the way the document claims, and it feeds Idea 5's suites cleanly.
- Per-task **signal-validity declarations** (summarization must not pretend mean token probability is calibrated correctness) is a rigor detail my treatment lacked. Adopt.
- The Draft-Conditioned Constrained Decoding citation is used honestly — "supports measuring constraint pressure, not adopting the method wholesale."

### Candid problems

1. **Adoption-dependency of the flywheel.** The loop requires sustained human correction labor. For most users it will be unused surface; for the minority running consequential recurring pipelines it's transformative. Fine — but the review/import/suite/fit surface is four new subcommand families whose maintenance cost is paid by everyone. Phase 6, not Phase 5, for everything past `--review-out`.
2. **It inherits Idea 1's telemetry demotion.** Constraint pressure is admitted as a signal only "when a full-vocabulary audit actually computed it." Since their own sparse projection makes that rare, one of the most decision-relevant signals is structurally absent from the fast path. My margin-based always-on variant costs ~nothing and should be the default input to this very policy; the two ideas as written quietly undercut each other, and neither document's author gets to feel smug — mine didn't anticipate sparse projection, theirs didn't re-examine the policy inputs after proposing it. A merged design (always-on margin/override; sampled full-mass audits) fixes both.
3. `policy fit --target-risk` implies selective-risk control that is only as good as exchangeability between calibration and production streams; the document says "distribution shift invalidates coverage" in Idea 5 but doesn't carry that caveat here, where it matters most.

**Overlap verdict:** ladder core — tie (mine tighter on the AF contract framing and tier mechanics; theirs adds the validity table). Flywheel — theirs, a real contribution with an honest adoption caveat.

---

## Idea 7 — Exact continuation-trie scoring · **810/1000**

**COD's claim:** compile multi-token candidate sets (labels, catalogs, rubrics) into a token trie; score each unique prefix state once with COW state forking and batched frontiers; two explicitly distinguished probability semantics; fall back to naïve scoring when prefix sharing is poor.

### What's excellent

- **Novel relative to my set** — I had nothing equivalent, and I rate it a genuine gap in my list. It converts §7.4's "multi-token labels via teacher-forced continuation" from a per-candidate loop into shared-prefix dynamic programming, and in doing so unlocks a real product surface: **large-taxonomy classification and catalog-constrained entity canonicalization** (`resolve --catalog`), which are exactly the corpus workloads this appliance courts. Trie-constrained candidate generation has solid prior art (GENRE-style entity linking over prefix tries), so feasibility risk is low.
- The **two-semantics honesty** (full teacher-forced logprob with per-unique-node denominator reuse, vs trie-conditional scoring over outgoing edges — "must never be labeled full-vocabulary likelihood") is exactly right, and the prefix-of-another-candidate terminal-edge detail shows real care.
- "The compiler estimates reuse and selects the naïve scorer when measurement says so" — the doctrine's measured-wins rule, correctly internalized. The proof story (exact equality vs naïve per declared mode, candidate-order invariance, batch-composition invariance) is clean and differential-testable.

### Candid problems

1. **The win is bounded by prefix sharing, and they say so — but the shared-scaffold portion is already the prefix cache's job.** The trie's marginal value over "batch all candidates through the engine with the scaffold prefix cached" is only the *label-internal* shared prefixes plus the per-unique-node denominator reuse. For flat, low-overlap label sets the gain is small; for hierarchical taxonomies and entity catalogs it's large. Beneficiary set: narrower than Ideas 1–6, as they admit.
2. **KV-fork memory at wide frontiers** is real (thousands of active trie nodes × per-node COW state); admission budgeting handles it but the constant matters and isn't estimated.
3. An alternative design goes unmentioned: **short-ID indirection** — constrain selection to compact canonical IDs (one to two tokens each) with the catalog in the prompt or via retrieval — which sometimes beats deep-trie scoring outright for very large catalogs. The trie is the more general and more principled mechanism, but a candid EV comparison should name the cheap rival.

**Overlap verdict:** theirs alone. Adopt, at Phase 4–5, behind the existing measured-wins discipline.

---

## Idea 8 — Cross-loop physical-layer wavefront coalescing · **580/1000**

**COD's claim:** since physical layer *i*'s weights are identical across both loops, coalesce ready rows from different sequences at `(layer i, loop 0)` and `(layer i, loop 1)` into one GEMM, routing KV by loop tag; profile-gated, static-fallback, explicitly allowed to die in `NEGATIVE_EVIDENCE.md`.

### What's excellent

- It is the most model-specific, intellectually impressive idea in their file — it takes "22 weights, 44 executions" seriously as a *scheduling* fact, not just a bandwidth fact, and the proposed gating is exemplary: trace-derived occupancy histograms **before** implementation, honest "not a free 2×" framing, counters demanded ("fewer partial weight sweeps, not merely a microbenchmark win"), automatic disable, death-without-regret. If every speculative systems idea were written this honestly, ledgers would be shorter.

### Candid problems — and they're structural

1. **The fragmentation it targets mostly does not exist under the plan's own design.** §6.7 specifies a *synchronized layer-major step planner*: each engine step sweeps `loop { layer { GEMM over all admitted rows } }`, with mixed prefill/decode rows co-batched in the same sweep. Under that design, every in-flight row is at the same `(loop, layer)` wavefront *by construction* — there are no "rows ready at layer 7, loop 0" waiting beside "rows ready at layer 7, loop 1," because nothing progresses independently. Loop-stage fragmentation arises only if the engine adopts per-cohort asynchronous progress (mid-sweep admission, per-sequence pacing) — which is to say, **the idea largely solves a problem that its own premise introduces**. Their steady-state cohort-pipeline picture is an argument for asynchronous admission, whose actual benefit versus the synchronized design is admission latency of at most ~one sweep (≈ one token-time, tens of ms) — negligible for a batch daemon.
2. The residual real cases (drain tails, cancellation churn, heterogeneous completion under continuous batching) produce *underfilled sweeps*, which coalescing across loop stages could thicken — but underfilled sweeps still stream weights once per sweep for everyone; the recovered quantity is GEMM row-occupancy at the margins, not duplicate weight streams. The honest upper bound is thin, and their own document half-knows it ("if occupancy is already dense, the expected value is zero").
3. **The proof surface is the heaviest in either document** — readiness-DAG model checking, hostile interleaving replay, batch-invariance under shifting coalescing partners — levied against the project's most sacred invariants (determinism, batch-M ≡ batch-1), for a benefit that is conditional on a workload pathology not yet observed.

**Why 580 and not lower:** because the *actual recommendation* — add trace instrumentation and a research card now, implement nothing until occupancy histograms prove fragmentation — is cheap, correct, and doctrine-perfect. As a deferred research note it's worth ~750; as a top-8 idea consuming a slot that telemetry, schema inference, or `fnlp tune` could have held, it's the clearest misallocation in their list. Scored as positioned.

---

## A. Overlap map and comparative judgment

| Theme | Mine | COD's | Whose treatment is better |
|---|---|---|---|
| Source-grounded decoding | #1 | #1c (`CopyFromSource`) | **Merge** — my mechanism depth (mask-cost asymmetry, memoization, task defaults, e-process invariant) + their ambiguity stance and compiler frame |
| Durable corpus jobs | #5 | #2 | **COD**, clearly — semantic keys, cross-run cache, privacy-default journal, exactly-once honesty |
| Escalation / selective automation | #2 (AF-6) | #6 | Core: tie. Flywheel: **COD's addition**, adopt later-phase. Signal inputs: **mine** (always-on margin telemetry) |
| Eval productization | #6 | #5 | **COD** — split-manifest leakage discipline is the single best upgrade to my list; trim their activation plumbing |
| Document-major execution | #7 | #4 | Core + ABI-now argument: tie (independent convergence). v1 scoping: **mine**; product vision (packs): **COD's** |
| Constraint telemetry | #3 (first-class, always-on) | absorbed/demoted to audit-only | **Mine** — demoting the cheapest honesty instrument to audit-only is COD's worst call, made worse by their own sparse projection depending on its absence |
| Schema toolchain | #4 | winnowed (check/sample folded into recipes; infer deferred) | **Mine** — deferring `infer` "for task-quality evidence" ignores that my `--holdout` design makes inference self-measuring |
| Per-install tuning | #8 | winnowed (near-winner, Phase 6 fold) | Effectively agree; theirs is a reasonable deferral of my weakest winner |

**Independent convergence** on five themes (grounding, durability, escalation, eval, document-major + the ABI-now argument) is the most decision-relevant output of this exercise: those five should be treated as review-validated and move toward plan amendments.

## B. Factual accuracy audit of COD's document

Verified live (2026-07-30):

- ✅ arXiv 2502.05111 — real: *Flexible and Efficient Grammar-Constrained Decoding* (Park, Zhou, D'Antoni). Aptly used.
- ✅ arXiv 2407.00023 — real: *Preble: Efficient Distributed Prompt Scheduling for LLM Serving*. Aptly used, with the honest "measure locally" caveat.
- ✅ arXiv 2607.18357 — real: *Decode-Time Grammars…*. Used appropriately as a broader precedent.
- ✅ arXiv 2603.03305 — real: *The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding*. Used honestly ("supports measuring pressure, not adopting the method").
- ⚠️ arXiv 2604.18170 (*Copy-as-Decode*) — the page exists but the paper was **withdrawn by its author** (authorship/contribution dispute). Citing a withdrawn paper as a "feasibility precedent" without noting the withdrawal is a genuine citation-hygiene lapse. The underlying concept (jump-forward/parallel prefill of grammar-forced spans) has other, standing prior art, so the argument survives; the citation as given should not.
- ❌ Phantom identifiers: Idea 8 says "beside **AA-S1** in §10.5" and the winnowing table says "after **AA-K1**… exists" — the plan contains no AA-* identifiers; its families are **AF-1…AF-5**. Small, but this project's culture treats invented cross-references as a real defect.
- ❌ "Amend §8.4–**§8.6**" — the plan's §8 ends at §8.5.
- ❌ Winnowing table: "Delta-compressed model downloads — premature **while public redistribution remains blocked**" — this contradicts the plan, which affirmatively verifies Apache-2.0 and de-risks redistribution (§2.2, §5.6). Redistribution is *planned*, not blocked; the deferral is right, the stated reason is wrong.

None of these invalidates an idea; all of them would be flagged in this repo's review cycle.

## C. Overall verdict

COD's list is **strong — on par with mine overall, with complementary centers of gravity.** It is better than mine at systems lifecycle (jobs, qualification, the recipe *vision*) and at model-specific ambition (trie scoring; the coalescing analysis, even though I score its EV low). It is weaker than mine on cheap, always-on honesty instruments — demoting constraint telemetry is its worst call — and it spends one of eight slots on a likely-zero-EV scheduler epic that its own text half-expects to die. Its single best contribution relative to my list is the semantic execution key / incremental-cache design in Idea 2; its single best small upgrade is the split-manifest leakage discipline in Idea 5.

**Recommended merged shortlist for the plan's review cycle:** COD-2 (durable jobs, as written, with my invariance framing) · merged grounding (my #1 + COD-1c + COD's compiler frame, with `FeedForced` re-specified at byte level behind a healing design or demoted) · my #3 telemetry restored to always-on margin/override with sampled full-mass audits · merged eval (my #6 + COD-5's split manifests; activation gating in Phase 6) · merged document-major (my scoping now, COD's packs and scorecard-eligibility rule later) · merged escalation (my AF-6 + COD-6's flywheel in Phase 6) · COD-7 (trie scoring) · my #4 schema toolchain (with `infer` kept, self-measured via holdout) · COD-3 reduced to *internal* TaskIR at v1 · COD-8 reduced to trace instrumentation + a research card.

*End of evaluation. Scores are one reviewer's candid judgment, produced after full reads of both documents and the plan, with citations live-verified; they are inputs to the ≥4-round review cycle (§15.2), not verdicts that bypass it.*
