# Dueling Idea Wizards Report — franken_nlp

> **Post-integration status (2026-07-31): historical synthesis, not specification.** Plan **v3.1 §10.6** is the authoritative disposition record for every idea below — it accepted, rewrote, deferred, or rejected each one after a further adversarial pass (which also caught errors this report repeats uncorrected, e.g. the token-vs-byte forcing conflation and "margin is free" telemetry claims). The "Recommended next steps" section below predates that integration and is superseded. The sibling-plan mining section (§6) remains live input for the next external review round: its Tier-1 mechanisms (claims matrix + documentation CI, in-repo `fnlp-reference` oracle, run receipts, operation-cost registry, G0 executable spikes, deterministic-math/certified reproducibility, complexity-witness locks, dependency displacement map) were **not** absorbed into v3.1 and remain open recommendations.

**Date:** 2026-07-30/31 · **Orchestrator:** YellowRaven (Claude Fable 5)
**Duelists:** `cc` = Claude Code (Fable 5, xhigh effort) vs `cod` = Codex (**gpt-5.6-sol, max reasoning** — per instruction, max not ultra; verified in-pane)
**Scope:** make franken_nlp more compelling, useful, radically innovative, ultra-high-performant, robust/reliable, and accretive — via (A) an adversarial cross-model idea duel and (B) systematic mining of the sibling plans (franken_lean, frankengraphdb, franken_manim).
**Caveat:** the reviewer agents (OrangeHill/NoblePelican) amended the plan/AGENTS/README *concurrently* with this duel (e.g., alien-artifact families renamed AF→AA, new LG-1 licensing gate, AA-A1 added from a duel idea). Section numbers cited below refer to the documents each artifact saw at its moment; reconcile against the current plan when integrating.

---

## Executive summary

Each model generated 30 ideas, winnowed to 8, cross-scored the other's 8 on a 0–1000 scale, reacted to the other's scores (with forced concessions), and answered the blind-spot probe. **Convergence was exceptional — six of eight themes appeared independently in both lists** — which is the strongest validation signal this method produces. The joint winners: **durable/incremental corpus jobs (941/900 — the highest cross-scores in either direction), grounded-by-construction decoding (904/875), user-owned eval/qualification (908/830), document-major multi-task execution with a decide-the-prompt-ABI-now argument both models derived independently (813/800), and selective automation/escalation (820/761)**. One idea was effectively killed (cross-loop wavefront coalescing, 580 — it "solves a problem the plan's synchronized design mostly doesn't have"; survives only as trace instrumentation + a research card). The reveal produced eight genuine concessions (both of cod's confident semantic claims and three of cc's were wrong and caught). The blind-spot probe yielded **six ideas absent from both lists and the plan**, with strong post-duel convergence on the audit-authority gap (acceptance-sampling sign-off for corpus runs).

Separately, mining the three sibling plans produced **36 ranked transferable mechanisms**; the top tier (claims matrix with documentation CI, `fnlp-reference` in-repo oracle, run certificates/receipts, operation-cost registry, G0 executable spikes, deterministic-math + certified reproducibility, dependency displacement map, decision log D-nn) is enumerated in §6 below.

---

## Score matrix

**cod scoring cc's ideas** (cod's weighted rubric: fit 20% / utility 30% / feasibility 25% / complexity-return 25%):

| cc # | Idea | cod score | Verdict |
|---:|---|---:|---|
| 5 | Durable batch contract | **941** | Exceptional; make durable semantics a Phase-4 design commitment |
| 6 | `fnlp eval` (bring-your-own-benchmark) | **908** | Exceptional; productize the evidence machinery |
| 1 | Grounded-by-construction decoding | **904** | Exceptional; core differentiator (claims must stay lexical, not "semantic correctness") |
| 7 | Document-major multi-task execution | **813** | Strong; freeze the prompt-segment ABI now |
| 2 | Calibrated escalation ladder (AF/AA-6) | **761** | Strong concept; aggregation + confidence claims need redesign (see concessions) |
| 8 | `fnlp tune` (per-install measured personalization) | **748** | Strong late-phase feature |
| 4 | Schema toolchain (`infer/check/sample`) | **740** | Split: `check`/`sample` excellent; `infer` underspecified as written |
| 3 | Constraint-pressure telemetry | **678** | Useful diagnostic, misleading "fabrication detector" name — revise before adopting |

**cc scoring cod's ideas** (mean 789):

| cod # | Idea | cc score | Verdict |
|---:|---|---:|---|
| 2 | Durable, incremental corpus jobs | **900** | Best idea in the file; strict superset of cc's #5 (semantic execution keys, cross-run cache, privacy-default journals, exactly-once honesty) |
| 1 | Stencil as an execution compiler (`ProjectLegal`/`FeedForced`/`CopyFromSource`/`FullProjection`) | **875** | Superb frame ("the grammar is an execution planner, not a filter"); `FeedForced` as specified has a trigger-rarity flaw (byte-trie states rarely have exactly one legal *token*) |
| 5 | User-owned qualification + safe activation | **830** | cc's eval + genuinely better rigor (dev/cal/locked-test split manifests with enforced leakage refusal); activation plumbing mildly over-scoped for v1 |
| 6 | Selective automation + correction flywheel | **820** | Escalation core = convergent; the explicit no-hidden-learning correction flywheel is cod's real addition |
| 7 | Exact continuation-trie scoring | **810** | Novel vs cc's set; unlocks large-taxonomy classify + catalog-constrained `resolve`; clean differential-test story |
| 4 | Document-major analyze packs | **800** | Convergent core + better bundle semantics; DAG/tiles/vLLM-style block graph is v2 scope |
| 3 | Bounded declarative task recipes (TaskIR) | **700** | Right shape, wrong time: build TaskIR as *internal* IR now, defer the *public* recipe format past v1 |
| 8 | Cross-loop wavefront coalescing | **580** | Most impressive analysis, but targets fragmentation the synchronized layer-major design doesn't produce; keep as instrumentation + card only |

---

## Consensus winners (adopt; merged designs specified in the reaction files)

1. **Durable, incremental corpus jobs** (cod #2 ⊃ cc #5; 941/900). Content-addressed **semantic execution keys**, transactional item state in fsqlite, `job start/status/resume/verify/materialize`, privacy-default journals (IDs/digests/metrics — never text; spooling explicit), exactly-once scoped honestly (materialized outputs yes, raw stdout at-least-once), cross-run incremental recomputation ("change five documents, recompute five documents"), kill-injection proof obligations at every state transition. cc's contribution folded in: resume-safety as a *provable invariant* via batch-invariance (resumed ≡ uninterrupted). **The single most accretive operational feature in either list.**
2. **Grounded-by-construction decoding** (cc #1 ≡ cod `CopyFromSource`; 904/875). Suffix-automaton-over-source ∩ grammar ∩ SP-transducer product: verbatim-declared fields **cannot** contain off-source bytes; anchoring collapses from heuristic to constructive; defaults ON for ner/redact/citations/keyphrases, per-field opt-in for extract. Merged stance: cc's mechanism depth (mask-cost asymmetry, per-document memos, unsatisfiable-at-entry semantics, e-process invariant) + cod's repeated-occurrence discipline (return all compatible intervals; never fabricate a unique offset) + cod's execution-compiler frame with `ProjectLegal` sparse lm_head projection (exact under masked greedy/conditioned sampling). `FeedForced` demoted to a byte-level jump-forward *research card* (needs token-healing + byte-equivalence gates — cc's trigger-rarity objection stands, cod conceded).
3. **User-owned eval / qualification** (cc #6 + cod #5; 908/830). `fnlp eval/calibrate/qualify` over the user's own labeled data, §9.6 metrics + paired-bootstrap compare + CI assertions, **split manifests with enforced calibration/locked-test leakage refusal** (cod's best single upgrade), qualification as a digest-bound scoped receipt; activation/rollback gating trails in Phase 6.
4. **Document-major multi-task execution** (cc #7 + cod #4; 813/800). **Decide the four-segment prompt ABI now** (both models independently derived the now-or-expensive-later argument — the strongest adoption signal in the exercise). v1 = cc's scoping (fork tree, `--tasks` list, independent branches); cod's eligibility-by-locked-scorecard rule and atomic bundle semantics adopted; packs-as-DAGs + hybrid tiles + block-graph generality deferred to v2.
5. **Selective automation: escalation ladder + correction flywheel** (cc #2 + cod #6; 820/761) — with the duel's corrections applied: **field-level majority voting is unsound as a generic aggregator** (cod's counterexample, cc conceded fully); **cross-sample agreement is not calibration-free confidence** (conceded); signals must be declared valid per task; the flywheel (spill → schema-validated corrections → named local suites → *explicit* refit; no hidden learning) lands later-phase; `--review-out` first.
6. **Exact continuation-trie scoring** (cod #7; 810, unopposed). Shared-prefix dynamic programming over candidate continuations with two explicitly-distinguished probability semantics; unlocks large-taxonomy `classify --labels taxonomy.ndjson` and catalog-constrained `resolve`; compiler falls back to naïve scoring when prefix-sharing is poor.
7. **Constraint telemetry, renamed and always-on-cheap** (cc #3 at 678 — *contested, resolved in the reveal*): cod's objection to the "fabrication detector" name was conceded (it detects **constraint intervention**, not fabrication — the cookie-recipe counterexample: confident fabrication sails through with zero pressure); cc's counter-objection stood: cod's own selective-automation policy *consumes* these signals while its sparse projection removes them, so the two cod ideas undercut each other. **Merged design: always-on margin/override indicators (≈free), sampled full-vocab mass audits (every k-th step + always during evals), honest field name `constraint_interventions`.**
8. **Schema toolchain** (cc #4; 740): `fnlp schema check` + `sample` adopted for Phase 4 (they're the fuzzer + exit-8 path productized); `infer` redesigned per the reveal (needs an intent contract — examples + a one-line task description — plus the holdout self-measurement), Phase 5+.
9. **`fnlp tune`** (cc #8; 748): fold into Phase 6 as the productized AF/AA-5 sweep, speed-only choices among bit-identical paths, both models agreed on deferral.
10. **TaskIR as internal architecture** (from cod #3 at 700): compile all twelve built-ins through one internal IR now (uniformity + hashes + budgets + `recipe explain`-grade introspection); the *public* `.fnlptask.json` format waits until the built-ins have stress-tested the IR (v2) — cc's sequencing, cod largely conceded. cod's **negative-capabilities list** (no code, no network, no template language, no retry-after-failure, no uncalibrated confidence claims) is adopted verbatim as standing doctrine for any future extensibility.

## Killed / demoted

- **Cross-loop wavefront coalescing** (cod #8; 580): the synchronized layer-major sweep already keeps all rows at one `(loop, layer)` wavefront; residual fragmentation (drain tails, cancellations) thins GEMM occupancy at margins, not duplicate weight streams. **Survives only as: loop-stage occupancy trace instrumentation now + a profile-gated research card; implement nothing until histograms prove fragmentation.** (cod's own recommendation, endorsed.)
- Generic field-level majority voting; "agreement = calibration-free confidence"; "fabrication detector" naming; energy metrics "where available"; vLLM-style KV block-graph generality at v1; public recipe format at v1 — all conceded/dropped as specified above.

---

## Blind spots (Phase 6.9) — six ideas neither original list nor the plan contained

**From cc (post-duel):**
- **B1 · The second reader** — entailment verification of *semantic* (non-verbatim) fields: `extract --verify` renders each field as a claim and runs the §7.6 faithfulness check against the source as a second, differently-shaped read; nearly free under the document-major ABI (fork the resident document KV; ~10–15% overhead for 10 fields on a 4K doc); output is a per-field `{entailed|contradicted|unsupported}` envelope block — a calibratable escalation input, never a certificate. Fills the exact hole the duel exposed: grounding covers verbatim fields, pressure telemetry can't see *confident* fabrication, and the fields that must differ from source spelling (dates, totals, categories) were checked by nothing.
- **B2 · Typed trust boundaries** — injection containment as a structural property: Lexicon guarantees untrusted-segment encoding can never emit special/control token IDs (fuzz-provable with L0 machinery; nearly free in Phase 1, expensive to retrofit), plus a §9.6 **injection-invariance eval suite** (attack success rate per task/family, regression-gated) and derived-from-untrusted provenance in envelopes. No detector promised; structural layer absolute, content steering measured.
- **B3 · Acceptance-sampling audit packs** — statistical sign-off for corpus runs: deterministic seeded stratified samples of a run's own outputs sized by a stated risk contract ("detect >2% error at 95% power"), graded by human/stronger-model, `fnlp audit grade` computes the acceptance decision + corrected error CI into the run receipt; graded packs feed the flywheel and eval suites. Zero model compute; sixty-year-old math; answers the operator's actual question ("can I sign off on 100K docs without reading them?").

**From cod (post-duel; notes cc's B3 convergence — "strong post-duel convergence on the audit-authority gap"):**
- **BS1 · Sentinel** — unbiased *production* audits + qualification decay: keyed-hash inclusion sampling over **accepted** results (not just spills — review-only-the-spills is selection-biased and cannot see silent failures among confident accepts), weighted estimators, worst-slice reports, qualification expiry/invalidation, shadow-candidate runs. Complementary to B3: B3 certifies one frozen job; Sentinel monitors the rolling population and the decay of calibration claims.
- **BS2 · Portable corpus snapshots** — scope-correct sharding/merge/lineage: per-document reuse keys are wrong for corpus-global tasks (`resolve` clustering, reduce steps, aggregates); define reuse scopes explicitly, and exploit exact manifests + batch invariance to shard a corpus across several offline CPU hosts and merge deterministically with lineage — no inference network stack added.
- **BS3 · A resident local engine rendezvous** — the loaded 4.7 GB process currently assumes one caller; multiple local agents/tools want to share it (local-socket rendezvous, capability-scoped, budget-arbitrated) without becoming an HTTP server product.

All six fit the doctrine (deterministic, measurable, kill-switchable, no new deps) and are recommended as inputs to the next external review round.

---

## Meta-analysis (biases and dynamics)

- **cc's bias:** mechanism depth and honesty instrumentation — richest on invariants, e-processes, envelope semantics; lighter on operational/infrastructure surfaces. Its weakest moments were *semantic* overclaims (voting, agreement-as-confidence) — precisely where cod's review was sharpest.
- **cod's (Sol Max) bias:** systems/compiler/infrastructure framing — execution planners, IRs, durable state machines, audit planes; its weakest moment was proposing machinery for a problem the plan's own design precludes (wavefront coalescing) and under-valuing a cheap standing instrument (telemetry) that its *own* other idea needed. It was also slower but consistently deeper per artifact (hash-pinned reviews, live-verified citations).
- **The adversarial pressure demonstrably worked:** eight material concessions, two ideas redesigned, one killed, and the blind-spot round produced arguably the most valuable product ideas of the whole exercise (B1/B3/BS1) — none of which either model produced un-pressured.

---

## Sibling-plan mining (the "what else can we take" half)

Three extraction agents read the franken_lean, frankengraphdb, and franken_manim plans in full and returned 36 ranked transferable mechanisms. Deduplicated and curated, the adoption list ordered by leverage:

**Tier 1 — adopt in the next plan revision:**
1. **The claims matrix + documentation CI** (lean §20.4 "Witness"): every public claim a row (`OBSERVED/TARGETED/HYPOTHESIS/…`, repro command, freshness bound, forbidden-stronger-wording); CI rejects doc text stronger than the evidence. The structural fix for the project's #1 named risk (overclaim).
2. **`fnlp-reference`: the in-repo oracle-as-a-program** (fgdb §15.4): a deliberately naive, single-threaded, obviously-correct forward + tokenizer + template + mask evaluator, test-only, CI-enforced dependency wall so the differential can't be gutted by code sharing — survives the day the pinned Python environment rots; turns L1–L5 into `cargo test`.
3. **Run certificates / `.fnlpr` receipts with honest completeness grades** (fgdb §8.6 + lean §8.6): every run replayable-or-graded (`Replayable / StructuralReplay / VerifiableIfArtifactsSupplied / AuditOnly`), pinning recipe/tokenizer/template/grammar/calibration/ISA-tier/threads/seeds; `fnlp replay <cert>`; composes with the duel's audit packs and job receipts.
4. **Operation-cost registry** (fgdb §Appendix G): `token_anatomy.toml` — one row per task class declaring structural counts (prefill tokens, decode steps, **forward passes = exactly 2**, lm_head rows, bytes streamed, KV traffic); CI laws: gates must be derivable from rows; protocol-weight changes update the registry in the same commit. Converts the roofline from advisory prose to machine-checked constitution.
5. **Gate G0: executable spikes before interface freeze** (manim §20.1 / lean §22.1): Phase −1 resolves *documentary* unknowns; add compile-tested spikes for *architectural* ones — mask-oracle cost at 166K vocab, batched-reduction invariance at M∈{1,8,64}, SP-transducer trie build, prefix-fork position stability, loop-boundary forward vs oracle on real weights, AVX2 worst-case on the actual 5995WX — each with a ratification doc + ADR.
6. **Deterministic math layer + certified reproducibility tier** (manim §6.6/§16.7): owned `dmath` transcendentals (platform libm differs across OSes — the unclosed hole in cross-platform bit-identity), `--reproducible` with a content-hashed input closure; **no LLM engine anywhere promises cross-machine bit-identical generation** — the most quotable claim available.
7. **Complexity-witness regression locks** (fgdb §15.6): counted structural bounds (forward passes/token == 2; lm_head rows for classify == |candidates|; trie visits/token) failing CI deterministically — immune to the >5% cv noise floor that hides 4% regressions.
8. **The dependency displacement map** (manim §1.7): the table of what `fnlp` deletes (torch+CUDA wheels → ft-kernel-cpu; transformers+trust_remote_code → native_engine; sentencepiece/tokenizers C++ → Lexicon; outlines/llguidance → Stencil; spaCy+Presidio → ner/redact; cloud APIs → the binary) — the README's strongest opening argument, plus the CI-enforced transitive-closure allowlist.

**Tier 2 — adopt during implementation phases:** per-surface evidence levels L0–L4/R0–R5 on the FeatureUniverse (lean) · typed outcome lattice with first-class `Inconclusive` (lean; already partially landed via reviewers' budget/no-result work) · decision log D-nn + OQ resolution protocol with permanent-refusal list (manim §23) · Behavior Notes + Reference-defect register (manim App. C — harvest HF tokenizer/template/sampler defects during Phase −1) · epoch ratchet for model/tokenizer updates (lean §22.5) · shadow-run promotion matrices with auto-demotion for dispatch/quant/speculative lanes (lean §18.9) · lab-runtime/LDFI/self-shrinking failure testing for the daemon under asupersync (fgdb §15.1–15.3) · keyed per-item RNG substreams — `(seed, request_id, step)` — making sampled batch reproducible by construction (manim §6.5; closes a real hole: seeded batch sampling is currently order-dependent) · mutation campaigns + kill -9 promotion drills (lean §18.2) · transactional artifact promotion for pull/convert (lean §7.3) · RaptorQ repair sidecars + `fnlp scrub` for the multi-GB artifacts (fgdb §5.7; asupersync already ships the primitive; **gated on LG-1**) · one API schema generating CLI/robot/library/FeatureUniverse surfaces (manim §16.2) · `ExecutionPlan` derived from `HardwareTopology`, printed by doctor, hashed into receipts (manim §17.4) · upstream-contributions ledger (tiled GEMM, SP tokenizer, grammar engine → frankentorch/suite; manim §2.9) · sole-constructor types + `DecodeCx` with no ambient env/clock/fs authority (fgdb §12) · out-of-tier ledger with revisit triggers + schema-keyword coverage ratchet from a harvested real-world schema corpus (manim §11.5).

---

## Recommended next steps

1. **Feed this report + the six WIZARD_*.md artifacts into the next external review round** (the plan is mid-review; the reviewers are already amending in place — route these through the same cycle rather than editing around them).
2. Integration order when the round lands: (a) prompt-segment ABI + TaskIR-internal + grounded-decoding amendments to Stencil/Atelier sections (design-stage-cheap, implementation-expensive-later); (b) durable jobs into Conveyor/Phase 4; (c) claims matrix + `fnlp-reference` + token-anatomy registry + G0 spikes into Assay/roadmap; (d) eval/qualification + escalation + telemetry into Phase 5; (e) blind spots B1/B2/B3+BS1 as new plan sections with their honest-boundary language intact.
3. After review: `/beads-br` + `/beads-workflow` conversion with the duel's proof obligations as test beads.
4. The NTM session `franken_nlp` (2 panes) is left alive for follow-up rounds — kill with `ntm kill franken_nlp` when done.

*Artifacts: WIZARD_IDEAS_CC.md · WIZARD_IDEAS_COD.md · WIZARD_SCORES_CC_ON_COD.md · WIZARD_SCORES_COD_ON_CC.md · WIZARD_REACTIONS_CC.md · WIZARD_REACTIONS_COD.md — all in the repo root.*
