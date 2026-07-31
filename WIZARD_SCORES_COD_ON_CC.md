# COD’s scores on `WIZARD_IDEAS_CC.md`

> **Post-review status (2026-07-30): historical cross-score, not specification.** The source hashes below bind the snapshots that were scored; the current idea files have disposition notes and therefore intentionally differ. Plan v3 §10.6 is authoritative. It adopted the corrected raw constraint diagnostics, durable jobs, user evaluation, lexical grounding, and a typed prompt ABI; rejected fabrication/confidence language and impossible two-dimensional KV reuse; narrowed `fnlp tune`; and made schema inference, document-major packs, semantic verification, typed untrusted-document handling, and acceptance audits explicitly evidence-gated. Scores remain useful review provenance, not implementation priority or project truth.

**Status:** adversarial cross-model review of the eight proposals in
`WIZARD_IDEAS_CC.md`, evaluated against the current
`COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md` and compared directly with
`WIZARD_IDEAS_COD.md`.

**Source snapshots reviewed in full:**

- `WIZARD_IDEAS_CC.md`: 323 lines, SHA-256
  `962cf86cb20967c2e20152cff9953ec879283ef0f34150e764da71351384d38e`
- `WIZARD_IDEAS_COD.md`: 773 lines, SHA-256
  `d6afbd49d5283a4a4757d2c54296e8d8f7e8a08dfd637307808860084cdd8bfc`

## Bottom line

This is a strong list. The substantial overlap with my independently produced
list is real evidence of convergence, not a reason to deduct points. Six of
CC’s eight ideas overlap one of my top eight, and the other two were in my
30-candidate winnowing record. The agreement is especially strong around
durable corpus work, user-owned evaluation, source-constrained fields, and
document-major execution.

That does **not** mean all eight treatments are equally sound. CC’s best ideas
are excellent. Its weakest treatment makes a useful runtime diagnostic carry a
much stronger semantic name—“fabrication detector”—than the diagnostic can
justify. The escalation ladder also calls cross-sample agreement
“calibration-free confidence,” which is not true: correlated samples can agree
and still be confidently wrong. The schema toolchain bundles two near-certain
wins with an inference command whose input does not contain enough information
to identify the user’s intended schema.

My cross-score ranking is:

| My rank | CC rank | Idea | Overall score | Verdict |
|---:|---:|---|---:|---|
| 1 | 5 | Durable batch contract | **941/1000** | Exceptional; make the durable semantics a Phase-4 design commitment |
| 2 | 6 | `fnlp eval` | **908/1000** | Exceptional; productize the internal evidence machinery |
| 3 | 1 | Grounded-by-construction decoding | **904/1000** | Exceptional; a core differentiator, with narrower claims than “semantic correctness” |
| 4 | 7 | Document-major multi-task execution | **813/1000** | Strong; freeze the prompt-segment ABI now, promote layouts only by task eval |
| 5 | 2 | Calibrated escalation ladder | **761/1000** | Strong concept, but the aggregation and confidence claims need redesign |
| 6 | 8 | `fnlp tune` | **748/1000** | Strong late-phase feature; useful only with noise-resistant, workload-scoped tuning |
| 7 | 4 | Schema toolchain | **740/1000** | Split verdict: `check`/`sample` are excellent; `infer` is underspecified |
| 8 | 3 | Constraint-pressure telemetry | **678/1000** | Useful diagnostic, misleading detector; revise before adopting |

## Scoring method

I scored the proposals **as written**, not as the best version I can imagine
after silently repairing their weaknesses. Each overall score is the rounded
weighted result of four 0–1000 components:

- **Insight and project fit — 20%:** smartness, differentiation, and alignment
  with the one-model CPU appliance.
- **Real-world utility — 30%:** practical value for humans and AI agents.
- **Correct implementation feasibility — 25%:** whether the feature can be
  made reliable under the plan’s parity, determinism, privacy, and evidence
  rules.
- **Complexity return — 25%:** whether the utility repays the implementation,
  proof, maintenance, and user-facing complexity.

The scale is intentionally strict:

- **900–1000:** exceptional; a no-brainer to pursue, though details may still
  need correction.
- **700–899:** strong; clearly accretive if its stated conditions are met.
- **500–699:** worthwhile kernel of an idea, but material redesign or evidence
  is needed before commitment.
- **Below 500:** costs or risks probably dominate.

| CC idea | Insight / fit (20%) | Utility (30%) | Feasibility (25%) | Complexity return (25%) | Overall |
|---:|---:|---:|---:|---:|---:|
| 1 | 960 | 940 | 820 | 900 | **904** |
| 2 | 850 | 860 | 660 | 670 | **761** |
| 3 | 770 | 680 | 650 | 630 | **678** |
| 4 | 855 | 820 | 620 | 670 | **740** |
| 5 | 900 | 985 | 900 | 960 | **941** |
| 6 | 910 | 945 | 850 | 920 | **908** |
| 7 | 915 | 900 | 670 | 770 | **813** |
| 8 | 810 | 760 | 720 | 710 | **748** |

Overlap itself did not affect a score. The “whose treatment is better” judgment
instead asks which document identified the sharper contract, the more honest
claim boundary, the safer fallback, and the more convincing proof plan.

---

## 1. Grounded-by-construction decoding — **904/1000**

### Candid verdict

This is an exceptional idea and deserves to become part of Stencil’s design
before its grammar IR freezes. It extends the project’s best guarantee in the
most natural direction: a field declared verbatim should not merely produce
valid JSON and then hope that a postprocessor finds the text; off-source bytes
should be outside the accepted language.

The score is not higher because the write-up occasionally slides from
**lexically sourced** to **semantically trustworthy**, and because the
tokenizer/grammar/source product is materially harder than “a textbook suffix
automaton slots into an existing seam” suggests.

### What is especially strong

The schema surface is excellent:

```json
{ "type": "string", "x-fnlp-source": "verbatim" }
```

It is declarative, local to the field, inspectable by agents, hashable with the
task recipe, and naturally rejectable during preflight. The distinction between
`verbatim`, a precisely defined normalized mode, and an ordinary semantic field
is exactly right. Dates, quantities, canonical entity identifiers, and derived
labels must remain free to differ from their source spelling.

The proposal also targets high-value real uses rather than a research demo:

- NER and redaction surface forms;
- evidence quotes and citations;
- keyphrases;
- exact fields in invoice, contract, and support-ticket extraction.

For humans, this prevents a particularly dangerous class of plausible-looking
invented evidence. For AI agents, it turns a result field into a mechanically
checkable capability: an agent can trust the stated lexical provenance without
implementing a fuzzy anchoring recovery loop.

The byte-level composition is compatible with the plan. Stencil already needs
the SentencePiece detokenization transducer because pieces can cross grammar
states. Letting the JSON lexer own escaping while the source automaton consumes
logical, unescaped value bytes is the correct separation of responsibilities.
The independent post-decode substring verifier gives the guarantee a second
authority rather than asking the generation automaton to certify itself.

### The claim boundary that must be tightened

“Grounded by construction” is defensible if it is explicitly defined as
**lexical source membership**. It is not factual correctness.

A constrained model can still:

- choose the wrong invoice number from among several real numbers;
- quote a real sentence that does not support the answer;
- select the wrong occurrence of a repeated name;
- omit the correct field;
- emit a source substring whose interpretation is wrong.

The source automaton prevents invention of bytes. It does not determine which
bytes are semantically correct. Marketing and envelopes should say
`source_membership: guaranteed`, not imply that correctness or entailment was
proved.

Likewise, offsets are not entirely “for free.” A suffix automaton proves that a
string occurs, but repeated occurrences preserve an `endpos` set rather than a
unique source interval. “Nearest to hint” is still a selection heuristic.
Ambiguity should remain explicit, or all compatible intervals should be
returned. The existing project doctrine is right not to turn ambiguity into a
fabricated unique offset.

`verbatim-normalized` also needs two coordinate spaces named in the contract:
the normalized byte language used during decoding and the original source
coordinates returned to the caller. If normalization is not reversibly mapped,
the result cannot honestly promise an original byte span.

### Implementation and complexity judgment

The suffix automaton itself is ordinary. The product construction is not:

- bounded source-index representation for adversarial large documents;
- tokenizer tokens that emit multiple logical bytes;
- JSON escapes and Unicode/byte-fallback behavior;
- liveness under `minLength`, enums, nullability, and output budgets;
- per-document mask-cache accounting;
- cancellation and batching with sequence-local automata;
- independent occurrence recovery.

Those are tractable because Stencil already owns most adjacent machinery, and
the utility easily justifies them. I would still phase it narrowly:

1. byte-exact `verbatim` strings;
2. explicit ambiguity sets and an independent verifier;
3. task defaults only after L5 evaluation;
4. normalized modes only after their coordinate mapping is fully specified.

An unsatisfiable required non-null field should fail the task with a typed
no-result, not silently weaken the constraint. A nullable field may emit null
only if null is already legal under the schema.

### Overlap with my ideas; whose treatment is better?

This directly overlaps my Idea 1, “Turn Stencil from a token-mask engine into a
grounded execution compiler.” Both independently chose a source-substring
automaton, logical unescaped bytes, an `x-fnlp-source`-style annotation, typed
unsatisfiable behavior, and independent proof.

The treatments have different strengths:

- **CC is better on the focused grounding pitch.** It isolates one highly
  legible product promise and gives it a clean schema/CLI surface.
- **COD is better on the larger execution architecture.** My treatment keeps
  the legal token IDs for sparse `lm_head` projection, turns deterministic
  token runs into causal micro-prefills, states that full-vocabulary mass is
  unavailable on sparse paths, and is more explicit about repeated-occurrence
  ambiguity.

There is no sensible reason to choose one document wholesale. The best plan
uses CC’s narrow user contract as one instruction in COD’s broader Stencil
execution IR. On the grounding subproblem alone, CC’s presentation is slightly
better; on implementation and proof boundaries, COD’s is stronger.

**Disposition:** adopt the contract now; implement it after the base
grammar/tokenizer product is correct; never market lexical membership as
semantic correctness.

---

## 2. AF-6 calibrated escalation ladder — **761/1000**

### Candid verdict

The strategic idea is strong: do not force every document through the most
expensive path, and do not pretend a 3B model has no hard tail. A measured
accept/retry/review policy is exactly how this model becomes useful in
consequential batch work.

The proposed ladder is less ready than the write-up claims. Tier 3 is not
well-defined generically, agreement is not calibration-free confidence, and
the design can absolutely lose if a universal controller adds compute and
false reassurance without reducing task risk.

### What is strong

The product posture is excellent:

```text
cheap local attempt → measured extra compute → explicit spill or abstention
```

The spill file is particularly pragmatic. `fnlp` remains offline and never
calls a cloud model, while humans and agents receive a structured hard-case
queue carrying source IDs, outputs, signals, hashes, and reasons. That makes the
3B engine a valuable first-stage triage appliance even when another system owns
the tail.

The proposal also correctly requires the alien-artifact contract:

- explicit signals and actions;
- a task/user-specific loss matrix;
- fitted thresholds;
- a deterministic disabled fallback;
- evidence receipts containing realized tier counts and costs.

An accuracy-versus-cost curve is far more useful than a vague
`--quality balanced` promise. It makes the choice inspectable and gives agents a
stable policy artifact rather than hidden orchestration.

### The weak point: self-consistency is not one generic operation

G8 guarantees that every sample is syntactically valid. It does **not** make
field-level majority voting semantically valid.

Consider:

- arrays of entities whose ordering and cardinality vary;
- nested objects with correlated fields;
- a date and amount that must come from the same transaction;
- free strings that differ by harmless spelling;
- evidence spans whose boundaries must be aligned before voting.

Independent field majorities can assemble an object that no sample produced,
break cross-field meaning, and even violate deterministic postconditions. An
exact-string majority can fragment all votes and provide no winner. Every task
therefore needs either a frozen task-specific aggregator, a whole-result medoid
or selector, or no consistency tier at all.

More importantly, **sample agreement is not calibration-free confidence**.
Samples from the same prompt and model are highly correlated. They can agree
because the model has one stable misconception. Agreement is a useful feature
whose relationship to correctness must be measured on locked labels; it is not
itself a confidence guarantee.

Thinking retries are similarly empirical. The master plan correctly makes
thinking off for structured/batch tasks until a task-specific scorecard shows
that quality gains repay p95 latency, energy, and token cost. Document-prefix
forking may make a retry cheaper, but only for prompt layouts that survive the
quality gate.

### Complexity and implementation judgment

The runtime state machine is small. The real cost lives in:

- per-task signal semantics;
- per-task aggregation;
- disjoint calibration and locked-test data;
- queue re-entry and resource admission for retries;
- stable review/spill schemas;
- selective-risk and full-cost evaluation;
- preventing stale policy artifacts from authorizing a changed recipe.

Those costs are justified for tasks where errors matter. They are not
“near-zero novel machinery,” and “the design cannot lose” should be removed.
The deterministic-off fallback limits harm, but a controller still consumes
implementation and maintenance budget.

### Overlap with my ideas; whose treatment is better?

This overlaps my Idea 6, “Selective automation plus a correction flywheel.”
Both use calibrated signals, thinking/sample escalation, abstention, and an
offline review/spill boundary.

CC’s tier diagram and escalation-curve story are clearer and easier to explain.
COD’s treatment is materially stronger technically:

- it says signals absent on a fast path are `not_computed`, never zero;
- it limits signals and actions per task;
- it avoids treating agreement as calibrated correctness;
- it defines deterministic policy replay;
- it adds review import, correction-data isolation, and an explicit
  recalibration/policy-fit flywheel without hidden online learning.

**Treatment verdict: COD is better.** CC supplies the better one-paragraph
product narrative, but the generic field-voting and confidence claims are
unsafe enough to matter.

**Disposition:** adopt a versioned `DecisionPolicy` concept, spill/abstain first,
and promote each retry/aggregation tier separately per task. Do not ship one
universal ladder.

---

## 3. Constraint-pressure telemetry — **678/1000**

### Candid verdict

There is a valuable instrument here, but the proposal gives it a name and
semantic interpretation it has not earned. Constraint pressure can measure
**how strongly constrained decoding overruled the unconstrained next-token
preference**. It cannot, by itself, detect fabrication.

As written, this is the only idea below 700 because a `fabrication_risk` flag
could make users more confident for exactly the wrong reason.

### What is genuinely useful

The plan should instrument constrained decoding. In the current full-vocabulary
generation path, one fused scan can cheaply retain:

- best legal logit;
- best illegal logit;
- whether the unconstrained argmax was illegal;
- the grammar/field state in which intervention happened.

When a full-vocabulary log-sum-exp is intentionally computed, true probability
mass on the legal language is also informative. Per-field aggregation is much
better than one opaque run-level number. These diagnostics can:

- expose a schema or enum that fights the model;
- identify copy-constrained fields where the model preferred off-source text;
- compare prompt revisions;
- feed a task-specific escalation policy after calibration;
- reveal release regressions in how often masks materially change trajectories.

This is useful to humans tuning schemas and prompts, and extremely useful to AI
agents because it is structured, local, and machine-routable.

### Why it is not a fabrication detector

Constraint pressure and factual correctness are different variables.

- A low-pressure model can confidently emit the wrong but schema-legal invoice
  number.
- A high-pressure step can be benign JSON punctuation or a canonical object key
  the model would have formatted differently.
- A narrow enum may correctly prevent an attractive synonym.
- A source-copy constraint may force the correct spelling even when the model
  preferred a normalized spelling.
- A model can assign high mass to a legal language containing many wrong
  values.

The grammar knows syntax and, for copy fields, lexical membership. It does not
know whether a chosen legal value answers the document correctly. Therefore:

- rename the envelope field to `constraint_intervention` or
  `projection_pressure`;
- separate structural-scaffold states from semantic-value states;
- expose raw components and `not_computed`;
- call any error-risk mapping “calibrated on dataset X,” with reliability
  evidence;
- never publish an e-process “fabrication rate” derived from pressure alone.

The proposed fixture where pressure “must” be high on an off-topic document is
also not a proof obligation. A model can confidently invent a legal flight
number from a cookie recipe and show low pressure. That is precisely why
pressure cannot stand in for labeled correctness.

### Performance caveat

The current plan’s full `lm_head` scan makes legal/illegal maxima cheap to
retain. That changes if Stencil adopts sparse legal-row projection. A sparse
path never computes illegal logits and cannot report a legal-vs-illegal margin
or full legal mass. It must emit `not_computed` or run an explicitly priced
audit projection.

Even on the full path, “zero cost” should mean “no additional vocabulary pass,”
not literally zero instructions. True legal mass needs a full denominator and
stable log-sum-exp; the existing candidate-conditional fast paths cannot
manufacture it.

### Overlap with my ideas; whose treatment is better?

This overlaps the telemetry in my Ideas 1 and 6. I deliberately did not elevate
it as a standalone top-eight feature because the full-mass signal disappears
on the sparse path and because it must not be presented as correctness.

CC deserves credit for seeing that constraint intervention should be
first-class, per-field telemetry. That is better productization than my brief
treatment. COD’s treatment is better on the critical honesty boundary:
full-vocabulary audits are explicit, unavailable signals remain
`not_computed`, and pressure is only one calibrated policy feature.

**Treatment verdict: COD is safer; CC is more visible.** A merged, renamed
version would be strong. The current “fabrication detector” framing is not.

**Disposition:** adopt raw projection-pressure telemetry, reject the
fabrication label, and require labeled calibration before mapping pressure to
error risk.

---

## 4. `fnlp schema infer | check | sample` — **740/1000**

### Candid verdict

This is three ideas with very different evidence:

- `schema check` is a near-certain win.
- `schema sample` is a strong, cheap developer/agent tool.
- `schema infer` is compelling but does not yet have a sufficient input
  contract.

The bundle earns a strong score because the first two are excellent and the
third could become useful after redesign. It does not earn an exceptional score
because “five example documents in, the intended extraction schema out” is not
a well-posed problem.

### Why `check` and `sample` should ship

`fnlp schema check` exposes an operation the engine must already implement:
compile the supported subset or return the exact unsupported keyword and
resource estimate before model load. It belongs in CI, editor integrations,
agent workflows, and `robot` automation. It makes the finite subset feel like a
real contract instead of a limitation discovered at inference time.

`fnlp schema sample` usefully repackages the grammar fuzzer’s accepting walk.
It lets users see:

- optional-key combinations;
- enum and array behavior;
- canonical key/number formatting;
- practical size bounds;
- surprising shapes admitted by the schema.

Sampling does not prove that a schema expresses the user’s intent, and random
walk distributions need to be documented, but it is a high-leverage diagnostic
with little novel runtime machinery.

### Why raw-document schema inference is underspecified

A document does not contain one intrinsic extraction schema. An invoice may
contain seller identity, line items, tax jurisdiction, bank details, shipping
terms, signatures, and legal boilerplate. One user wants `{invoice_id,total}`;
another wants line-item tax reconciliation. Five raw invoices do not tell the
model which question the user intends to ask.

The meta-schema grammar guarantees only that the output is **compilable**. That
closure property is elegant, but it says nothing about whether:

- the chosen fields are useful;
- optionality was inferred correctly;
- an enum is a real closed set rather than sparse observed values;
- two differently named fields represent the same concept;
- absent values are truly optional rather than extraction misses.

The proposed holdout test is partly circular. Running the inferred schema
against withheld raw documents and reporting field coverage or constraint
pressure does not measure field accuracy without desired outputs. A model can
consistently extract an irrelevant self-invented schema.

A sounder interface needs at least one source of intent:

- `--goal "capture payment and shipping terms"`;
- a user-supplied seed field list;
- paired `{text, desired_output}` examples;
- or a draft schema to widen/check.

The result should be explicitly labeled a **draft**, report disagreements and
low-evidence inferences, and prefer permissive optionality over fabricated
certainty. Enum closure should require user confirmation.

### Complexity judgment

Depth-unrolling a meta-schema for the supported finite subset is feasible.
Deterministic merging is implementable. The hard part is not grammar
compilation; it is defining and evaluating semantic inference. That part adds
prompt, merge-policy, provenance, and task-eval surface that CC’s “one preset
plus one merge module” estimate understates.

The pragmatic plan is to split delivery:

1. ship `check`, `sample`, and resource/explain output with Stencil;
2. prototype `infer` only with explicit intent or paired examples;
3. gate promotion on held-out desired-output accuracy, not structural coverage.

### Overlap with my ideas; whose treatment is better?

The tooling overlaps my Idea 3, the bounded declarative task-recipe compiler:
`recipe check`, `recipe explain`, and `recipe sample` all reuse the same
compiled grammar/TaskIR machinery. My winnowing appendix also named
`schema infer/check/sample` a strong near-winner and explicitly recommended
waiting on inference quality.

The comparison is split:

- **CC is better on schema-specific DX.** It gives `check` and `sample` a
  memorable first-class surface and explains the meta-schema closure well.
- **COD is better on roadmap judgment and extensibility.** A bounded TaskIR
  serves custom tasks beyond schemas, and deferring inference avoids treating
  successful compilation as usefulness.

For `check`/`sample`, CC wins. For `infer` as proposed, COD’s decision to defer
is better. Overall, neither treatment should replace the other: the schema
commands should be the first tooling slice of the TaskIR, while inference stays
experimental until its intent contract is repaired.

**Disposition:** adopt `check` and `sample`; redesign and separately evaluate
`infer`.

---

## 5. Durable batch contract — **941/1000**

### Candid verdict

This is the best idea in CC’s list by overall project value. It is not the most
novel, and that is part of why it scores so highly. It converts already planned
pieces—bounded queues, fsqlite, hashes, deterministic per-document output, and
cancellation—into the operational contract required by the project’s central
“100K documents overnight” use case.

A fast batch system that loses its progress at 3 a.m. is not an
ultra-high-throughput product in practice.

### What is especially strong

All five user-visible pieces solve recurring real problems:

- durable per-document state;
- explicit input spooling;
- exact-configuration resume refusal;
- out-of-band heartbeat/progress events;
- a hash-stamped end-of-run receipt.

Humans get safe recovery and an inspectable provenance record. Agents get
stable document IDs, machine-readable status, a deterministic way to resume,
and an artifact they can verify before advancing a larger workflow.

This idea also compounds every future performance gain. Kernel optimization
saves minutes on one run; durable incremental state can save hours after a
failure. The model’s two-loop architecture makes every repeated token
expensive, so refusing needless recomputation is especially valuable.

The receipt is not decorative reporting. For redaction, compliance intake,
classification, or corpus conversion, it is the durable statement of which
artifact, prompt, recipe, thinking mode, policy, and inputs produced which
outputs.

### Corrections needed for a production-grade contract

The current plan deliberately makes local text/result persistence disabled by
default. `--spool` must therefore be explicit and privacy-affecting, with:

- owner-only permissions;
- preflighted disk budgets;
- retention/purge behavior;
- no reversible redaction map;
- an inspectable statement of what content is retained.

`output_offset` alone is not a sufficient crash protocol. A process can die
between writing output bytes and committing the journal, or vice versa. The
owned output should use checksummed/framed item records and a result digest,
then materialize the final NDJSON from committed records. Recovery must detect
torn tails.

Exactly-once language also needs scope:

- an owned spool plus canonical materialization can produce one committed
  record per item;
- raw stdout across a crash and an arbitrary downstream consumer cannot promise
  exactly once; it needs stable IDs and documented at-least-once replay.

Likewise, resumed equality should compare canonical semantic result bytes.
Timing fields, run IDs, completion order, and retry counts legitimately differ.

These are amendments to an excellent idea, not reasons to reject it.

### Complexity judgment

This is medium, bounded infrastructure complexity with unusually high testability:

- kill injection at every transition;
- torn-frame and disk-full fixtures;
- semantic-key mismatch tests;
- uninterrupted versus resume equivalence;
- privacy-schema assertions;
- ordered materialization determinism.

The utility overwhelmingly justifies the work. It should not be postponed to
“hardening”; its state model needs to shape the batch implementation from the
start.

### Overlap with my ideas; whose treatment is better?

This directly overlaps my Idea 2, “Make corpus work a durable, incremental
job—not an immortal pipe.”

CC provides the cleaner minimal product amendment and a very good receipt/
heartbeat story. COD’s treatment is better by a meaningful margin on the
load-bearing semantics:

- a separate `fnlp job` surface leaves lightweight `fnlp batch` composable;
- an explicit semantic execution key covers every input that can change an
  output;
- item transitions distinguish admitted, running, committed, and materialized;
- stdout and owned-spool delivery promises are separated;
- content persistence is opt-in and privacy-tested;
- crash framing and digest recovery are explicit;
- unchanged inputs can be reused on later corpus runs, not just one interrupted
  run.

**Treatment verdict: COD is better technically; CC is an excellent MVP.** The
ideal plan adopts COD’s state/identity/privacy contract and CC’s concise
heartbeat and receipt presentation.

**Disposition:** commit to the durable state model in Phase 4 and make kill/
resume equivalence a release invariant.

---

## 6. `fnlp eval`: bring-your-own benchmark — **908/1000**

### Candid verdict

This is an exceptional and highly pragmatic feature. The plan already requires
Assay to implement task-specific metrics, immutable datasets, recipe/prompt
hashing, and locked scorecards. Exposing a disciplined subset to users turns
internal quality engineering into one of the product’s strongest trust and
adoption features.

The score is lower than the durable-batch score because its direct audience
needs labeled data and because maintaining stable adapters/metrics for the
whole task portfolio is more work than “10% incremental” suggests.

### Why it is compelling

The right answer to “will this 3B model work on my contracts/tickets/entities?”
is not another project benchmark. It is:

```bash
fnlp eval --task classify --dataset tickets.ndjson --gold label
```

with a receipt binding the result to:

- dataset and item digests;
- model/quant recipe;
- task and prompt hashes;
- thinking/numerics mode;
- metric version;
- exact comparison candidates.

`--compare` and `--assert-min` are particularly valuable for both humans and
agents. They turn prompt/schema/preset changes into measured CI decisions. An
agent can propose a recipe edit, run the locked local suite, and report a
machine-checkable keep/revert result rather than an anecdotal sample.

The feature also strengthens the project’s honest positioning. It lets users
measure the model ceiling on their data before automating a consequential
workflow.

### Where CC understates the work

The internal harness and a stable public tool are related but not identical.
The product surface must define:

- versioned gold shapes for every task;
- duplicate and split-leakage detection;
- exact/relaxed matching and normalization contracts;
- behavior for small or imbalanced datasets;
- paired resampling that preserves item pairing after resume;
- calibration versus locked-test separation;
- `INSUFFICIENT_DATA` rather than impressive-looking unstable intervals;
- resource bounds for large local suites;
- forward rejection when a dataset adapter version is unknown.

Paired bootstrap is feasible without a new dependency, but it is still
statistical code that needs golden fixtures and independent checks. “Distributed,
unfakeable evidence base” is too grand: users can cherry-pick data or
misconfigure gold fields. The receipts make claims reproducible and scoped,
not inherently representative.

None of these concerns defeats the feature. They define what makes it worthy of
the project’s honesty doctrine.

### Overlap with my ideas; whose treatment is better?

This overlaps my Idea 5, “Productize Assay as user-owned qualification,
calibration, and safe activation.”

CC’s treatment is arguably the better **first release slice**: one clear
`fnlp eval` surface, comparison, scorecard, and CI assertions. COD’s treatment
is better as the complete lifecycle:

- disjoint development/calibration/locked-test manifests;
- `calibrate` and `qualify`;
- digest-bound policy decisions;
- candidate installation separated from activation;
- `--require-qualification`;
- exact rollback of weights, prompt, calibration, and recipe together;
- substitution and split-leakage tests.

**Treatment verdict: COD is more complete; CC is more immediately pragmatic.**
The best sequencing is CC’s `eval` first, then COD’s qualification/activation
layer once artifact management is real.

**Disposition:** add the public dataset/scorecard contract while building
Assay, then make it the prerequisite for calibration and safe upgrades.

---

## 7. Document-major multi-task execution — **813/1000**

### Candid verdict

This is a strong, design-stage-sensitive amendment. Multi-task document
pipelines are a natural use of the planned portfolio, and repeatedly prefilling
the same long document is exactly the sort of structural waste a CPU-first
engine should remove.

The proposed speedup is plausible, but only for tasks whose prompt layout
retains quality and only where the token sequences have an actual common
prefix. A causal KV cache cannot generally factor a document × task grid in two
dimensions.

### Why it is valuable

The user workflow is compelling:

```bash
fnlp multi --tasks ner,redact,extract,summarize document.txt
```

or, preferably, one versioned analysis pack. Humans avoid five invocations and
five inconsistent envelopes. Agents receive one source coordinate system, one
provenance root, and per-task success/error objects.

For a long document and several short task tails, one eligible document prefill
can save several full traversals through all 44 effective layer executions. The
same fork also makes thinking retries or multiple bounded suffixes cheaper.
This is a genuine model-execution win, not CLI sugar.

The prompt-segment ABI should be decided now because prompt bytes, role layout,
and hashes become scorecard inputs. Retrofitting explicit segments after all
built-in prompts are frozen would trigger broad re-fixturing.

### The causal-prefix limitation

To share document KV across tasks, the exact token prefix through the document
must be identical:

```text
[common system policy] [document] [task-specific instruction/tail]
```

If CC’s `[task-static preamble]` differs by task, the branches do **not** share
the document prefix. It must be a genuinely common system segment, not merely a
static segment for each task.

More generally, one causal sequence cannot simultaneously obtain arbitrary
two-dimensional reuse:

- task-first ordering shares a task prefix across documents;
- document-first ordering shares a document prefix across tasks;
- a common-prefix DAG can exploit identical segments;
- it cannot splice a task KV prefix and an independently computed document KV
  prefix together when their causal token order differs.

A scheduler may choose corpus-major, document-major, or mixed cohorts, but the
plan should not promise both reuse directions for every cell of one M×K grid.

### Quality, memory, and reliability obligations

Instructions-last is an empirical candidate, not a universal improvement. The
pinned chat template and role order constrain how it can be expressed, and
recency intuition is not a quality proof. Each task needs a locked A/B
scorecard. Ineligible tasks should still participate in the bundle through a
cold independent path.

The 44-deep KV makes shared prefixes valuable but also makes retained forks
expensive. Admission must price:

- the common document pages;
- every branch tail;
- task grammars and output budgets;
- concurrent batch cohorts.

The advertised 3.5–5× prefill arithmetic should be reported only for eligible
tasks and separated from tokenization/index reuse and scheduler batching.

### Overlap with my ideas; whose treatment is better?

This is almost the same idea as my Idea 4, “Document-major `analyze` packs.”
That independent convergence is strong evidence that the prompt ABI deserves
an explicit plan decision.

CC is better at the compact arithmetic and “cheap now, expensive later”
argument. COD is better on the implementation contract:

- four explicit prompt segments with a truly common global policy;
- task-by-task layout eligibility and cold fallback;
- content-addressed common-prefix graphs rather than implied 2-D KV
  factorization;
- analysis-pack DAGs and `blocked_by` semantics;
- one atomic bundle with independent branch errors;
- privacy namespaces and separately attributed benchmarks.

**Treatment verdict: COD is better technically; CC is better rhetorically.**

**Disposition:** freeze a segmented prompt ABI now, implement analysis packs in
Phase 4, and promote document sharing per task only after exact fork parity and
quality A/B gates.

---

## 8. `fnlp tune`: per-install measured personalization — **748/1000**

### Candid verdict

This is a strong Phase-6 productization of the measured-dispatch doctrine. It
is safe in principle because it selects only among bit-identical paths, and it
could recover meaningful performance on machines unlike the reference hosts.

Its magnitude is uncertain, and several proposed knobs are workload- or
operating-state-specific rather than properties of a permanent CPU signature.
The feature is worthwhile if it starts narrower than the write-up suggests.

### What is strong

The user experience is excellent:

```text
tuned: +14% docs/min on this machine
```

or:

```text
tuning found no statistically credible improvement; defaults retained
```

That makes the project’s “measured faster wins” rule visible on the user’s own
hardware. `robot backends` can state exactly which profile selected which
kernel and why. AI agents can run a bounded tune, preserve its receipt, and
reproduce or disable the choice.

Restricting tuning to bit-identical execution paths is the correct hard line.
A local tuner must not alter quantization algebra, reduction order, calibration,
or task numerics. Artifact/binary/CPU mismatch should ignore the profile and
fall back to shipped defaults.

### What makes this harder than an in-binary benchmark wrapper

Microbenchmark winners are vulnerable to:

- thermal state and frequency throttling;
- background load and OS scheduling;
- Apple P/E placement and QoS;
- NUMA placement and available memory channels;
- container/cgroup CPU limits;
- cold versus warm page cache;
- measurement noise smaller than the claimed win.

The profile key may need OS/kernel, topology, memory-placement, and scheduler
context in addition to CPU signature. More importantly, some choices should
not be stored as one host-wide constant:

- kernel implementation per fixed shape is a good stable tuning target;
- latency thread caps can be stable within a declared environment;
- batch-M depends on context length, task mix, KV budget, and latency SLO;
- mmap versus owned loading depends heavily on page-cache and workload state.

A two-minute quick sweep can easily overfit noise. Promotion needs repeated
samples, warmup, a minimum effect size above confidence/noise, hysteresis, and a
“defaults retained” bias. A full sweep should test end-to-end representative
regimes, not only tiny kernels.

Shipping benchmark control logic in the binary also creates a maintenance
surface, though the underlying kernels and timing harnesses already exist. That
cost is justified after the static reference dispatch is mature, not before.

### Overlap with my ideas; whose treatment is better?

I considered `fnlp tune` among my 30 candidates and called it a strong
near-winner for Phase 6, but it did not make my top eight. CC’s treatment is
therefore plainly better and more complete for this feature.

There is only thematic overlap with my Idea 8, cross-loop physical-layer
wavefront coalescing:

- CC’s tuning idea is less novel but far more practical and lower risk.
- COD’s wavefront idea is more model-specific and potentially more
  differentiated, but it should not proceed without traces proving loop-stage
  fragmentation.

If choosing implementation priority today, I would place CC’s `tune` ahead of
COD’s wavefront experiment. That is a genuine point in CC’s favor. My top-eight
ranking rewarded potential architectural leverage; CC’s choice better rewards
near-term certainty.

**Treatment verdict: CC wins.**

**Disposition:** keep it in Phase 6; initially tune per-shape bit-identical
kernels and conservative thread caps, then add workload-sensitive choices only
when the profile schema can express their validity domain.

---

## Overall overlap and synthesis

| CC idea | Closest COD idea | Convergence | Better treatment |
|---:|---|---|---|
| 1. Grounded decoding | COD #1, Stencil execution compiler | Very high | Split: CC on focused UX; COD on execution/proof |
| 2. Escalation ladder | COD #6, selective automation/flywheel | Very high | **COD**, due task-specific aggregation, calibrated signals, and correction lifecycle |
| 3. Constraint pressure | Parts of COD #1 and #6 | High | **COD** on honesty; CC on visibility |
| 4. Schema toolchain | COD #3 plus COD near-winner | Medium-high | Split: **CC** for `check/sample`; **COD** for deferring underspecified `infer` |
| 5. Durable batch | COD #2, durable incremental jobs | Very high | **COD**, due semantic keys, privacy, crash protocol, and incremental reuse |
| 6. `fnlp eval` | COD #5, qualification/calibration/activation | Very high | **COD** overall; **CC** is the better first slice |
| 7. Document-major | COD #4, `analyze` packs | Near-identical | **COD**, due exact prompt/fork and bundle semantics |
| 8. `fnlp tune` | COD near-winner; thematic COD #8 | Low direct overlap | **CC** |

### What I would carry into the master plan

The combined review suggests four different dispositions rather than treating
all eight as one feature batch:

1. **Design commitments now**
   - field-level lexical source constraints in Stencil’s IR;
   - a segmented prompt ABI that can express document-major layouts;
   - durable job identity/state semantics alongside the batch engine.

2. **High-confidence product surfaces**
   - `fnlp schema check` and `schema sample`;
   - public `fnlp eval` scorecards and CI assertions;
   - status streams and end-of-run receipts.

3. **Evidence-gated Phase-5/6 features**
   - per-task escalation policies and task-specific aggregation;
   - safe qualification/activation around user evals;
   - conservative per-install tuning.

4. **Ideas requiring semantic correction before adoption**
   - projection pressure must not be called a fabrication detector;
   - schema inference needs explicit user intent or desired-output examples;
   - sample agreement must not be called calibration-free confidence;
   - a causal KV cache must not promise impossible two-dimensional prefix
     factorization.

The three strongest consensus investments are therefore:

1. **durable, exact corpus jobs;**
2. **user-owned evaluation and qualification;**
3. **lexically grounded structured fields.**

Those three make `franken_nlp` faster in the only sense that ultimately
matters—useful work survives—more trustworthy on the user’s own domain, and
mechanically safer at the output boundary. They are ambitious without relying
on speculative model behavior, and their complexity is paid back across nearly
the entire task portfolio.
