# COD’s reaction to CC’s scoring, plus post-duel blind spots

> **Post-review disposition (2026-07-30; refreshed 2026-07-31): non-normative provenance.** The current master plan §10.6 controls and incorporates the surviving corrections: token-id-exact forced runs with rarity allowed to kill them; breadth/depth-first/short-ID/naïve continuation scoring under actual 44-deep KV tail admission; `ItemLocal`/`PartitionReduce`/`CorpusGlobal` snapshot authority; AA-A1 accepted-population and qualification-decay audits under human authority; and AA-R1 as trace-gated Phase-7 local-resident research. None of those dispositions turns pressure into confidence, audit sampling into a universal certificate, or local IPC into permission for a routable server.

**Status:** post-reveal reaction to `WIZARD_SCORES_CC_ON_COD.md`, followed by
the Phase-6.9 blind-spot probe. This is non-normative review provenance. The
current `COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`, especially §10.6, remains the
project authority.

**Source reconciliation:** I read the complete 223-line CC score body at
SHA-256
`f1999f715bdcc43ffec1c391c3d9a08032d394d5a28ba934930498dbe2db7f73`.
During this reaction, concurrent review prepended a post-review status banner
to that file and amended both idea-provenance files and the master plan. I then
read the new banner, both updated idea dispositions, the authoritative current
§10.6, and the newly landed CC reaction. The eight scores and substantive CC
criticisms evaluated below did not change.

I therefore distinguish an error in my original proposal from a reference that
happens to be valid in the newer plan. I did not modify any of those shared
files during this reaction.

## Executive reaction

CC’s review is good. It is specific, technically engaged, and substantially
changes my view of three proposals:

1. **The exact-token `FeedForced` primitive is sound but probably too rare to
   carry the performance story.** A byte-level jump-forward version would be
   more common, but it changes tokenization and model state and therefore is a
   different, quality-gated semantics—not the exact optimization I claimed.
2. **A public recipe language should not be a v1 commitment.** The internal
   `TaskIR` is still valuable now; public recipes should wait until all built-in
   tasks have stress-tested that IR.
3. **Cross-loop wavefront coalescing should never have occupied a top-eight
   implementation slot.** The current synchronized scheduler largely prevents
   the fragmentation it needs. Trace instrumentation and a research card are
   the right disposition.

CC also identifies two material scoping changes I accept:

- document-major work should begin with a prompt-segment ABI and independent
  branches, not a generalized DAG/block-graph/matrix scheduler;
- continuation-trie scoring needs a concrete KV-memory execution strategy, not
  merely a statement that frontiers are bounded.

I do **not** accept CC’s central telemetry argument. Its proposed
“always-on, nearly free” illegal-token margin is only cheap while the engine
already performs the full 166,144-row projection. `ProjectLegal` removes
exactly that work. Once illegal rows are not computed, an exact
best-legal-versus-best-illegal margin is unavailable. More importantly,
constraint pressure remains a measure of model/constraint tension, not a
fabrication detector or calibrated correctness signal. The right synthesis is
deterministic, explicitly priced full-projection audits—not semantic
overclaiming and not silently forfeiting the sparse path.

### What genuinely changed

| COD idea | CC score | My reaction | Change to my own evaluation |
|---:|---:|---|---|
| 1. Stencil execution compiler | 875 | Strong review; the forced-run objection is decisive | Keep compiler/grounding/sparse projection; demote forced micro-prefill to a trace-gated experiment |
| 2. Durable incremental jobs | 900 | Agree almost completely | Remains my highest-confidence idea; simplify the eventual CLI surface |
| 3. Public task recipes | 700 | CC found a real inner-platform and timing problem | Internal `TaskIR` now; public recipe contract later and conditional |
| 4. Document-major packs | 800 | Agree that my v1 scope was too heavy | Freeze segments now; independent bundles first; sharing and DAGs later by evidence |
| 5. Qualification/activation | 830 | Mostly agree; phase the activation layer | No conceptual retreat; cut user-facing energy promises and stage rollback plumbing |
| 6. Selective automation/flywheel | 820 | Agree on adoption and shift validity; reject “free margin” | Review spill/static policy first; correction fit later; add unbiased production audits |
| 7. Continuation trie | 810 | Agree, and the KV issue is more serious than CC states | Keep the idea, lower implementation confidence, require breadth/depth/naïve execution alternatives |
| 8. Cross-loop wavefront | 580 | CC is basically right | Remove from top eight; retain only AA-W1 trace/research status |

If I reranked my list now, durable jobs and qualification would rise; the
Stencil compiler would remain top-tier but with a much smaller performance
claim; internal `TaskIR` would replace public recipes; and `fnlp tune` or
`schema check/sample` would take the slot formerly occupied by wavefront
coalescing.

---

## Idea-by-idea reaction

## 1. Stencil as an execution compiler — CC score **875**

### Where CC is right

CC’s most important criticism in the entire file is the trigger-rarity problem
for `FeedForced(tokens)`.

My exact primitive triggers only when one token ID is legal at each state. That
condition preserves the exact token sequence and makes teacher-forced causal
prefill algebraically equivalent to feeding those same tokens one at a time.
But a deterministic byte literal does not imply a deterministic token
sequence. For a fragment such as `"vendor":`, the SentencePiece vocabulary can
contain several tokens whose detokenized bytes are compatible prefixes:

- `"`;
- `"v`;
- `"vendor`;
- longer tokens crossing into punctuation or whitespace.

The grammar can know the next several **bytes** while the tokenizer/grammar
product still admits several token IDs. Exactly-one-token runs may therefore be
short and rare. My own proposed trace—legal-token cardinality and forced-run
length distributions—would expose this, but I ranked the optimization before
having that evidence.

The obvious more aggressive alternative is not an exact repair. Choosing a
canonical tokenization for a forced byte run changes the token history relative
to ordinary masked decode. The emitted structural bytes may match, but KV,
hidden state, and later free-value tokens can differ. “Token healing” at the
exit boundary does not make the intermediate model state equal. Such a
byte-jump path would need:

- a separately frozen tokenization rule;
- an explicit statement that it changes decode semantics;
- L5 task-quality and downstream-token evaluation;
- its own prompt/recipe hash;
- ordinary masked decoding as fallback.

It cannot inherit my claimed state-equivalence gate. That is a genuine
concession.

CC is also right to bound the sparse-projection claim. The untied `lm_head` is
about 0.5 GB of an illustrative 3.7 GB decoded-token weight stream—roughly
13–14%. Avoiding it can be meaningful in a mature memory-bound engine, but it
is not a whole-forward-order speedup. The end-to-end win is that fraction
multiplied by the share of decode positions whose legal set is sparse, minus
planner overhead. “Radically cheaper constrained decoding” was too broad;
“exactly avoids a measurable projection tax on sparse grammar states” is the
honest claim.

I also accept CC’s treatment comparison for source grounding. Its task defaults,
copy-mask cost discussion, and unsatisfiable-field treatment improve my
proposal. My repeated-occurrence stance remains the safer one: return
compatible intervals or explicit ambiguity rather than silently minting one
offset.

### Where CC is wrong

CC says my worst decision is demoting constraint telemetry because an
always-on margin/override signal costs approximately nothing. That combines two
incompatible execution modes.

On the current full-projection path, retaining best legal and best illegal
logits during the scan adds little incremental work. But the scan itself is the
roughly 0.5-GB-per-token operation that `ProjectLegal` skips. An exact illegal
maximum cannot be recovered from rows never computed. “Margin is cheap” and
“sparse projection saves the full head” cannot both describe the same step.

The correct design is:

- full projection with legal/illegal telemetry during all eval and
  qualification runs;
- a deterministic, digest-keyed audit sample in production;
- `not_computed` on sparse steps that were not audited;
- grammar-only facts such as legal-set cardinality reported separately;
- no conversion from pressure to correctness risk without labeled
  calibration.

CC’s scoring also repeats the semantic mistake from its own Idea 3. A model can
confidently select the wrong legal invoice number with low pressure. A high
margin against the grammar can reflect useful enforcement of punctuation,
canonical spelling, or an enum. Pressure may be one calibrated policy feature;
it is not a fabrication probability.

### Revised position

Keep the execution-compiler frame, `ProjectLegal`, `CopyFromSource`, and
`FullProjection`. Rename the exact forced primitive `FeedUniqueTokens` and
treat it as opportunistic until traces prove useful runs exist. Any byte-level
jump becomes a separate semantics-changing research card.

The idea remains excellent because grounding and exact sparse projection stand
without forced runs. Its performance ceiling is lower than I originally
implied.

---

## 2. Durable, incremental corpus jobs — CC score **900**

### Where CC is right

I agree with nearly all of this assessment, including CC’s conclusion that the
semantic execution key, privacy-default journal, and delivery-authority
boundary are the strongest parts of my list.

CC’s only substantial concern is public surface area. That is fair.
`start/status/resume/verify/materialize`, retention, inspection, and purge can
become a small job-management product adjacent to the composable batch pipe.
The state machine is necessary; five frozen verbs on day one are not.

The implementation should therefore freeze the authority model before the
command spelling:

- `batch` remains a stateless stdin/stdout contract;
- a durable run owns a manifest, journal, and optional spools;
- result commitment precedes completion;
- resume requires an exact semantic key;
- owned materialization and arbitrary stdout make different delivery promises.

The eventual CLI could put inspection/verification/materialization under the
already planned `runs` surface, or expose fewer `job` verbs initially. That is
an ergonomic question to test, not an excuse to weaken crash semantics.

### What CC did not criticize, but the duel exposed

My per-item cache language is safe only for item-local task stages. Some product
operations are corpus-coupled:

- entity resolution produces final clusters from a globally ordered mention
  set;
- map-reduce has a global reduce;
- aggregate receipts and some policies depend on the whole snapshot.

An unchanged document’s mention extraction may be reusable while its final
entity cluster changes because another document was added. A semantic key needs
**dependency scope**, not just more hash fields. This becomes Blind Spot 2
below.

### Revised position

No ranking retreat. Adopt the state, identity, privacy, and exact-delivery
contract in Phase 4. Minimize the initial CLI and explicitly type item-local
versus corpus-global stages.

---

## 3. Bounded public task recipes — CC score **700**

### Where CC is right

This is the largest product-scope concession.

CC walks my motivating examples against the existing task surfaces and is
right:

- support routing is mostly `classify --labels` plus a preset;
- legal-clause extraction is already `extract --schema`;
- a faithfulness rubric is already data;
- a house PII policy is already a policy pack.

A public recipe language adds value primarily for **recombination**: custom
prompt segment order, candidate scoring semantics, bounded postconditions, and
composed task steps. I described the delta as “twelve tasks to a task
appliance,” which overstates what is missing from an already highly
parameterized portfolio.

The inner-platform warning is also real. If built-ins must all pass through a
public declarative format, every specialized task creates pressure to expand
the language. Soon the supposedly bounded format grows conditionals,
task-specific escape hatches, migrations, and a second implementation of Rust
logic. The one-model appliance pays the framework tax it was designed to avoid.

Most importantly, public recipes freeze a contract before the twelve built-ins
have demonstrated what the common representation actually is. Design stage is
the right time to establish an **internal** IR, not necessarily the right time
to publish it.

### What remains valuable

The architectural core survives:

- compile every built-in `TaskPlan` to one bounded internal `TaskIR`;
- make prompt segments, decode strategy, grammar, candidate trie, budgets,
  provenance, and postconditions explicit;
- require specialized Rust hooks to be typed and visible rather than silent
  bypasses;
- ship `check/explain/sample` over existing schemas, labels, rubrics, and
  presets.

Built-ins should dogfood common IR fields without forcing specialized semantics
into a generic bytecode. After Phase 4–5 equivalence fixtures show which parts
are stable, a public data-only subset can be proposed with real evidence.

### Revised position

CC’s 700 is fair for the idea as written. I would no longer rank a public
recipe compiler third. I would rank **internal `TaskIR` design** highly and make
public recipes a conditional Phase-5/v2 surface.

This is a genuine change of mind, not a rebranding defense.

---

## 4. Document-major analyze packs — CC score **800**

### Where CC is right

The prompt-segment ABI is the design-stage decision. I bundled too much runtime
machinery behind it.

A disciplined first implementation needs only:

1. explicit global/document/task/scaffold segments;
2. independent tasks in one result bundle;
3. shared tokenization, coordinate maps, substring indexes, and deterministic
   detectors;
4. ordinary cold task execution when prompt prefixes do not match;
5. an experimental document-KV fork only for tasks whose locked scorecards
   preserve quality.

Typed inter-task DAGs, hybrid document×task tiles, and a generalized
content-addressed KV block graph are later optimizations. The existing
snapshot/COW tree should be exhausted first. CC is right that my version was
the better v2 product vision than v1 scope.

The current plan’s disposition—segment ABI accepted, document-major execution
empirical and Phase 7—is more conservative than either original proposal and
is justified. Instructions-last is not a free performance switch; it can change
the model’s task behavior.

### Where I only partly agree

CC calls privacy namespaces and timing-leak testing likely ceremony in a local
single-user process. A generalized cross-namespace timing program would indeed
be premature. But namespace-complete cache keys are not ceremony for a
re-entrant library or a daemon serving independent callers. They cheaply
prevent one caller from obtaining another caller’s user-document prefix. Keep
isolation in the key and defer broader timing claims until a real multi-client
threat model exists.

### Revised position

Adopt the segmented ABI now. Ship a coherent multi-task envelope and shared
non-neural preprocessing before shared document KV. Add dependencies, block
graphs, and hybrid scheduling only from measured demand.

---

## 5. User qualification, calibration, and safe activation — CC score **830**

### Where CC is right

CC fairly identifies the split-manifest discipline as the strongest part and
activation/rollback as the easiest part to over-scope.

The user value begins with:

- canonical labeled-data adapters;
- disjoint development/calibration/locked-test IDs;
- leakage and duplicate refusal;
- paired baseline/candidate scorecards;
- digest-bound qualification receipts;
- `INSUFFICIENT_DATA` instead of unstable authority.

Those should not wait for a sophisticated active-artifact manager.

I also concede the energy-metric point. The performance project may measure
energy where a host has a credible instrument. A stable user qualification
contract should not half-promise portable energy numbers. Make energy an
optional evidence attachment with named authority, not a canonical v1 metric.

Paired-under-resume is nontrivial, but durable jobs give it a clean solution:
pair by immutable item ID after both runs complete, never by completion order.
That obligation stays.

### Where I retain the idea

Activation and rollback are not useful only for a future model zoo. One model
still has several semantic configurations:

- int8 and int4 artifacts;
- prompt and task-pack versions;
- calibration and decision policies;
- packing/numerics profiles.

A stale qualification must not authorize a different combination. Separating
installation from activation and retaining one prior content address are
valuable. They simply belong in Phase 6 after `eval/calibrate/qualify`, not in
the first Assay surface.

### Revised position

No major confidence reduction. Phase the feature: public evaluation and
qualification first; explicit activation/rollback later; portable energy
claims removed.

---

## 6. Selective automation and correction flywheel — CC score **820**

### Where CC is right

The flywheel’s value is adoption-dependent. Most casual users will not label a
review queue, maintain split manifests, refit calibration, and compare policies.
For recurring consequential pipelines, that work is precisely what makes the
system safe and can be transformative. The implementation should not make the
full workflow a Phase-5 prerequisite for everybody.

The right sequence is:

1. static, task-specific accept/abstain/spill policy;
2. `--review-out` with stable source/result IDs;
3. schema-validated correction import into an explicitly named local suite;
4. offline recalibration/policy fitting only after users demonstrate demand.

CC also catches a validity condition that I referenced but did not carry
through strongly enough. `policy fit --target-risk` is scoped to the population
represented by calibration data. The policy artifact must contain that
validity scope, shift indicators, and expiry. On detected shift, the
conservative action is to invalidate the calibrated claim and increase review
or abstain—not continue publishing the old target-risk number.

### Where CC is wrong

The proposed “merged design” of always-on model/grammar margin plus sampled
full mass is impossible on a genuinely sparse `ProjectLegal` step. Exact
best-illegal margin requires illegal logits. Computing them forfeits the
projection saving even if the code accumulates the maximum in the same pass.

What should always be present is the **availability state**:

- full-audit step: legal/illegal margin and mass components as actually
  computed;
- sparse step: legal-set facts plus `model_constraint_pressure:
  not_computed`;
- no zero-fill and no proxy relabeled as the missing quantity.

A deterministic sample of full projections can estimate behavior without
charging every production step. All locked evals can use the audit path. This
is an honest interaction between Ideas 1 and 6, not one idea undercutting the
other.

CC also continues to overrate pressure as a decision signal. It may correlate
with error on a named task and dataset. Until that correlation is calibrated,
it is a diagnostic, not “one of the most decision-relevant signals.”

### What genuinely changed

CC’s selection-bias point is implicit in its adoption critique even though it
does not fully develop it: a flywheel that labels only low-confidence spills
learns about the selected hard tail, not the accepted production population.
That observation produces Blind Spot 1 below—an unbiased audit stream of
accepted results.

### Revised position

Keep the static decision policy and review boundary. Move correction fitting
later, propagate calibration validity/shift semantics into the policy, and add
an unbiased production-audit plane before claiming the flywheel improves
overall risk.

---

## 7. Exact continuation-trie scoring — CC score **810**

### Where CC is right

CC correctly says the beneficiary set depends on label-internal prefix sharing,
not merely the task scaffold that the ordinary prefix cache already handles.
Hierarchical taxonomies and catalogs can win substantially; flat unrelated
labels may not. My proposed reuse estimate and naïve fallback address that.

The omitted short-ID baseline is also worth adding. Compact canonical IDs can
make candidate continuations cheap, especially if descriptions already appear
in the prompt. It is not universally better:

- opaque IDs can weaken the model’s semantic association;
- a large catalog still has to be represented or retrieved somehow;
- prompt cost may dominate;
- calibration changes with the encoding.

But it is a cheap rival and belongs in the EV comparison.

### The KV issue is more serious than CC states

The current plan’s bf16 KV census is about **176 KiB per token per sequence**
across all 44 loop-layer slots. With an illustrative 16-token page, one page is
about **2.75 MiB**. A naïve COW trie whose many children append into a shared
partially filled page can create an enormous branch cost. Thousands of broad
frontier nodes are not made practical merely by saying admission is bounded;
the bound may force an unusably small frontier.

The scorer therefore needs several exact execution modes:

- **breadth/frontier mode** for small tries where batching repays retained
  state;
- **depth-first stack mode** that reuses tail storage and trades batching for
  memory;
- **smaller/special tail pages** if their allocator/proof complexity earns its
  keep;
- **naïve per-candidate scoring** when trie shape or memory makes sharing lose;
- **short-ID scoring** as a quality/performance baseline.

The compiler should estimate not only node reuse but COW tail bytes under the
actual KV page geometry. It should refuse or choose a different traversal
before allocation.

### Revised position

The finite-language dynamic program remains sound and attractive, and exact
differential testing remains clean. I lower implementation confidence because
the 44-deep KV makes the memory/scheduling constant load-bearing. This stays a
strong idea, but “easy to implement correctly” was too optimistic.

---

## 8. Cross-loop physical-layer wavefront coalescing — CC score **580**

### Where CC is right

CC’s central objection is correct.

The planned scheduler’s simple state is a synchronized forward sweep:

```text
loop 0, layers 0…21 for the admitted rows
loop 1, layers 0…21 for the same rows
```

Under that design, different admitted rows do not normally wait at the same
physical layer in different loops. My mixed-stage opportunity requires
asynchronous cohort progress or mid-sweep admission. In other words, the
optimization depends on first creating a more complex scheduling state.

Even where two staggered cohorts can be pipelined, admitting them together
would ordinarily process both cohorts’ loop-0 rows in one sweep and both
loop-1 rows in another: two physical-layer weight streams either way. The
pipeline may affect admission latency or fill drain tails, but it does not
magically remove one of the model’s two semantic executions.

Cancellation, drain tails, and heterogeneous work can still create partial
groups. That makes loop-stage occupancy instrumentation useful. It does not
justify a readiness DAG, new compatibility semantics, model checking, and
hostile interleaving campaign before any trace shows material fragmentation.

### What changed

I over-weighted model-specific cleverness and under-weighted prerequisite
existence. The fact that I wrote an excellent kill gate does not make the
underlying optimization a top-eight investment.

CC’s distinction is exactly right:

- as cheap trace instrumentation and a research card: worthwhile;
- as a top-eight implementation proposal: misallocated priority.

The current plan’s AA-W1 disposition is the correct one. If no mixed-loop
fragmentation exists, the idea should die without source code.

### Revised position

Full concession. Remove this from my top eight. `fnlp tune`, schema tooling, or
one of the blind spots below deserves the slot more.

---

## Reaction to CC’s factual-accuracy audit

CC usefully separated factual hygiene from idea value. Two findings are valid,
two are not, and one is snapshot-sensitive.

### Valid: the withdrawn citation

CC is right. The official arXiv record says that
[Copy-as-Decode was withdrawn](https://arxiv.org/abs/2604.18170) after internal
review concerning authorship and contribution agreements. I cited it as a
feasibility precedent without noting that status. That was a real
citation-hygiene failure even though the algebraic question can be investigated
independently.

The current provenance file has already been amended to label it withdrawn and
to deny it authority for feasibility, correctness, performance, or novelty.
That correction is warranted. More importantly, CC’s token-trigger criticism
shows why the withdrawn citation never resolved the project-specific
specification anyway.

### Valid at authorship time: invented AA identifiers

The plan snapshot against which I originally wrote did not contain `AA-S1` or
`AA-K1`; I used names that resembled the project’s alien-artifact taxonomy
without verifying them. That is precisely the sort of plausible cross-reference
this project must reject. I concede the error.

The current v3 plan now defines `AA-K1`, `AA-S1`, and `AA-W1`, so the strings in
the preserved provenance file happen to resolve today. That does not
retroactively make the original references valid.

### Incorrect: “the plan ends at §8.5”

CC says my “§8.4–§8.6” range is invalid because the plan ends at §8.5. That is
wrong. §8.6 is the threat, privacy, and resource model. It existed in the plan
snapshot I read and remains present in v3.

### Incorrect: redistribution was not blocked

CC also says my appendix contradicted the plan by saying public redistribution
remained blocked. The current master plan explicitly labels the public
derivative-weight redistribution conclusion **blocked** until §5.7 records
adequate immutable license/NOTICE authority or rights-holder confirmation.
Apache-2.0 metadata is strong evidence of intent, but the plan deliberately
distinguishes that from authority to publish transformed assets.

My appendix’s deferral reason was therefore consistent with the fail-closed
plan. “Blocked pending the authority gate” would have been more precise than
the shorter wording, but it was not the contradiction CC reports.

---

## Blind-spot probe: what neither list saw

I compared both original top-eight lists and both 22-candidate appendices. The
three ideas below are not renamed versions of grounding, recipes, evaluation,
durability, escalation, document-major execution, tuning, trie scoring, or
wavefront scheduling. They arise from assumptions that **both** original lists
shared:

- qualification was treated as an offline event rather than a claim that can
  decay in production;
- durable jobs were treated mostly as collections of independent documents;
- the loaded process was treated as if it naturally had one caller.

After my blind-spot draft, CC’s concurrent reaction independently proposed a
frozen-corpus human acceptance audit, now represented in the plan as AA-A1.
That is strong post-duel convergence on the audit-authority gap. Blind Spot 1
below is the complementary longitudinal problem—monitoring accepted production
results and qualification decay across successive populations—rather than
claiming that one frozen job passed an acceptance plan. Blind Spots 2 and 3 do
not overlap CC’s three post-duel proposals.

### Blind Spot 1 — Sentinel: unbiased production audits and qualification decay

#### The missing problem

Both lists propose excellent offline evaluation and a low-confidence review
queue. Neither addresses the statistical trap created by that combination:
if humans review only abstentions and low-confidence spills, the correction
suite becomes a sample of cases selected by the current policy. It cannot tell
us the error rate among confidently **accepted** results—the exact place where
silent failures matter most.

Calibration and qualification can also decay after deployment because document
language, schema mix, labels, vendors, or business processes shift. Saying
“distribution shift invalidates coverage” is honest, but the product still
needs a way to discover that the claim may no longer apply.

#### The idea

Add a small, deterministic **risk-limiting audit plane**:

```bash
fnlp job start corpus.ndjson --policy invoices.policy.json \
  --audit-rate 0.005 --audit-out accepted-audit.ndjson

fnlp audit import reviewed-audit.ndjson --qualification invoices.qual.json
fnlp audit status invoices.qual.json --json
```

Audit inclusion is determined by a keyed hash of
`(semantic_execution_key, document_id, task, stratum)`, not by model output.
That makes the sample reproducible and gives every accepted result a known
nonzero inclusion probability. Strata may include task, predicted label,
language, document-length band, and confidence band, but the sampling
probability is recorded so estimates can be correctly weighted.

The audit stream contains stable references and provenance by default; source
text is rehydrated by the caller or explicitly spooled under the existing
privacy contract.

Reviewed audit items support:

- unbiased estimates of accepted-result error/selective risk;
- worst-slice reports with minimum sample sizes;
- comparison with the qualification baseline;
- qualification expiry or invalidation;
- a shadow candidate run on the same sampled items before activation.

AA-A1 answers a different authority question: whether a human-graded sample
supports a scoped acceptance decision for one frozen owned job. Sentinel reuses
its sound finite-population sampling primitives but asks whether a
qualification remains credible over a sequence of new populations. It cannot
carry one job’s acceptance interval forward; each window/job is separately
identified, and any longitudinal alarm is a trigger for new human evidence, not
a model-issued rejection certificate.

The conservative action on an invalid or underpowered claim is explicit:
increase audit/review coverage, mark confidence `uncalibrated`, or abstain. The
system does not silently retune itself.

#### Why neither model saw it

CC focused on escalation and telemetry; COD focused on a correction flywheel.
Both looked at the cases the model already considered hard. The duel exposed
that those corrections are selection-biased and therefore cannot monitor the
accepted population.

#### Utility versus complexity

This is mostly composition of durable IDs, review records, Assay metrics,
fsqlite state, and qualification digests. The new hard parts are a frozen
sampling design, weighted estimators, minimum-data rules, and explicit
qualification invalidation. That is moderate complexity with very high value
for any user who relies on a target-risk claim.

**Recommendation:** stronger than automatic self-consistency escalation and
more urgent than the correction-fit flywheel. Design with qualification; ship
after the static policy and review format exist.

---

### Blind Spot 2 — Portable corpus snapshots with scope-correct sharding, merge, and lineage

#### The missing problem

Both durable-job proposals implicitly make “unchanged document” the reuse unit.
That is correct for extraction or classification, but not for every product
task.

`resolve` is corpus-global: adding one mention can alter candidate pairs,
connected components, and final clusters for otherwise unchanged documents.
Map-reduce reuses map nodes but must rerun the complete reduce. Aggregate
policies may also depend on a full corpus snapshot. A cache keyed only per
document can therefore return locally valid but globally stale results.

At the same time, exact manifests and batch invariance make an attractive
capability visible: a large corpus could be partitioned across several offline
CPU hosts and merged without adding any inference network stack.

#### The idea

Make a durable job a **versioned corpus snapshot DAG**, and require each task
stage to declare one of three dependency scopes:

- `ItemLocal`: result depends only on one item plus the semantic execution key;
- `PartitionReduce`: item/map nodes are reusable, but a complete deterministic
  reduce runs for the current child set;
- `CorpusGlobal`: final authority depends on the entire snapshot.

Then expose portable partition/merge artifacts:

```bash
fnlp job partition <snapshot> --parts 8 -o shards/
fnlp job merge shards/*.receipt --snapshot <snapshot> -o merged/
fnlp resolve --since <prior-snapshot> --lineage-out entity-lineage.ndjson
```

Each partition carries:

- parent snapshot digest;
- exact assigned item IDs under a frozen partition rule;
- full semantic execution key;
- allowed host/numerics compatibility class;
- expected stage scope;
- completion receipt.

`merge` refuses missing, duplicated, wrong-snapshot, wrong-recipe, or
incompatible-host records. It may merge `ItemLocal` results directly, must
rerun a declared reduce over the complete current child set, and cannot pretend
independently computed corpus-global final IDs are composable.

For entity resolution, mention extraction and canonical pair scores may be
content-addressed and reused. Final clustering remains snapshot-qualified.
Across snapshots, emit deterministic lineage events such as
`unchanged`, `new`, `retired`, `merge`, `split`, and `ambiguous_match` rather
than promising that a global entity ID can never change when the corpus
changes.

#### Why neither model saw it

Both lists celebrated exact per-item reuse and stopped one level too early.
The adversarial review made the omitted authority boundary visible: task
results have different dependency scopes. The no-network doctrine also
anchored both models on one machine, even though portable manifests let agents,
CI, rsync, or an external scheduler distribute work without adding network code
to `fnlp`.

#### Utility versus complexity

The scope type and snapshot digest are low-cost design decisions now. Portable
shards and lineage are medium complexity, but they unlock:

- safe recurring entity-resolution corpora;
- exact horizontal use of several Macs/servers/CI workers;
- agent-orchestrated air-gapped processing;
- explicit cross-ISA proof boundaries;
- no stale global answers masquerading as incremental-cache hits.

Cross-host canonical byte equality must not be assumed. A shard declares a
certified compatibility class; heterogeneous receipts may be merged only under
the claim level actually proved by the cross-ISA conformance suite.

**Recommendation:** add dependency scope to `TaskIR` and durable semantic keys
now. Implement portable partition/merge and entity lineage after the single-host
job path is green.

---

### Blind Spot 3 — A resident local engine rendezvous for many agents and tools

#### The missing problem

Both lists optimize one loaded process extremely well. Neither asks how
independent local clients discover and share it.

On an agent workstation or CI host, several shells, coding agents, editor
plugins, and background jobs may invoke `fnlp` independently. If each process
loads its own 3.1–4.7 GB artifact, owns a kernel pool, and performs its own
prefill cache, the host suffers:

- duplicated weight memory and cold loads;
- competing full-core pools;
- destroyed cache locality;
- no global admission budget;
- worse latency and throughput than one well-batched engine.

`fnlp batch` solves “many documents from one cooperating pipe.” It does not
give unrelated processes a safe rendezvous point.

#### The idea

After the core CLI is proven, add an opt-in **local-only resident engine**:

```bash
fnlp resident start --endpoint <owner-only-local-endpoint>
fnlp --resident auto extract --schema s.json doc.txt
fnlp resident status --json
fnlp resident stop --drain
```

This is not an HTTP/OpenAI server and never opens a routable network listener.
It uses an owner-only OS-local IPC endpoint—Unix-domain socket on Unix and the
appropriate local primitive on Windows—carrying the same versioned framed
request/result schema as robot/batch mode.

One process owns:

- one artifact and measured dispatch profile;
- one asupersync runtime;
- one physical-core kernel pool;
- one global memory/admission certificate;
- fair per-client queues and cancellation;
- namespace-isolated prefix/cache state.

Clients remain ordinary synchronous CLIs. If the resident process is absent,
incompatible, or disabled, they fall back to the current in-process engine.
Results remain client-composition invariant under the same batch-M gate.

Security and privacy are part of the mechanism:

- owner-only endpoint permissions and peer identity where available;
- no text-bearing logs;
- per-client caps and explicit cancellation ownership;
- user-document cache sharing off unless namespace-authorized;
- drain/restart receipts;
- no remote transport or credential surface.

#### Why neither model saw it

Both models reasoned from the Threadripper batch workload inward: keep weights
loaded once **inside one invocation**. The user prompt explicitly values AI
agents, but neither list followed that fact outward to several independent
agent processes on the same host.

#### Utility versus complexity

The potential savings are large on exactly the target machines: gigabytes of
duplicate memory, repeated cold starts, and oversubscribed cores disappear.
The scheduler and re-entrant engine already need most server-side semantics.

The cost is real:

- cross-platform local IPC;
- protocol/version negotiation;
- peer permissions;
- lifecycle and stale-endpoint recovery;
- a new long-lived operational surface.

The closed dependency universe and Windows implementation may make the first
portable version awkward. This should therefore be a measured Phase-7 feature,
not a v1 expansion. First record multi-process load/memory/latency traces on
agent-heavy hosts. If mmap/page-cache sharing plus ordinary batch composition
already solves the problem, keep the resident layer out.

**Recommendation:** add a research/design card and preserve the robot protocol
so it can be multiplexed later. Pursue only after traces show repeated
multi-process loading or pool contention.

---

## Final synthesis

The exchange changes the merged roadmap in a healthy direction:

- **Keep and prioritize:** durable jobs, source-grounded fields, sparse legal
  projection, user evaluation/qualification, static selective policies, and
  exact continuation scoring with a real memory plan.
- **Design internally, expose later:** `TaskIR`, public recipes, activation
  gating, correction fitting, multi-task DAGs.
- **Measure before promising:** forced byte/token jumps, document-KV sharing,
  per-install tuning, resident multi-client service.
- **Research card only:** cross-loop wavefront coalescing.
- **Reject as claims:** constraint pressure as fabrication probability,
  “free” illegal-token margins on sparse paths, and holdout structural coverage
  as schema-inference accuracy.

The blind spots add three missing forms of reliability:

1. **epistemic reliability over time**—audit accepted production results so a
   qualification claim can expire honestly;
2. **semantic reliability over an evolving corpus**—scope caches and cluster
   identities to the snapshot they actually depend on;
3. **operational reliability across local callers**—avoid turning several AI
   agents into several competing copies of a multi-gigabyte engine.

Those were invisible when both models looked only at features inside one
forward pass, one evaluation run, or one batch process. The adversarial
exchange made the omitted lifecycle around the engine visible.
