# WIZARD_IDEAS_COD.md — Top eight improvement ideas for `franken_nlp`

**Status:** Idea-wizard output for consideration during the plan’s external-review cycle. These are proposed amendments to `COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md`, not current commitments and not authority to bypass any `[OPEN]`, `[BLOCKED]`, parity, licensing, or release gate.

> **Post-review disposition (2026-07-30):** preserved as ideation provenance, not specification. Plan v3 §10.6 is authoritative. It accepted the exact execution-compiler primitives, source-grounded fields behind OQ-17, durable jobs, continuation tries, user qualification, and a bounded internal `TaskIR`; made public recipes conditional; moved document-major packs and schema inference to empirical stretch work; and retained cross-loop wavefront coalescing only as AA-W1 research. In particular, constraint diagnostics are not fabrication probabilities, illegal-vocabulary margins/full mass require full projection, an instructions-last prompt is not assumed better, automatic escalation is not a default, and comparative novelty/magnitude statements below are hypotheses rather than project claims.

**Premise:** The plan is the product today. The best intervention is therefore not “add more commands.” It is to make a few architectural decisions now that deepen the existing moat—one looped model, schema-valid output, CPU specialization, corpus throughput, and unusually honest evidence—before prompts, artifact contracts, and subsystem boundaries become expensive to change.

## How the 30 candidates were winnowed

I generated 30 candidates spanning structured decoding, model execution, corpus operations, task extensibility, evaluation, trust, memory footprint, and user ergonomics. Each candidate was tested against the idea-wizard rubric: robustness, reliability, performance, intuitiveness, friendliness, ergonomics, usefulness, compellingness, accretiveness, and pragmatism. Usefulness and pragmatism received double weight; accretiveness received 1.5× weight. I then applied four harder project-specific filters:

1. **Does it deepen the one-model appliance rather than turn it into a generic framework?**
2. **Can it preserve exact semantics or expose an explicit, conservative fallback?**
3. **Can its benefit be demonstrated by the project’s own conformance and performance machinery?**
4. **Is there a design decision worth making now, before implementation makes it expensive?**

The eight winners are ranked by overall project value, not implementation order.

| Rank | Idea | Primary gain | Complexity | Confidence |
|---:|---|---|---|---|
| 1 | Stencil becomes an execution compiler | Grounding plus radically cheaper constrained decoding | Medium-high | High on correctness; medium-high on magnitude |
| 2 | Durable, incremental corpus jobs | Crash safety, exact resume, and zero needless recomputation | Medium | Very high |
| 3 | A bounded declarative task-recipe compiler | User extensibility without plugins or framework tax | Medium | High |
| 4 | Document-major multi-task analysis packs | One document prefill serving many tasks | Medium-high | Medium-high |
| 5 | User-owned qualification and safe upgrades | Evidence on the user’s domain before activation | Medium | High |
| 6 | Selective automation plus a correction flywheel | High-confidence automation with an honest hard-case path | Medium-high | High on mechanism; task-dependent on magnitude |
| 7 | Exact continuation-trie scoring | Large taxonomies and entity catalogs at practical cost | Medium | High |
| 8 | Cross-loop physical-layer wavefront coalescing | Recover batch utilization lost to loop-stage fragmentation | High | Medium; explicitly profile-gated |

These ideas reinforce one another, but none requires all the others. In particular, Ideas 1–7 have useful deterministic fallbacks and remain valuable if Idea 8 proves slower and is buried in `NEGATIVE_EVIDENCE.md`.

---

## 1. Turn Stencil from a token-mask engine into a grounded execution compiler

**Proposed plan integration:** Amend §6.10–§6.11, §7.1–§7.3, §9.2–§9.5, and the Phase 4 exit gate.

### The idea

The current plan gives Stencil one runtime action: compute the ordinary model logits, then mask illegal vocabulary items. That is sufficient for validity, but it leaves two enormous opportunities unused:

1. The grammar often knows that only a small subset of vocabulary rows can possibly win.
2. At many JSON positions, the grammar knows the next token—or a whole run of tokens—without consulting the logits at all.

Compile each grammar state into a small **decode execution program**, choosing among four exact primitives:

- **`ProjectLegal(rows)`** — compute only the `lm_head` rows for grammar-legal tokens.
- **`FeedForced(tokens)`** — when the tokenizer/grammar product has one legal token repeatedly, omit token selection and advance the model state for the maximal forced run using a causal micro-prefill.
- **`CopyFromSource(language)`** — for fields declared verbatim, intersect the JSON grammar with a source-substring automaton so invented values are unrepresentable.
- **`FullProjection(mask)`** — the existing full-vocabulary path, retained as the universal fallback.

This makes the grammar part of the execution planner rather than a filter bolted onto generation.

### What users would perceive

The product’s flagship sentence becomes materially stronger:

> Structured output is schema-valid by construction; fields marked as sourced are grounded by construction; and the engine exploits those guarantees to avoid work the model cannot affect.

For an invoice schema, object keys, punctuation, booleans, fixed enums, and many short literals should no longer pay for a 166,144-row projection at every token. For NER, citations, keyphrases, redaction spans, and evidence fields, an `x-fnlp-source` annotation could require a byte-exact source substring. Users would not receive a plausible-looking invented citation with an `unanchored` warning after the fact; a successful grounded field could not contain off-source text.

The feature should be visible but not fiddly:

```json
{
  "type": "string",
  "x-fnlp-source": "verbatim"
}
```

Supported modes should be narrow and honest: `verbatim`, `verbatim-normalized` with a named normalization contract, and ordinary unconstrained semantic fields. Dates or normalized currency values should not be forced into source-copy mode unless the schema author requests it.

### How it would work

#### A. Sparse legal-row projection

Stencil already walks a vocabulary byte trie to determine which token IDs are legal from a grammar state. Preserve the resulting legal-ID set, not merely its bit mask. If its cardinality falls below a measured threshold, call the row-sliced `lm_head` path already planned for classification.

For constrained greedy decoding this is exact: illegal rows cannot win, so omitting them changes no allowed argmax. It is also exact for sampling from the grammar-conditioned distribution: compute every legal logit, apply temperature/top-k/top-p within the legal set according to the frozen constrained-sampling contract, and use the same seeded RNG order. A full-vocabulary denominator is unnecessary for the conditioned distribution because the common denominator cancels. It is necessary for claims such as “the model assigned 82% mass to legal continuations,” so that telemetry must either run a full projection or report `not_computed`; it must never pretend a candidate-conditional value is full-vocabulary mass.

When the legal set is dense—free-form string content is the obvious case—the compiler emits `FullProjection(mask)`. No heuristic is allowed to drop a legal token.

#### B. Forced-run causal prefill

At a grammar/tokenizer-product state, simulate legal token transitions. If exactly one token ID is legal, select it without computing logits. Continue until the state branches, accepts, or hits a configured micro-prefill cap. The selected tokens still must pass through all 44 effective layer executions to update KV; “forced” does **not** mean “skip the model.” Instead, process the known run as a causal prefill block, turning N memory-bound autoregressive steps into one compute-friendlier matrix path.

A 2026 preprint sketched updating KV for forced copy spans through parallel prefill rather than N decode calls, but the authors later **withdrew it following internal review**. It is retained only as bibliographic/negative-history context, not evidence for feasibility, correctness, speed, or novelty: [withdrawn Copy-as-Decode record](https://arxiv.org/abs/2604.18170). `franken_nlp` must derive and prove its CPU/looped-model path independently.

Mathematically, teacher-forced causal prefill and sequential decoding evaluate the same transformer. Numerically, different kernels can change reduction order. Promotion therefore requires:

- per-token hidden/KV differential fixtures for forced runs,
- L3/L4 equality under the designated deterministic profile,
- batch-M ≡ batch-1 with forced runs present,
- and automatic fallback to one-token feeding if identical row reduction cannot be maintained.

This is an optimization, never a new tolerance loophole.

#### C. Grounded source languages

For a verbatim field, build a linear-size suffix automaton or equivalent bounded substring index over the normalized source bytes. Intersect its transitions with the JSON string lexer and the tokenizer detokenization transducer. The grammar owns JSON escaping; the source automaton sees logical unescaped bytes. A successful value is therefore a substring by construction.

Repeated occurrences remain a real ambiguity. The result should return all compatible source intervals or use a separately constrained occurrence selector; it must not fabricate a unique offset. Unsatisfiable required fields yield a typed no-result/field error according to the task contract—never a silent switch back to unconstrained text.

Environment-indexed grammars are a broader research precedent for tightening generation to values actually present in a runtime environment: [Decode-Time Grammars](https://arxiv.org/abs/2607.18357). The project’s narrower source-substring language is much easier to bound and verify.

### Proof obligations and measurements

- Independent verifier: every successful grounded value occurs under the declared coordinate/normalization contract.
- Random supported schemas, adversarial Unicode, JSON escapes, repeated substrings, byte-fallback tokens, and empty-language cases.
- Exact comparison of sparse projection against full projection for every grammar state in a large fixture corpus.
- Exact comparison of forced-run output/state against ordinary constrained decode.
- Ledger separate wins for projection avoidance and forced-run prefill; do not combine them into an attribution-free benchmark.
- Report distributions of legal-set size, forced-run length, full-projection fallbacks, and wall-time contribution by task.

### Why the utility justifies the complexity

This is not a side feature. It simultaneously strengthens G8, reduces hallucination in extraction-shaped fields, accelerates the project’s flagship workload, reuses the row-sliced `lm_head`, and turns the tokenizer/grammar work into a larger competitive advantage. Efficient sound grammar/tokenizer alignment is known to be a substantial systems problem rather than a trivial mask: [Flexible and Efficient Grammar-Constrained Decoding](https://arxiv.org/abs/2502.05111). `franken_nlp` is already paying that cost; compiling the result into execution choices extracts far more value from it.

It ranks first because it improves correctness, trust, and performance through one coherent seam unique to this product.

---

## 2. Make corpus work a durable, incremental job—not an immortal pipe

**Proposed plan integration:** Amend §3.3, §8.4–§8.6, §9.3–§9.5, and Phase 4. Reuse `fsqlite`, but make content persistence explicitly opt-in.

### The idea

The motivating workload is repeatedly “run tens or hundreds of thousands of documents overnight.” A plain stdin/stdout daemon with bounded queues is excellent plumbing but insufficient job semantics. A power loss, disk-full condition, killed shell, or broken downstream consumer should not make the user guess what completed or rerun an entire corpus.

Add a first-class durable job contract:

```bash
fnlp job start manifest.ndjson --task classify --output results.ndjson
fnlp job status <job-id> --json
fnlp job resume <job-id>
fnlp job verify <job-id>
fnlp job materialize <job-id> --ordered -o final.ndjson
```

`fnlp batch` remains the lightweight composable pipe. `fnlp job` is the infrastructure surface for work whose completion matters.

### What users would perceive

- A 3 a.m. crash becomes “resume from item 78,421,” not “start over.”
- Re-running a corpus after changing five documents recomputes five documents, provided every semantic input to the other results is unchanged.
- Every job ends with a compact receipt explaining exactly which model, prompt, task recipe, grammar, calibration, numerics profile, and inputs produced the result set.
- Progress and ETA are inspectable without corrupting stdout’s data contract.

This is the moment the tool stops feeling like a clever model wrapper and starts feeling like dependable local data infrastructure.

### How it would work

#### A. Content-addressed job manifest

Normalize the input manifest into stable item IDs and SHA-256 digests. Freeze a **semantic execution key** containing at least:

- input-byte digest and declared text normalization,
- artifact/source/packing digest,
- tokenizer/template/task-recipe/prompt/grammar digests,
- task args and output schema version,
- thinking, sampler, quantization, KV, and numerics profile,
- calibration and decision-policy digest,
- engine semantic version.

An item result is reusable only under an exact key match. This is not semantic similarity caching; a one-byte input or prompt change invalidates the key. That conservatism is why reuse can be trusted.

#### B. Transactional item state

Use `fsqlite` for job metadata and item transitions such as:

`pending → admitted → running → result_committed → materialized`

The result bytes or result digest must be durably committed before an item becomes complete. Recovery treats any interrupted pre-commit state as retryable. Attempts are counted and error taxonomy retained. Disk-full, cancellation, and timeout each have explicit transitions.

Exactly-once semantics need careful scoping:

- When `fnlp job` owns a checksummed output spool and later materializes a file, it can guarantee one canonical record per item.
- Raw stdout cannot guarantee exactly-once delivery across a process crash and an arbitrary downstream consumer. It should promise stable IDs and at-least-once replay semantics, plainly documented.

That distinction prevents a common infrastructure lie.

#### C. Privacy-preserving modes

The default journal stores IDs, digests, status, metrics, and errors—not document text or model output. Resume then requires the original replayable manifest/input.

`--spool-input` and `--spool-results` are explicit privacy-affecting choices. Spools use owner-only permissions, a named retention policy, checked disk budgets, and an inspectable purge command. Reversible redaction maps remain excluded. There is no hidden “helpful” persistence.

#### D. Incremental corpus recomputation

The same semantic key enables an opt-in result cache. A recurring compliance scan can reuse unchanged document-task results and deterministically rerun only invalidated items. Map-reduce jobs key map nodes by chunk bytes and recipe, so an edited document can reuse unchanged chunk results while always rerunning the declared deterministic reduce over the complete current child set.

This is where the feature becomes accretive: the second and tenth run become cheaper without weakening correctness.

### Proof obligations

- Kill injection at every state transition, including after result fsync but before status commit and vice versa.
- Power-loss/corrupt-tail simulation for output frames and database WAL.
- Disk-full tests for journal, spool, and materialization.
- Uninterrupted run ≡ killed-and-resumed run by per-item semantic bytes under ordered deterministic mode.
- Configuration mismatch on resume must produce a named field-level digest diff and refuse to mix results.
- Cache reuse tests where each key component is perturbed independently.
- Privacy schema test proving default job tables contain no text-bearing columns.

### Why the utility justifies the complexity

Nearly all underlying pieces are already planned: bounded queues, per-document isolation, fsqlite, deterministic envelopes, hashes, and cancellation. The idea specifies their operational composition. Its complexity is ordinary and testable, while its value applies to every corpus task and every CPU architecture.

It ranks second because it is the highest-certainty way to make the ambitious performance work useful in the real world. Fast processing that has to restart is not high throughput.

---

## 3. Compile bounded declarative task recipes into the same IR as built-in tasks

**Proposed plan integration:** Amend §4.1–§4.2, §7.0, §7.8–§7.9, §8.2, and Phase 4. This is task-level extensibility, not model/runtime generality.

### The idea

The plan offers twelve excellent built-in tasks and data presets, but real NLP value is dominated by domain-specific variations: a support-ticket routing policy, a legal-clause schema, a product taxonomy, a custom faithfulness rubric, or a house PII policy. Requiring a Rust change for each variation turns the maintainers into a bottleneck. Allowing arbitrary scripts/plugins would violate the safety and closed-universe design.

The pragmatic middle is a **bounded, declarative task recipe** compiled into a frozen `TaskIR`:

```bash
fnlp recipe check support-routing.fnlptask.json
fnlp recipe explain support-routing.fnlptask.json --json
fnlp recipe sample support-routing.fnlptask.json
fnlp run support-routing.fnlptask.json ticket.txt
```

Built-in tasks should compile through the same IR. The built-ins remain curated, evaluated product experiences; the recipe format lets users safely express adjacent tasks without creating a generic ML framework.

### What a recipe may declare

- Versioned request and response schemas.
- Typed prompt segments and a small fixed placeholder vocabulary.
- Decode strategy: prefill-only candidates, constrained JSON/pattern, distribution, or bounded free text.
- Candidate continuations and normalization rule.
- Thinking default and hard budgets.
- Grounded/source fields from Idea 1.
- A bounded set of deterministic postconditions: evidence required, range checks, field implication, exact-source membership, uniqueness, and canonical normalization.
- Calibration/qualification artifact references.
- Stable output envelope additions under a namespaced extension field.

### What it may not do

- Execute code or tools.
- Open the network.
- Interpret Jinja, shell, Python, WASM, or an unbounded expression language.
- Read undeclared files at inference time.
- Retry unconstrained after failure.
- Introduce a new neural operator, tokenizer, model, or dependency.
- Claim calibrated confidence without a matching calibration artifact.

These negative capabilities are as important as the positive ones.

### How it would work

#### A. A small compiler, not runtime interpretation

Parse and validate the recipe before model load. Compile it into:

- exact prefix/tail token IDs and hashes,
- a grammar automaton plus resource estimate,
- legal-row plans and forced-run opportunities from Idea 1,
- an optional continuation trie from Idea 7,
- a postcondition program over a deliberately finite instruction set,
- budget/admission requirements,
- and the response schema.

The runtime executes `TaskIR`; it does not branch on arbitrary recipe JSON. The compiled form and source digest enter every result envelope.

#### B. Built-ins dogfood the abstraction

Extraction, classification, sentiment, and the other built-ins should have ordinary Rust modules only where they provide genuinely specialized deterministic logic. Their prompt/decode/budget contracts should still compile to `TaskIR`. A policy test should fail if a built-in silently bypasses the same hash, budget, and provenance mechanisms exposed to users.

This prevents the recipe system from becoming a neglected second-class surface.

#### C. Recipe tooling

`recipe check` performs compile/resource validation without loading weights. `recipe explain` prints prompt segment order, token counts, reachable schema features, expected `lm_head` strategy, memory/output bounds, and required calibration. `recipe sample` walks the grammar to show valid structural instances. `fnlp eval` from Idea 5 evaluates a recipe on user data.

Recipe packs can be directories with a canonical manifest, explicit license/provenance, and no executable content. Sharing a pack is therefore sharing auditable data, not installing a plugin.

### User perception

Users see a focused appliance that adapts to their domain:

> “The project ships twelve excellent tasks, and I can define a thirteenth safely in one versioned file without forking Rust or accepting arbitrary plugin code.”

This is a much more compelling extensibility story than either extreme: “only our presets forever” or “here is a generic prompt framework.”

### Proof obligations and complexity

- Parser/compiler fuzzing and strict unknown-key rejection.
- Checked state/byte/token estimates before allocation.
- Canonical compilation: same recipe bytes and referenced assets produce the same `TaskIR` digest across hosts.
- Built-in recipe output equivalence fixtures.
- No-file/no-network/no-code-execution sandbox tests.
- Golden `recipe explain` output as part of the robot contract.
- Schema-version migration policy should be forward-rejecting by default; early development should fix recipes directly rather than accumulate compatibility shims.

Complexity is moderate because the plan already has `TaskSpec`, decode strategies, embedded presets, prompt hashes, grammars, and budgets. The compiler consolidates them instead of inventing another engine.

It ranks third because it changes the value ceiling from “twelve tasks” to “a safe local task appliance,” while preserving the one-model specialization and closed dependency universe.

---

## 4. Add document-major `analyze` packs that share one document prefill across many tasks

**Proposed plan integration:** Decide the prompt-segment ABI in §7.0 now; amend §6.7, §7.0, §8.2–§8.4, and Phase 4.

### The idea

The planned prefix cache optimizes one task across many documents: cache the task prompt, then fork per document. Many real workflows are the transpose: run NER, PII detection, classification, keyphrases, and a summary over the **same** document.

Add a first-class multi-task surface:

```bash
fnlp analyze contract.txt --pack contract-intake --json
cat corpus.ndjson | fnlp batch --pack support-triage > analyzed.ndjson
```

An analysis pack is a declarative DAG of task recipes. Its default shape is independent branches over one document; explicit dependencies are allowed only when a downstream task consumes a typed upstream result.

### The architectural amendment needed now

Prompt segments need an explicit ABI, not an opaque rendered string. At minimum:

1. immutable global/system policy,
2. document payload,
3. task-specific instruction/tail,
4. answer scaffold.

For tasks whose quality survives an instructions-last layout, prefill the global policy plus document once and copy-on-write fork the KV pages into task-specific tails. Tokenization, Unicode coordinate maps, source-substring indexes, rule-based detectors, and chunk maps are also built once.

This must be measured, not assumed. Some tasks may require instructions before the document or a different chat-template role order. The recipe compiler should support both layouts; the task’s locked scorecard decides whether it is eligible for document-major sharing. Ineligible tasks fall back to their ordinary independently prefixed path inside the same bundle.

### How execution would work

#### A. Fork graph

Represent prompt state as a content-addressed KV block graph. A pack compiler finds common token prefixes among branches and emits fork points. Hash keys include every semantic component already required by the plan, plus a privacy namespace/salt.

Official vLLM documentation provides a useful, well-understood precedent for content-hashed KV blocks, reference counts, and cache isolation salts: [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/). `franken_nlp` still needs its own 44-slot loop-aware representation and exact prefix-fork proof.

#### B. Matrix scheduler

The scheduler now sees a document × task matrix. It may choose:

- **corpus-major:** many documents for one task, maximizing shared task prefix;
- **document-major:** many task tails for one document, maximizing shared document KV;
- **hybrid tiles:** a bounded rectangle when both dimensions have enough work.

The choice starts as a deterministic table derived from measured shapes and memory admission. No adaptive policy is needed for v1. Prompt-aware scheduling and chunked prefill have credible systems precedents, including [Preble](https://arxiv.org/abs/2407.00023), but CPU and loop-specific gains must be measured locally.

#### C. Atomic bundle semantics

An `analyze` result contains one envelope and per-task results/statuses. One failed branch does not erase successful independent branches. If a task depends on a failed upstream result, it emits `blocked_by` rather than running on missing data. Ordered mode fixes task order and result bytes.

### User value

The user no longer assembles five CLI invocations, pays five document prefills, reconciles five provenance envelopes, or wonders whether offset conventions match. One command produces a coherent analysis bundle with common source coordinates and common execution provenance.

This is especially compelling for:

- support-ticket intake,
- contract/compliance intake,
- local RAG ingestion verification,
- email/document PII and routing,
- and agent preprocessing.

### Proof obligations

- Fork graph ≡ independent cold prefills for every branch.
- Pack result ≡ the same recipes run separately.
- Quality A/B for every prompt layout; no task is made shareable by lowering its quality gate.
- Privacy isolation, collision-resistant keying, and timing-leak tests across namespaces.
- Memory admission includes all branch tails and worst-case output/grammar state.
- Cancellation/interleaving tests with partial branch completion.
- Benchmarks attribute tokenization reuse, document prefill reuse, and batch-shape effects separately.

### Why the utility justifies the complexity

The project already pays for COW KV pages, prefix hashes, task specs, batch scheduling, and uniform envelopes. The amendment generalizes them from a line into a DAG. The prompt ABI decision is cheap now and expensive after prompts and scorecards freeze.

It ranks fourth because it makes the existing portfolio feel like one product rather than twelve adjacent commands and creates a structural prefill win on a common workload.

---

## 5. Productize Assay as user-owned qualification, calibration, and safe activation

**Proposed plan integration:** Add `fnlp eval`, `fnlp calibrate`, `fnlp qualify`, and digest-gated model/recipe activation in §8 and Phases 5–6.

### The idea

The plan rightly insists that project claims use named datasets, prompt hashes, recipes, and calibration. Serious users need the same protection on their own domain. A support team does not care only about AG News accuracy; it cares whether a new artifact preserves macro-F1 on its ticket taxonomy. A legal user needs to know whether a prompt change damages clause extraction on its documents.

Expose Assay as a product surface:

```bash
fnlp eval --task classify --dataset tickets.test.ndjson --gold label
fnlp calibrate --task classify --dataset tickets.cal.ndjson -o support.cal.json
fnlp qualify --baseline active --candidate nanbeige-int4-v3 \
  --suite support-suite.json --policy production-gates.json
fnlp models activate nanbeige-int4-v3 --qualification qualification.json
fnlp models rollback
```

### What users would perceive

The upgrade story becomes:

> “Before changing the active model, quant recipe, prompt pack, grammar version, or calibration, replay my locked suite, show paired quality/performance deltas, and activate only if my gates pass.”

That is profoundly more trustworthy than “the release benchmark improved.” It also makes prompt customization safe: users can compare recipe A and B rather than choosing by anecdote.

### How it would work

#### A. A small set of canonical dataset adapters

Use NDJSON with task-specific gold fields and a versioned split manifest. The manifest binds item IDs/digests to `development`, `calibration`, and `locked_test`. The engine refuses to fit calibration on locked-test IDs and records every overlap.

Metrics reuse §9.6:

- exact/relaxed span F1 for NER,
- field F1 and validity for extraction,
- accuracy/macro-F1 and reliability for classification,
- rank/MAE/selective-risk for scoring,
- evidence/abstention metrics for QA and faithfulness,
- recall-first metrics for redaction.

#### B. Qualification as a digest-bound receipt

A qualification record binds:

- candidate and baseline semantic digests,
- dataset and split digests,
- exact commands/config,
- quality metrics with paired intervals,
- p50/p95/p99, memory peak, and energy where available,
- user-defined pass/fail thresholds,
- and the final decision.

It is a reproducibility receipt, not a universal certificate. Its claims are explicitly scoped to the named data and host.

#### C. Safe activation and rollback

Model/recipe installation and activation are separate. New artifacts remain inactive until explicitly selected. `--require-qualification` can make activation refuse a record whose digests do not exactly match. The previous active content address remains available for instant rollback. A stale qualification cannot authorize a different prompt, calibration file, or packing.

#### D. Local calibration

Fit only simple, inspectable methods the project already accepts—temperature/isotonic where implemented and conformal sets only when assumptions are stated. Output is a versioned calibration artifact with validity metadata. Distribution shift invalidates the coverage claim exactly as §7.8 requires.

### Proof obligations

- Split-leakage and duplicate-item detection.
- Reproducible metric implementations with golden fixtures.
- Paired comparison remains paired under resume and ordered/unordered execution.
- Qualification digest substitution tests.
- Activation failure leaves the previous model untouched.
- Rollback restores exact prior semantic configuration, not merely weight bytes.
- Clear `INSUFFICIENT_DATA` results rather than unstable confidence claims on tiny sets.

### Utility versus complexity

Most machinery is required internally already. The incremental work is stable dataset adapters, CLI ergonomics, qualification policy evaluation, and activation plumbing. It opens a major adoption path among users who will not automate consequential decisions without domain evidence.

It ranks fifth because it turns the project’s honesty doctrine into direct user value and makes every future release more accretive rather than risky.

---

## 6. Add selective automation and an explicit human-review correction flywheel

**Proposed plan integration:** Amend §7.8–§7.9, §8.4, §9.6, and the deterministic-controller contract. Phase 5, after calibration exists.

### The idea

A 3B model will have a hard tail. The pragmatic goal is not to disguise it or force all documents through an expensive mode. It is to automate the easy majority cheaply, spend more compute only where evidence supports it, and emit the residue in a form a person or external pipeline can handle.

Define a versioned `DecisionPolicy` with signals, actions, and an explicit loss table:

**Signals**

- calibrated task confidence or prediction-set size,
- grounded/postcondition status,
- constrained-decode pressure when a full-vocabulary audit actually computed it,
- candidate captured mass where defined,
- agreement across deterministic prompt variants or seeded samples,
- distribution-shift/OOD flags,
- and per-field evidence coverage.

**Actions**

- accept,
- rerun with thinking,
- run N seeded samples and aggregate under a frozen rule,
- abstain,
- or spill a review record.

`fnlp` never calls a cloud model. The spill/review NDJSON is the composable boundary.

### The user experience

```bash
fnlp batch --task extract --quality balanced \
  --review-out uncertain.ndjson

fnlp review import corrected.ndjson --suite invoices-local
fnlp policy fit --suite invoices-local --target-risk 0.02 \
  -o invoices.policy.json
```

The user sees coverage, selective risk, escalation counts, added compute, and review volume. A “balanced” preset is not a magic adjective; it names a frozen loss matrix and threshold artifact.

### Why this is more than an escalation flag

The correction path makes the system **operationally accretive**:

1. Uncertain items become structured review records with stable source/result IDs.
2. Human corrections are schema-validated and stored only in an explicitly named local suite.
3. Corrections expand the user’s eval/calibration data.
4. A user explicitly reruns `calibrate`, `policy fit`, or prompt comparison.
5. A new digest-bound policy may reduce future review load if locked evaluation supports it.

There is no hidden online learning, silent prompt mutation, or unreviewed “memory.” The model remains unchanged. Improvement is explicit, reversible, and inspectable.

For safe deterministic domains, imported corrections may also populate narrow rule assets—approved entity aliases or exact secret patterns—but those assets are versioned recipe inputs and evaluated like everything else.

### Implementation shape

Start with a static policy table. Each task declares which signals are semantically valid and which actions it supports. For example:

- classification may use calibrated margin and an abstention set;
- grounded extraction may use field evidence and postconditions;
- sampled sentiment may use cross-seed dispersion;
- free-text summarization should not pretend token average probability is calibrated correctness.

Threshold fitting is offline against the user’s calibration split. Runtime evaluation is a small deterministic function. Only after static policies are proven should any adaptive controller be considered, and the existing alien-artifact contract would apply.

Hard constraints can force locally legal but semantically poor trajectories when the model assigns little mass to the feasible region; this is a documented concern in recent structured-generation research, including [Draft-Conditioned Constrained Decoding](https://arxiv.org/abs/2603.03305). That supports measuring constraint pressure, not adopting that paper’s method wholesale.

### Proof obligations

- Every accepted result names the policy and signals that authorized acceptance.
- Signals unavailable on a fast path are `not_computed`, never zero-filled.
- Policy replay is deterministic from the recorded signal vector.
- Locked selective-risk/coverage curves with confidence intervals.
- Cost curves include all retries and discarded samples.
- Review import rejects source/result digest mismatches.
- Correction data cannot silently enter project release scorecards or telemetry.
- A policy can be disabled instantly, falling back to ordinary task output/abstention semantics.

### Why the utility justifies the complexity

This composes calibration, thinking toggles, seeded sampling, batch queues, and NDJSON—all planned already. The new work is a principled policy and review contract. It answers the project’s hardest product objection without pretending the 3B ceiling disappeared.

It ranks sixth because its value is enormous in consequential workflows, but the realized gain is task- and dataset-dependent and must wait for calibration evidence.

---

## 7. Replace per-label teacher forcing with an exact continuation-trie scorer

**Proposed plan integration:** Amend §6.10, §7.0, §7.4–§7.6, OQ-11, and Phase 4. This is exact acceleration and capability expansion for closed-set tasks.

### The idea

The plan’s row-sliced `lm_head` makes single-token labels extraordinarily cheap. Multi-token labels are scored by teacher-forced continuation, but a naïve implementation repeats shared work:

- `account locked`
- `account closure`
- `account verification`

all recompute the `account` continuation independently. Real taxonomies can contain hundreds or thousands of labels, product names, policy codes, entity aliases, or tool names.

Compile every candidate continuation into a token trie and score each unique prefix state once:

1. At a trie node, compute either the outgoing token rows or one full-vocabulary denominator, according to the declared scoring contract.
2. Accumulate the declared path score.
3. Feed each outgoing token once and fork KV/hidden state for its child.
4. Batch sibling/frontier nodes through the layer-major engine.
5. Reuse shared prefixes until paths diverge.

The result must exactly match the frozen naïve per-candidate scorer **under the same scoring semantics**.

### User-facing capability

This changes `classify` from “a few labels” into a credible **large-taxonomy classifier**:

```bash
fnlp classify doc.txt --labels taxonomy.ndjson --top 10
fnlp resolve mentions.ndjson --catalog entities.ndjson
fnlp judge --rubric large-rubric-pack.json
```

It also supports constrained selection from large enum catalogs without free-form generation. The product can solve entity canonicalization and routing workloads that are currently expensive or awkward, while remaining one model and one exact scoring mechanism.

### How it would work

#### A. Candidate compiler

Tokenize candidates under the exact answer scaffold and build a compact trie. Store:

- token ID edges,
- terminal candidate IDs,
- prompt/label-order digest,
- length-normalization rule,
- and any opaque display label separate from the scored continuation.

If one candidate is a prefix of another, termination is represented as an explicit scored terminal/EOS action rather than being smuggled in as a zero-cost edge.

The recipe compiler from Idea 3 can precompile and hash this structure. Frequently used outgoing token rows may be packed into a small candidate projection blob, while the canonical `.fnlpq` rows remain the authority.

#### B. Batched frontier evaluation

Each active trie node owns a COW continuation state. Process nodes in bounded frontiers grouped by depth/context compatibility. After scoring all children—not pruning them—feed each edge token and enqueue its child. Chunk the frontier to the admission budget.

This is dynamic programming over a finite candidate language, not beam search. No candidate is dropped.

#### C. Probability semantics

There are two honest modes:

- **Full teacher-forced logprob:** compute the full-vocabulary denominator once at every **unique trie node**, then reuse that prefix computation for every candidate below it. This exactly matches ordinary per-candidate sequence logprob while eliminating repeated transformer and `lm_head` work for shared prefixes.
- **Trie-conditional scoring:** project only outgoing candidate rows and normalize over the canonical outgoing edge set (including an explicit terminal edge where needed). This is much cheaper, but it defines a different candidate-language distribution; it must never be labeled full-vocabulary likelihood.

The default is chosen per built-in task only after calibration. User recipes declare the mode, and their calibration artifact is keyed to it.

Multi-token score length bias is not solved by pretending one normalization is universally correct. Recipes choose a frozen sum, mean, or other narrowly supported rule, and user-domain qualification determines whether it works.

### Correctness and performance gates

- Exact score/ranking equality against the naïve implementation of each declared mode across random candidate sets.
- Adversarial shared-prefix, Unicode, added-token, and duplicate-display-label cases.
- Candidate-order invariance after canonical compilation.
- Memory bounds for broad/shallow and narrow/deep tries.
- Batch composition cannot change any path score.
- Report trie node count, unique token evaluations, row projections, KV forks, and speedup versus naïve scoring.
- If a label set has little prefix sharing, overhead may lose; the compiler estimates reuse and selects the naïve scorer when measurement says so.

### Why the utility justifies the complexity

The implementation reuses row slicing, tokenizer exactness, COW KV, and layer-major batching. The algorithm is finite, exact, and easy to differential-test. It expands a high-value product surface—taxonomy classification and canonicalization—without adding a new model task primitive.

It ranks seventh because the beneficiary set is narrower than Ideas 1–6, but within that set the win can be dramatic and the proof story is unusually clean.

---

## 8. Investigate cross-loop physical-layer wavefront coalescing—carefully and only for fragmented throughput batches

**Proposed plan integration:** Add a profile-gated scheduler card beside AA-S1 in §10.5; do not make it a Phase 4 requirement. Static fallback: the existing loop-major schedule.

### The idea

The model has 22 physical layers reused in two semantic loops. KV destinations differ by loop, but the q/k/v/o and MLP weights for physical layer `i` are identical. The current compatibility key includes `loop`, so a continuous workload can produce two underfilled groups:

- rows ready at physical layer 7 in loop 0,
- rows ready at physical layer 7 in loop 1.

For the linear work, these rows can be concatenated into one GEMM/GEMV batch, stream physical-layer-7 weights once, then route outputs/KV writes according to each row’s loop tag.

Call this **physical-layer wavefront coalescing**. It does not parallelize the two loops of one sequence; that dependency is inviolate. It coalesces different ready sequences/cohorts occupying different loop stages.

### A useful steady-state picture

```text
stack sweep A: cohort 1, loop 0
stack sweep B: cohort 1, loop 1  +  cohort 2, loop 0
stack sweep C: cohort 2, loop 1  +  cohort 3, loop 0
...
```

Within a sweep, both cohorts traverse physical layers 0…21 together. Their hidden states, cache slots, and completion semantics remain independent.

### The honest upper bound

This is **not** a free 2× improvement and must never be marketed that way.

If an ordinary synchronized batch is already full, it processes M loop-0 rows together and then M loop-1 rows together. Cross-loop coalescing performs the same amount of math and may offer no benefit. Its target is fragmentation:

- asynchronous arrivals,
- mixed prefill/decode morsels,
- cancellations,
- heterogeneous task tails,
- and document-major branches.

The win is the recovered utilization and reduced duplicate weight streaming of partial loop-stage groups. If traces show loop-stage occupancy is already dense, the expected value is zero and the card dies.

### Implementation approach

#### A. Readiness DAG

Represent each row/morsel as a node keyed by `(sequence, token range, loop, physical_layer, phase)`. Edges enforce:

- layer order within a loop,
- loop-0 final norm before loop 1,
- cache-position and token dependencies,
- and sampling before the next decoded token.

The scheduler may coalesce ready nodes only when artifact, physical layer, phase, numerics, activation shape, and attention compatibility match. Loop is metadata, not necessarily a grouping barrier.

#### B. Split linear and stateful compatibility

Projection and MLP kernels are easiest to coalesce because weights and shapes match. Attention may need subgroups for different context lengths/layouts even if q/k/v projection was shared. The planner can therefore:

1. coalesce norm/quant/projection rows,
2. route them through loop-aware attention subgroups,
3. coalesce compatible o-projection/MLP rows again.

Do not force an all-or-nothing fused group that makes attention worse.

#### C. Static policy first

Start with a deterministic rule: coalesce immediately when compatible rows are already ready; never delay a latency request solely to manufacture a cross-loop pair. A later measured throughput profile may permit a tiny bounded wait under daemon mode, but that would require the full scheduler-controller contract and p99/fairness proof.

### Proof and promotion gates

- Bounded-state model checking of the readiness automaton: no skipped layer, duplicate execution, stale KV slot, starvation, or cancellation leak.
- Per-sequence output exactly equals the simple loop-major schedule.
- Batch-invariance with rows repeatedly changing coalescing partners.
- Hostile interleaving replay across loop boundary, EOF drain, timeout, and cancellation.
- Trace-derived occupancy histogram showing a real fragmentation problem before implementation.
- Counters showing fewer partial weight sweeps or higher effective GEMM M, not merely a microbenchmark win.
- End-to-end R2/R3 p50/p95/p99 and energy through thermal steady state.
- Automatic disable when full synchronized batches win.

### Why it makes the top eight

This is the most model-specific idea in the set. It treats “22 weights, 44 semantic executions” as a scheduler opportunity while respecting the exact loop semantics. It could make mixed continuous workloads materially faster and compound Ideas 4 and 7.

It ranks eighth because its benefit is conditional and its scheduler proof surface is real. That is not a weakness in the recommendation: the pragmatic recommendation is to add the **research card and trace instrumentation now**, not to promise the optimization. If the occupancy evidence is absent, preserving the negative result is success.

---

## Winnowing record: the other 22 candidates

The following candidates were genuinely considered. Several were absorbed into broader winners; the rest lost on scope, evidence, or marginal value.

| Candidate | Verdict |
|---|---|
| Copy-constrained source fields | Absorbed into Idea 1 |
| Sparse legal-token `lm_head` projection | Absorbed into Idea 1 |
| Forced JSON-literal micro-prefill | Absorbed into Idea 1 |
| Constraint-pressure telemetry | Absorbed into Ideas 1 and 6; full-mass claims remain audit-only |
| Schema `infer/check/sample` toolchain | Strong near-winner; `check/sample` fit Idea 3, while `infer` should wait for task-quality evidence |
| Per-install `fnlp tune` | Strong near-winner; fold into Phase 6 after AA-K1/static dispatch exists |
| Lazy/scoring-only `lm_head` residency | Measure after Ideas 1 and 7 reveal actual row-working sets |
| Expanded `robot plan` cost/capacity estimator | Already substantially present; amend ergonomics rather than create a separate initiative |
| Eco/balanced/turbo energy profiles | Fold into tuning and performance ledgers after real energy instrumentation exists |
| Persistent prompt-KV snapshots on disk | Defer until recompute time, storage cost, and invalidation traces justify it |
| Semantic postcondition DSL | Bounded useful subset absorbed into Idea 3; reject a general expression language |
| Policy-validated tool-call generation | Expressible as a recipe/grammar pack; `fnlp` still executes no tool |
| Prompt-injection “firewall” | Keep threat-model hardening, but do not promise a detector can make untrusted text safe |
| Streaming tokenizer/zero-copy ingestion | Valuable implementation detail, not a top-level product amendment |
| Incremental map-reduce cache | Absorbed into Idea 2 |
| Artifact canary/rollback | Absorbed into Idea 5 |
| User correction lexicons | Narrow, explicit form absorbed into Idea 6 |
| Delta-compressed model downloads | Premature while public redistribution remains blocked and install frequency is low |
| Semantic document-diff task | Convenient but composable from two recipe runs and deterministic comparison |
| First-class frankensearch RAG bundle | Preserve the NDJSON boundary; do not entangle repositories at v1 |
| Separate scoring-only artifact | Risks artifact proliferation; first measure section residency and row-cache behavior |
| Parallel per-field extraction | Changes model semantics and can damage cross-field consistency; not justified as a default |

## Recommended plan-space sequencing

If these ideas survive review, the cheapest sequencing is:

1. Amend the Phase 4 design now for `TaskIR`, prompt segments/fork graphs, and the Stencil execution primitives.
2. Add proof fixtures and trace fields before optimizing: legal-set sizes, forced-run lengths, candidate-trie reuse, job recovery transitions, and loop-stage occupancy.
3. Build simple deterministic fallbacks first: full projection, ordinary one-token decode, built-in Rust tasks, independent task runs, non-durable batch, naïve candidate scoring, and loop-major scheduling.
4. Promote each optimization independently through L0–L5/invariant gates and per-regime measurement.
5. Productize Assay and decision policies only after task metrics and calibration are real.
6. Let cross-loop wavefront coalescing die without regret if production-shaped traces do not show fragmentation.

The common theme is deliberate: make `franken_nlp` more powerful by compiling more **known structure**—schemas, source text, task recipes, candidate languages, corpus identities, and loop readiness—into exact execution decisions. That is where a one-model, CPU-first appliance can be both radically innovative and obviously pragmatic.
