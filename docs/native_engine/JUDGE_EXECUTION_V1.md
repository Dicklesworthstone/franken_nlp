# Native pairwise, rubric and faithfulness judging

The former `tasks::judge` stub now has three implemented task paths: two-order
pairwise preference, independent ordinal rubric criteria and source-bound
faithfulness. They accept compiled judge-v1 TaskPlans or raw text through
`JudgePlanner` and reuse CandidateScorer and the admitted HfBf16EagerEngine.

This is source implementation, not runtime qualification. Compilation, builds,
tests, harnesses, DSR, real-model quality, conformance and benchmarks have NOT
been run for these changes. No inference CLI command, model activation gate or
process-admission bypass is enabled. The native/library faithfulness and
experimental extraction second-reader APIs are described in
[Faithfulness and semantic extraction](FAITHFULNESS_AND_SEMANTIC_EXTRACTION_V1.md).
The planned `extract --verify-semantic` CLI flag is not activated here.

## Pairwise preference

`PairwisePlan::from_task_plans` requires exactly two plans: display original
answers A/B, then display B/A. Each plan has the strict segmented ABI:

    global / instruction / criterion-data / instruction / first-answer-data /
    instruction / second-answer-data / answer-scaffold

The compiler checks that reversing presentation swaps ONLY the answer data.
The criterion, trusted instructions, scaffold and first/second verbalizers
must match exactly. Documents and verbalizers cannot contain any ID from the
caller-supplied archived TemplateControlIds, including nonspecial markers.

Every finite continuation includes an explicitly scored EOS. Both orders use
full-vocabulary sequence log probabilities: all vocabulary rows contribute to
each denominator. There is no silent switch to sparse conditional scoring,
raw terminal logits or a one-token label assumption.

For the A/B presentation, let d0 = log P(first) - log P(second). For the B/A
presentation, remap to ORIGINAL identities before computing
 d1 = log P(second) - log P(first). The reported margin is (d0+d1)/2.
This averages oriented log ratios, not already-normalized probabilities.
The conditional weight for original A is logistic((d0+d1)/2), a descriptive
candidate-set projection, NOT correctness confidence.

A pure first-position preference cancels in the mean. The result also retains
abs(d0-d1), so cancellation cannot hide presentation-order disagreement.
Explicit, uncalibrated minimum-margin and maximum-disagreement thresholds
can abstain. An exact zero mean within the disagreement policy returns a tie;
a nonzero margin below the threshold abstains rather than claiming equality.
Both complete candidate score sets and all algorithmic work are retained.
A failed second order returns no preference at all.

## Rubric scoring

`RubricPlan::from_task_plans` accepts at most sixteen named criteria, each
with a positive bounded integer weight, and a complete shared 0..N scale
where 1 <= N <= 10. Empty, duplicate or malformed criterion IDs, zero or
oversized weights, incomplete candidate grids and divergent document/scaffold
bindings fail before inference. Criteria are ordered by stable identifier.

Each head's ABI is:

    global / instruction / criterion-data / instruction / document-data /
    answer-scaffold

Only criterion-data differs between heads. Every criterion receives its OWN
full-vocabulary finite distribution. Criteria never compete in one softmax.
Host Rust computes per-criterion expectations, normalized entropy and numeric
modes. Exact modal ties prefer the lower numeric score, not lexical label
order; raw sequence scores prevent exponential underflow from inventing ties.

The host computes weighted expected and modal scores on the shared scale.
A criterion can abstain under explicit peak-weight/entropy thresholds. If ANY
criterion abstains, accepted aggregate scores are absent. Diagnostic weighted
means still include ALL criteria; the implementation never drops abstentions
and renormalizes the remaining weights into an inflated accepted result.

Raw `RubricDefinition` carries schema_version=1, a revision, a declared origin
digest, a scale maximum and criteria with descriptions and integer weights.
These are caller-owned local rubrics, not provenance-cleared shipped presets.
Origin declarations, definitions, order-independent criterion metadata, scale
and weights are bound into the private identity. They do not create a quality
qualification, calibration artifact or provenance certificate.

## Faithfulness

`FaithfulnessPlan` scores the claim against the complete source and, when the
source needs more than one bounded window, every window in its exact partition.
It returns an entailed/contradicted/unsupported model relation or an explicit
policy abstention. Accepted support or contradiction requires matching local
evidence and no policy-passing contrary window. Quotes and byte/scalar offsets
are independently checked against original source bytes. Membership is not
proof of semantic support. Missing local evidence and conflicting judgments
abstain; timeouts or failed heads remain errors rather than unsupported labels.

## Raw text and identity

`JudgeRequest` is a strict tagged enum with `pairwise`, `rubric` and
`faithfulness` modes. `JudgeRequest::from_json(source, max_request_bytes)` checks
input size before JSON parsing, rejects duplicate keys through canonjson, and
rejects unknown fields through the typed request schema.

`JudgePlanner::pinned(controls, eos)` constructs reusable pinned tokenizers and
fixed templates with the existing TemplateBuilder. Thinking is disabled and
no tools are declared. The planner binds all ten rubric-scale template variants,
faithfulness scaffolds, exact answer continuations, archived controls, EOS and
all four pinned tokenizer assets in a content-free template digest. The expanded
prompt-family identity is `judge-segmented-pairwise-ordinal-faithfulness-v2`.
Old prepared identities do not silently gain the new template binding.

The renderer sees ONLY trusted instructions and internal placeholders.
Criteria, candidate answers, rubric descriptions, sources and claims use the
existing byte-preserving untrusted encoding path. Their literal role/thinking
marker spellings remain ordinary bytes. Typed token segments are never flattened
and retokenized; the first trusted segment inserts BOS once, later fragments
suppress BOS/EOS insertion. Numeric, A/B and E/C/U continuations are exact
byte-fallback tokens, without standalone BPE dummy-prefix assumptions.

Before source token allocation, planning counts source BYTES (one fallback
ID per byte), all trusted overheads and every presentation order, criterion or
evidence head against per-head and aggregate prompt ceilings. Source text is
not trimmed, normalized or silently truncated. Private encoding diagnostics
map to fixed safe categories rather than exposing source context windows.
Marker containment does not prove immunity to semantic prompt injection.

The caller sets planner template/tokenizer digests on its judge-v1 execution
identity before constructing PlanContext. Planning checks the task, tokenizer,
template, eager numerics profile, BF16 KV, disabled thinking and no tools.
TaskPlan compilation retains the context's budget ceilings.

The resulting `PreparedJudge` retains a complete immutable ExecutionIdentity.
It fills task-owned aggregate TaskIR, exact prompt, schema and decision-policy
bindings plus the scoring version; artifact/model/backend authority remains
caller-owned. The caller admits its engine against that identity. Execution
compares the supplied admitted identity with the entire retained identity BEFORE
callbacks or engine mutation; mismatches are refused, not repaired. Neither
private identities nor unkeyed prompt/source digests enter judge results.
Calibration remains explicitly unqualified.

## Native execution and integration

The adapter preflights every head's context capacity, complete engine KV
reservation and aggregate forward/projection work before the first forward.
Each exact prompt prefills once; EagerPrefixSession reuses continuation prefixes
within that head. Between distinct heads, logical KV clears while retaining
existing buffers and weights. No second KV cache, model clone or runtime is
introduced. A nonempty caller cache is refused without mutation.

For head i with P_i prompt tokens and U_i distinct nonempty continuation
prefixes, native forward work is sum(P_i+U_i). Full-vocabulary projection work
is sum((U_i+1)*166144). EOS contributes projection edges but not KV forwards.
These are algorithmic counts, not measured throughput. Each head receives its
exact share of the preflighted aggregate budget. Actual forwards, prefix calls
and projected rows must match the plan before a result is returned.

`PreparedJudge::planned_native_budget` reports this cold-head resource bound.
`preflight_eager` checks the identity, empty engine, every head's context and
full KV reservation without mutation or callbacks. Multi-field callers use
it to check all later requests before starting the first request; it does not
replace artifact/model/process admission.

Cancellation uses the caller's prefill channel because candidate scoring does
not commit generated tokens. Typed cancellation/native causes survive wrapping.
The existing prefix-session drop/error cleanup owns logical KV reset. Failure
in any head aborts the task; no partial success is emitted. Output ceilings
cover raw task results, the tagged JudgeResult, and the full EagerJudgeRun
wrapper including native-work metadata.

Integration sequence:

1. Construct `JudgePlanner::pinned` with the caller's archived controls and EOS.
2. Set `task_spec="judge-v1"`, planner template/tokenizer digests and explicit
   eager/no-thinking/no-tool facts on the caller-owned base identity.
3. Create PlanContext, then call `planner.plan(&request, &context, limits)`.
4. Retain `prepared.execution_identity()` privately and admit the engine
   against it through the existing caller-owned admission path.
5. Call `prepared.execute_eager_with_control(&admitted_identity, &mut engine,
   prefix_budget, &mut control)`. Non-native providers implement JudgeLogits
   and use `prepared.execute` with the same identity check.

The original twenty pairwise/rubric/native/planning source regression tests
are retained. Additional faithfulness and extraction cases are documented in
the companion note. All remain UNRUN. They do not establish task accuracy,
calibration, positional-bias removal on real models, provenance clearance,
factual-support quality, or performance superiority.
