# Executable dimensional sentiment

The `tasks::sentiment` surface now has text planning, finite-continuation
scoring and native eager execution. It replaces the former sentiment stub.
This is code-first implementation: compilation, tests, DSR, real-model
conformance, sentiment accuracy and throughput have NOT been measured here.
It does not enable an artifact loader or an inference CLI bypass.

## Text to exact task plans

`SentimentPlanner::pinned(controls, options)` retains the pinned embedded
tokenizer and the caller's archived TemplateControlIds. It compiles four fixed,
versioned task prompts using the existing Nanbeige TemplateBuilder, with
thinking disabled and no tools. The builder receives only trusted instructions
and an internal slot marker, never the request's document.

`SentimentRequest` contains the UTF-8 document, a unique nonempty subset of
valence/arousal/dominance/approach, and an explicit TaskBudget. The planner
checks source byte counts and aggregate per-axis prompt overheads before
encoding. It neither truncates nor normalizes the document. The existing
UntrustedDocumentEncoder supplies byte-preserving, control-free IDs; those IDs
are composed with trusted global/instruction/scaffold token segments without
flattening and retokenizing. Source marker spellings remain source bytes.
Marker containment is not a guarantee against semantic prompt injection.

The first trusted segment inserts the pinned BOS once. Later trusted fragments
suppress BOS/EOS insertion. Five exact byte-fallback candidates represent -1,
-0.5, 0, 0.5 and 1; no standalone BPE dummy prefix changes their output bytes.
The scorer appends and scores the explicit configured EOS separately.

The caller sets the content-free planner template/tokenizer digests on its
execution identity before constructing PlanContext. Planning verifies those
digests, sentiment-v1, the eager numerics profile, disabled thinking and no tools.
TaskPlan construction applies the context's budget ceilings to every head.
The compiled bundle's private binding digest additionally includes all exact
TaskIRs, axes, coordinate mappings, score rules and abstention policy. It is
prompt-derived private state, not a public result or telemetry field.

## Independent dimensions, not one emotion softmax

`SentimentPlan::from_task_plans` also accepts already-compiled sentiment-v1
heads for trusted orchestration. It requires the same document/global-policy
sequence and distinct exact prompts across axes. Each head's finite candidates
must map exactly once to a uniform, centered, zero-containing grid in
[-1000,+1000]. Values are integral in the plan and normalized to [-1,+1] only
when projecting results.

The interpreted low/high anchors are negative/positive valence,
calm/activated arousal, powerless/in-control dominance and withdrawal/approach.
These are task interpretations, not validated psychological measurements or
access to the author's mental state. Dimensions never compete in a joint
softmax. Each complete candidate set is normalized independently by the same
shared CandidateScorer used for classification.

Results retain every candidate, scored EOS, score space, denominator disclosure
and coordinate mapping. Full-vocabulary sequence probability includes all
vocabulary rows at each scoring prefix. Trie-conditional and raw sequence-score
modes remain distinct named spaces. The reported moments describe conditional
candidate weights, not calibrated posterior uncertainty or correctness.

Callers explicitly choose minimum peak-weight and maximum normalized-entropy
thresholds. A dimension that fails the policy returns `abstained` with no
`estimate`, while preserving its distribution and descriptive moments. A
diffuse symmetric distribution therefore need not masquerade as certain
neutrality. A genuine modal tie uses the lower coordinate; raw sequence scores
prevent exponential underflow from inventing ties.

## Native execution and budgets

`SentimentPlan::execute_eager_with_control` runs the complete bundle on one
already-admitted HfBf16EagerEngine. It preflights all heads' context capacities,
the entire engine KV reservation and aggregate native work before any forward.
Each axis prefills its own exact prompt once and uses EagerPrefixSession for
candidate branch reuse and selected-row projections. Different axis prompts
are not interchangeable: logical KV is cleared between axes, retaining the
existing buffers and weights. No second KV cache, weight clone or runtime is
created, and no cross-axis prefill-reuse performance claim is made.

For axis i with P_i prompt tokens and U_i distinct nonempty continuation
prefixes, native forwards total sum(P_i+U_i). Full-vocabulary mode projects
sum((U_i+1)*166144) rows; selected-row modes project sum(U_i+C_i), including
one EOS edge for each of C_i completed candidates. These are algorithmic work
counts, not benchmark results. Aggregate limits are partitioned into exact
per-head allowances, rather than renewed independently for each axis.

Cancellation retains the caller's typed cause and uses the prefill channel:
scoring does not commit generated tokens. Prefix-session failure/unwind cleanup
clears logical KV while retaining preallocated buffers. Any failed dimension
rejects the whole bundle; there is no partial-success response. Both the task
result and the full native response envelope are checked against the output
byte ceiling. A caller's nonempty cache is refused without modification.

The native response names `eager-independent-affect-prefix-heads-v1` and reports
completed forward/projection counts checked against the plan. Existing
artifact, model, source-trust, process-admission and qualification gates remain
separate. The low-level native call assumes the caller has bound its admitted
engine to the task identity; a TaskPlan or template digest alone is not model
activation authority.

## Integration sequence

Given caller-owned controls, an appropriate execution identity and an admitted
engine: construct the planner, set its template/tokenizer digests on the identity,
construct PlanContext with a TaskBudget ceiling, compile SentimentRequest using
`planner.plan`, and call `plan.execute_eager_with_control` with a PrefixBudget
and the caller's DecodeStepControl. For a non-native provider,
`SentimentPlan::execute` accepts SentimentLogits and supplies the exact typed
prompt and axis on every projection request.

Seventeen source regression tests were added across distribution scoring,
native budgeting and text planning. They cover axis independence, diffuse
abstention, deterministic order, source/prompt isolation, coordinate grids,
aggregate resource ceilings, full denominators, no partial success, native
work accounting, cancellation causes, fixed rendering, byte-preserving source
markers and exact numeric continuations. They have not been executed.
