# Native INT8 dimensional sentiment

`SentimentPlanner::plan_int8_with_control` now compiles the existing raw-text
`SentimentRequest` into `PreparedInt8Sentiment`. The context must already name
`strict-quantized-v1`, the exact `STRICT_INT8_EXECUTION` backend, BF16 KV and the
pinned planner's template/tokenizer identities. There is no BF16 relabeling,
plan conversion or new tokenizer/template/scorer implementation.

The prepared value seals the whole execution identity, including canonical
axis order, exact prompt segments, complete candidate languages and anchors,
EOS, scoring mode, policy and result cap. Model facts remain caller-owned;
execution verifies the admitted identity and actual materialized model before
any forward. These private content-derived bindings are not public telemetry.

`execute_with_control` takes the existing `StrictInt8Engine` and
`Int8ScoringBudget`. It preflights every axis before the first forward and
executes each axis through the existing prefix-first candidate scorer. KV is
cleared between dimensions; one admitted context is reused, not cloned. The
maximum live context is the largest head, while forward positions, head and
decoder projections and causal attention are summed across independent heads.
Each head receives only its exact slice, not a refreshed whole-task budget.
Every candidate includes its EOS score. Full-vocabulary, trie-conditional and
sequence-softmax modes retain their distinct score spaces.

The existing dimensional finalizer checks the complete candidates, anchors,
normalization and work. There is one distribution per requested axis, never a
softmax across valence/arousal/dominance/approach. Distributions and abstention
policies remain explicitly uncalibrated and are not psychological measurements.
A failed or cancelled axis yields no result for the bundle. The complete
`Int8SentimentRun`, not just its inner result, must fit the result-byte cap;
a final cancellation check precedes return. No task error becomes neutral
sentiment or a successful abstention. Native session cleanup remains RAII.

This is code-first implementation. Added tests use the real pinned compiler
and private synthetic candidate-score fixtures for identity, exact source
containment, all-axis accounting, score-space/finalizer contracts, failure,
cancellation and output limits. No Cargo/Rust test, real-model, numerical
fidelity, performance, release or controller DSR qualification was run in the
implementation environment. The existing candidate artifact status is not
promoted. Process hosting and CLI admission are separate integration layers.
