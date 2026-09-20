# Native INT8 judge execution

`JudgePlanner::plan_int8_with_control` compiles the existing raw `JudgeRequest`
into `tasks::judge::quantized::PreparedInt8Judge`. Its `execute_with_control`
method runs against an already admitted `StrictInt8Engine`, complete
`Int8ScoringBudget`, exact admitted identity and caller control. It returns an
`Int8JudgeRun`, or a typed `Int8JudgeError`; there is no partial judgment result.

The three existing tasks share the same native path:

- Pairwise: both presentation orders are scored before the order-aware decision.
- Rubric: every criterion is scored and the existing weighted/abstention policy
  finalizes the complete set in canonical order.
- Faithfulness: the full source and every bounded evidence window are scored;
  contrary or uncertain windows remain visible. The existing finalizer owns
  byte-verified evidence, conflicts and abstention. It never certifies truth.

This is a real INT8 candidate-scoring route, not a BF16 executor with a different
label. The raw planner builds its TaskIR under the supplied strict-v1 identity
and pinned backend. A private inner plan shares the existing template, exact
byte-preserving data encoding, finite languages and semantic finalizers; callers
cannot convert it into an eager plan. The original eager planner retains its
profile restriction. Scoring includes the full-vocabulary denominator and EOS.

The native schedule prices each distinct prefix, all decoder projections,
attention geometry and head rows. Independent heads reset logical KV while
retaining one allocation. Context is the maximum live head, not the sum of all
forward positions. Aggregate allowances are checked before any forward; native
head execution receives exact slices rather than renewable whole-run budgets.
Actual counters, result version/profile and complete candidate scores are
validated before finalization. One failed or cancelled head fails the task.

Cancellation checkpoints surround planning and occur between heads and before
and after finalization. Existing tokenization is not interruptible mid-encode;
no cancellation-latency claim is made. The native scorer retains its deeper
cooperative checkpoints and poisoned-engine cleanup rules. Model identity uses
actual materialized source facts without claiming publisher authentication.

The new regression source covers all three modes, shared prompt semantics,
profile and model mismatches, every native work axis, malformed score envelopes,
missing denominators/EOS/candidates, partial-head failure, cancellation and total
output bounds. Synthetic logits exercise the actual shared scorer/finalizers,
not model fidelity. These Rust tests were written but not run in this session.
No real-model quality, throughput, artifact-activation or public CLI gate is
promoted by adding this route.
