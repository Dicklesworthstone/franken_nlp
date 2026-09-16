# Native prefix-reuse candidate scoring

`native_engine::hf_bf16_eager::candidate_scoring::classify_eager_cached`
and its `_with_control` form execute the existing validated ClassificationPlan
with one live native KV sequence. They require an already-admitted empty eager
engine and never load/activate a model or create another runtime.

The engine prefills the exact TaskIR prompt once without projecting discarded
intermediate logits. Each distinct continuation prefix executes one further
native forward. Siblings reuse their longest common prefix by truncating all
44 completed KV slots in place. This is destructive backtracking, not a retained
fork: there is no second KV buffer, model-weight clone, saved logit table, or
claim of cross-request prefix-cache authority. The same 44-binding loop runner,
layer primitive, bf16 cast schedule, attention and RoPE implementation are used.

Full-vocabulary sequence probability still computes every vocabulary row at
each scoring prefix. Trie-conditional and raw sequence-score modes project only
the requested ascending distinct matrix rows, including each required EOS edge.
EOS is scored but not fed into KV. The result retains the existing score-space,
normalization, candidate-completeness and uncalibrated-policy disclosures.

The reusable `EagerPrefixSession` also implements CandidateLogits for other
finite tasks. Its prompt is immutably borrowed and its engine exclusively
borrowed for the complete session. Repeated identical prefix requests reuse the
last hidden state; revisiting an ancestor recomputes just its final token rather
than retaining hidden states for every trie node. Arbitrary query order is
supported, subject to the aggregate real-work budget.

`PrefixBudget` bounds actual forward positions and actually projected head
rows. Classification preflights its exact native-work bound before compute.
For a prompt of P tokens, U unique nonempty continuation prefixes and C complete
candidates, the deterministic traversal performs P+U forwards. Full-vocabulary
mode projects (U+1)*166144 rows; selected-row modes project U+C rows. These are
algorithmic work counts, NOT measured throughput or a model-quality receipt.

Every operation is charged before native work or callbacks. The session is
poisoned before mutation, preventing failed/cancelled/unwound work from being
retried against refunded budgets. Cancellation is polled between logical layers,
loop norms and bounded head-row tiles through the prefill channel. Dropping the
session clears logical KV positions on success, ordinary failure or unwind,
while retaining engine-build buffers. An incomplete 44-slot token cannot be
rewound into apparent validity. Storage truncation is not secret zeroization.

`CachedClassificationRun` names `eager-single-kv-prefix-rewind-v1`, keeps the
`hf-bf16-eager` numerics label, and records actual completed forwards, prompt and
continuation positions, projected logits, prefix evaluations and rewinds. The
complete result envelope is checked against the task's output-byte budget.
The original replay functions and receipt remain available unchanged.

Source regression tests cover selected-row reference bits, exclusion of
unrequested rows, KV prefix/capacity preservation, failed rewind atomicity,
work bounds, arbitrary-prefix scheduling and prefill cancellation routing.
Compilation, tests, DSR, real-model equivalence and performance measurements
have NOT been run for this code-first change. Artifact OQ-31 activation and
platform authority gates remain separate and unchanged.
