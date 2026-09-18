# Quantized scoring, classification and ordered batches

This is a library implementation over the existing materialized int8 engine.
It neither activates a production artifact nor adds a binary CLI route. The
artifact bridge remains synthetic/non-authoritative; profile fidelity, model
quality, calibration, performance and release gates remain separate.

## Finite candidate scoring

`native_engine::strict_int8::scoring::Int8CandidatePlan` executes an exact prompt
and bounded candidate language through the existing `CandidateScorer`. It scores
EOS for every candidate and preserves shorter candidates that prefix longer
ones. Its mode selects the existing full-vocabulary sequence-logprob, trie-local
conditional, or explicitly named sequence-score-softmax semantics. None of those
candidate-relative weights is calibrated correctness confidence.

The native adapter prefills the prompt once. Each distinct nonempty continuation
prefix is forwarded once in the scorer's deterministic traversal. It rewinds
completed KV suffixes between branches and recomputes the next branch's hidden
state. It does not clone weights, keep KV forks, cache every hidden state or
replay the entire prompt for every candidate. Selected-row modes really project
only requested head rows. Full-vocabulary mode computes complete denominators.
The low-level leaf has no artifact/identity admission authority; its embedding
caller must bind the exact prompt, language and mode to an admitted task.

Attention work is computed at each branch's actual causal depth, not by treating
all traversed branches as one increasingly long sequence. The plan prices
integer decoder/head dot products and multiply-accumulates, attention pairs,
forwards and projected head rows. Native and scorer work plus total rewinds must
agree with the complete plan before a successful result is returned.

## Raw-text classification

`ClassificationPlanner::plan_int8_with_control` accepts the existing
`ClassificationRequest` and produces `PreparedInt8Classification`. The shared
planner still handles pinned templates, exact untrusted byte encoding, opaque
response codes, original UTF-8 labels and descriptions. The old public planner
continues to admit BF16 only. The new prepared type does not expose its inner
plan for accidental eager execution, and never rewrites a caller's BF16 identity.

The host identity must explicitly name strict-quantized-v1 and the
portable-int8-bf16-rails-eager-gqa-v1 backend. Execution checks the complete
admitted identity and the actual materialized model's revision, recipe and
logical digest. These comparisons do not authenticate the publisher or invent a
packing certificate missing from the source bridge.

Exclusive classification retains the complete label ranking. Multi-label
classification scores a separate yes/no head for each label, allowing multiple
inclusions, exclusions and independent abstentions. No softmax is taken across
labels. A failed head aborts the whole bundle rather than becoming an abstention
or emitting provisional per-label success. The original finalizer and output
schema are reused; the native wrapper adds explicit profile and complete work.

All heads, aggregate work and the engine's complete resident KV allocation are
preflighted before the first forward. Execution flattens only one prompt at a
time and reuses one admitted engine. Head schedules add small geometry records,
not duplicate persistent prompts, scorers or model weights.

## NDJSON integration

`batch::classify::quantized::Int8ClassificationBatchPlanner` uses the same
`ClassificationBatchArgs` and existing `batch::run_ndjson` protocol as the BF16
path. Per-record arguments can only shrink the host's frozen task ceilings;
omitted arguments require explicit host defaults. Model identities, token
sequences, EOS overrides and template code are not input fields.

`NativeInt8ClassificationBatch` takes the compiler, an existing
`StrictInt8Engine`, the host's genuine `Int8ClassificationAdmission` hook and a
whole-adapter `Int8Work` allowance. The hook receives complete integer/attention
work, not merely the transport's forward/head-row pair. Its actual guard stays
with the result through serialization, writing and flushing.

The adapter charges all five work dimensions atomically before admission and
execution. Neither failed attempts, explicit flushes nor separate runner calls
refund them. Fatal failures and unwinds prevent adapter reuse. A prepared bundle
from another host/limit factory is refused before admission. The transport's
existing work limits independently constrain the same work; they do not mint a
second allowance or replace process-wide host admission.

Failed or partial writes stop the runner without retrying, reading another
record, or appending an error to uncertain output. The transport does not report
delivery back to this adapter, so a sink failure does not itself change its
native-ready state; its attempted work remains charged. Any later runner use or
retry after uncertain delivery is an explicit host decision, not exactly-once
or remote-durability assurance. The host also owns aggregate weights, memory,
thread and scheduling admission; no new broker or runtime is created here.

## Verification boundary

This increment adds 38 Rust regression tests: 14 candidate-scoring tests, 10 raw
classification tests, and 14 admission/batch tests. They exercise shared code
with synthetic logits or clearly named transport-only payloads; they do not
produce public fake-native execution receipts. Rust compilation and those tests
have not been run in this session. Native model execution remains unverified.

Independent Python checks executed 40000 schedule comparisons, 20000 actual
scorer-insertion traversals containing 2717994 prefix queries, 59049 exhaustive
five-axis admission cases, and 25000 multi-call ledger sequences. These are
specification oracles, not execution of the Rust implementation or evidence of
model accuracy. They do not replace clean-SHA DSR or model-present qualification.
