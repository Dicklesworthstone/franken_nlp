# Native INT8 corpus entity resolution

`ResolutionPlanner::prepare_int8` in `corpus::native_resolve::quantized`
connects the existing source-anchored resolution graph to the actual strict-INT8
finite-candidate scorer. It does not relabel an eager plan or introduce another
model, tokenizer, renderer, runtime, retry policy or clustering implementation.

## Semantic contract

`ResolutionPlan::prepare` checks each mention against its original document and
byte/Unicode-scalar coordinates. Lexical blocking only proposes comparisons;
matching names are never enough to authorize a merge. Both presentation orders
score the exact same/different/uncertain language, including EOS, with full
vocabulary denominators. Both orders must independently satisfy the existing
margin rule. Order disagreement produces abstention, not an averaged match.

All pair assessments reach the existing deterministic complete-link finalizer.
A=B and B=C cannot override A!=C, uncertainty, or an unexamined A/C pair. Results
retain original mention coordinates and the existing warnings: lexical blocking
can miss aliases, scores are uncalibrated, and cluster identifiers belong only
to the exact snapshot. This implementation does not establish entity accuracy.

## Complete planning and native execution

`Int8ResolveLimits` combines the existing finite pair/prompt/context limits with
explicit ceilings for forward positions, projected logits, attention pairs,
integer dot products and multiply-accumulates. All pairs and both orders are
compiled before any forward. `CandidateSchedule` prices the actual prefix-trie
traversal: sibling labels rewind to the prompt rather than extending a single
causal sequence. Each native head receives its exact slice, never a renewed
copy of the entire snapshot allowance.

The host admits every identity from `execution_identities()`. The complete
identity vector, actual materialized model, resident KV capacity and every head
schedule are preflighted before the first forward. Execution consumes the
prepared snapshot, verifies complete native receipts, and returns no partial
graph on failure. The eager entry point remains eager-only; the shared compiler
is private and no conversion exposes the INT8 core to eager execution.

A graph without candidate pairs can use `finalize_without_model`. It still
performs the real source-verified finalization and observes cancellation, but
records `model_evaluated: false` and zero native work. This method refuses a
graph containing any required comparison; it cannot authorize lexical merges.

## Process-hosted API

`NlpEngine::resolve_int8` takes a resident model, owned `Vec<ResolutionDocument>`,
a pinned `Arc<ResolutionPlanner>`, `hosted::ResolveConfig`, and one cancellation
token. One existing process-owned invocation covers source validation, blocking,
planning, all scoring and clustering. Native KV/RoPE/scratch is reserved once,
after complete planning; zero-candidate graphs allocate none of that storage.

The host charges actual nested Vec/String capacities, not just visible input
lengths. Caller-priced preparation and graph reservations must cover retained
prompts, identities, schedules, lexical blocks, scores, clustering, metadata and
allocator overhead. These modeled commitments are not measured RSS or allocator
interception. The complete output has a separate charge retained by
`HostedOutput<Int8ResolutionRun>` through caller serialization/delivery. Inputs,
plans, native state and transient graph storage drain before physical completion.
Typed `HostedError::Resolution` retains cancellation and failure causes while
default diagnostics omit private source text.

## Evidence boundary

Regression sources exercise real pinned planning, private synthetic score
receipts, the real complete-link finalizer, host geometry and ownership types.
Rust compilation/tests, production DSR, actual model inference, task quality and
performance have not been executed or established by this change. Controller
validation remains governed by `WIRING.md`. No artifact activation, neural CLI,
durable-job, calibration or phase gate is promoted.
