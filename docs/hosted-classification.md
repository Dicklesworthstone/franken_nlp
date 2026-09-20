# Process-hosted INT8 classification

With `asupersync-runtime`, `NlpEngine::execute_int8_classify` now executes a
`PreparedInt8Classification` against a charged `ResidentInt8`.
`NlpEngine::batch_int8_classify` accepts an `Arc<ClassificationPlanner>`, a
`ClassificationCorpusConfig`, the existing `CorpusLimits`, and owned input/output.
The config is exported from `hosted::corpus`.

Both routes use the existing full-vocabulary candidate scorer, scored EOS,
exclusive and independent multi-label semantics, and complete-result finalizer.
They do not generate labels and parse model prose. Native model identity and
all heads are checked before forwarding. The single-result route retains the
original `Int8ClassificationError` through `HostedError::Classification`.

One resident engine and KV allocation serve the whole corpus. Context capacity
is the largest live head, while native work sums all heads and all documents.
Forwards, projected logits, attention pairs, dot products and multiply-accumulates
remain nonrefundable across failures and epoch flushes. A classification with
many heads may legitimately consume more total forwards than the context cap.
The host charges the full allocated KV, not only the prompt currently in use.

The corpus's concrete admission provider uses the real process ledger. It
conservatively reserves the common task output-byte ceiling per document; this
avoids inferring an allocation ceiling from projected-logit counts. Requests
cannot enlarge that ceiling. Output storage retains its reservation through
serialization and write/flush. KV and scratch are not repeatedly charged as
per-document copies. The existing bounded NDJSON runner and argument schema are
unchanged; check both host completion and `BatchSummary.failed`.

The existing hosted lifetime, restricted-context, one-blocking-coordinator and
physical-drain rules still apply. Preparation and IO estimates must remain
honest; these are modeled commitments, not measured RSS limits. The inherited
batch preparation and blocking IO have noninterruptible portions. No public
model-backed CLI, artifact activation, numerical qualification, parallel batch
speedup or durable-delivery claim is added.

Regression cases cover exact/oversized contexts, whole KV accounting, independent
head work, output admission limits, typed error retention and Send ownership.
They are source additions, not recorded Rust test passes. Compilation, runtime
execution and real-model quality/performance evidence require the authorized
validation environment.
