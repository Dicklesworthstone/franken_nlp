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

## One persistent team for the scored-task portfolio

With `asupersync-runtime`, the following explicit APIs run the complete
request on one fixed INT8 output-row team:

| Task | Sequential prompt | Layer-major prompt |
| --- | --- | --- |
| Classification | `execute_int8_classify_with_threads` | `execute_int8_classify_layer_major_with_threads` |
| Judgment | `execute_int8_judge_with_threads` | `execute_int8_judge_layer_major_with_threads` |
| Sentiment | `execute_int8_sentiment_with_threads` | `execute_int8_sentiment_layer_major_with_threads` |

Classification and judgment take the resident model, sealed prepared plan,
`NativeLimits`, kernel, compute width and cancellation token. The layer-major
variant inserts `Int8PrefillLimits` before the cancellation token. Sentiment
uses `SentimentHostLimits` in place of `NativeLimits`, retaining its complete
work ceiling and additional preparation charge. Width must be 2 through 64 and
fit the installed host envelope; width 1 uses the existing serial APIs.

One private executor admits only these existing sealed plan types. It keeps
their original score spaces, full denominators, EOS and semantic finalizers.
Context admission uses the largest live head while work admission covers all
heads. Successful completion requires exact work and head counts, empty KV and
an unpoisoned native engine. Prepared inputs and all native/output reservations
remain owned through physical worker join, including cancellation and panic.
The returned `HostedInt8TeamResult<T>` keeps the usual result-memory guard and
exposes completed physical work separately from semantic output.

Coordinator prompt scratch and fixed child prompt capacity are separately
derived and reserved before the first projection. The existing exact kernels
and their reduction order remain unchanged. Nine model-free regression
definitions and one ignored real-artifact differential gate were added for this
route. They remain **unrun** pending DSR and model qualification.

## Complete-document classification on one team

`fnlp candidate classify-map` now exposes the same physical schedules for a
complete document, with a team reused across every chunk and classification
head:

```sh
fnlp candidate classify-map request.json --model ./model.fnlpq \
  --memory-mib 12288 --compute-threads 4 --prefill-rows 4
```

The request remains `{document,labels,mode?,policy?}`. Both exclusive and
independent multi-label modes retain their existing full-vocabulary scores,
thresholds, exact source coordinates and source-scalar-weighted aggregate.
Every chunk and all five work counters are admitted before loading weights.
The complete-document work allowance never renews at a chunk or head boundary.

`--compute-threads 1..=64` counts the numerical coordinator and its row workers;
runtime infrastructure is additional. Width one retains serial execution.
`--prefill-rows 1..=64` selects bounded layer-major prompt staging with either
serial execution or the same persistent team. Omitting both flags preserves
the original serial schedule and JSON response. Compact document encoding
continues to use the declared cumulative preparation policy.

Admission prices the largest native context, one coordinator prompt workspace,
fixed worker buffers, both preparation owners, all retained chunk results and
final output staging before request/model IO. KV and prompt storage are reused
across chunks. All claims survive physical worker join on success, failure,
cancellation and panic. A failed chunk yields no successful document result.
After join, threaded responses add a separate `compute` receipt; the sealed
classification body and source binding remain unchanged.

The hosted entry points are `map_int8_classify_with_threads` and
`map_int8_classify_layer_major_with_threads`; both accept the complete request,
base identity, host config, encoding policy, kernel and compute width. The
layer-major variant also takes `Int8PrefillLimits`. Serial prefill uses
`map_int8_classify_layer_major_with_encoding`. The shared memory quote's
`with_prefill` method adds coordinator storage, while worker storage remains
separately admitted.

Regression definitions cover original response shape, schedule-independent
work and seals, exact memory boundaries, stale geometry, cancellation and
guard lifetime. An ignored real-artifact gate compares complete serial,
prefill and team results and checks underfunded work and physical drain.
Compilation, executable regressions and that model gate remain pending the
required DSR checkpoint. No model parity or measured speed is established.
