# Explicit INT4 structured cohorts

`candidate source-int4 extract-batch`, `ner-batch`, `keyphrases-batch`,
`summarize-batch` and `answer-batch` accept `--cohort-documents N` to execute
independent documents through one native shared-layer INT4 cohort. `N` is a
ceiling from 1 through 64. The model remains resident across windows, and each
window allocates its exact number of sequence contexts. Omitting the flag keeps
the serial path.

This is an explicit candidate path. The implementation and physical grouping
counters do not establish numerical parity, task quality, or measured speedup.
Those claims require separate controller-selected validation and model evidence.

## Command line

For an independently pinned existing snapshot:

```sh
fnlp candidate source-int4 extract-batch documents.ndjson \
  --schema extraction.schema.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 --context-tokens 1024 \
  --max-new-tokens 128 --cohort-documents 4
```

The source-conversion `--source-dir` and `--source-manifest` route is also
supported. `--source-membership` binds annotated verbatim fields to each
document's original bytes. The schema remains one exact shared declaration;
non-null per-record `task_args` are rejected, including attempts to repeat or
replace the schema or budget.

The four source-task streams use the same option and window quotation:

```sh
fnlp candidate source-int4 ner-batch documents.ndjson \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 --context-tokens 1024 \
  --max-new-tokens 128 --cohort-documents 4
```

Their existing typed `SourceBatchArgs` defaults and per-document overrides
remain available. Overrides may tighten task budgets and change bounded task
options, but cannot switch the stream's task kind or model. For `answer-batch`,
each record's `text` is the question; its separate `task_args.passages` supplies
the original evidence. Questions, passage IDs, quotes and repeated-occurrence
coordinates stay independent across rows. No QA passage set is invented.

Input uses the existing ordered NDJSON protocol:

```json
{"id":"first","text":"Alice works in Shanghai."}
{"id":"second","text":"The invoice is due Friday."}
{"flush":true}
```

A window executes when its document/error capacity is reached, at an explicit
flush, or at EOF. Interactive producers must fill a window or send a flush
before waiting for responses. There is no timer-driven partial-window execution.
Malformed or rejected records occupy their original sequence positions. The
duplicate-ID set resets only after the drained flush event has been delivered.
The last input record need not end with a newline.

Each document retains an independent prompt, grammar/source language, EOS,
causal KV, and original task budget. Grouping does not widen a document's
`TaskBudget` or change its TaskIR identity. Full-vocabulary greedy token
selection remains the same: this option introduces no sparse projection,
sampling, repair retry, or prompt-prefill selector.

## Memory, work, and delivery

The CLI and hosted stream use the same checked memory quote. It includes all
declared simultaneous native contexts, shared activation/RoPE storage, the full
prepared-document window plus one lookahead, source/grammar runtime state,
decoder cursor workspace, transport and ID staging, and every retained result's
output reservation. The maximum complete cohort result, including its envelope,
is 64 MiB. A requested window that cannot fit refuses before input/model IO;
its width is not silently reduced.

`--preparation-mib` continues to be the per-document preparation estimate in the
stream. A cohort prices `N + 1` such slots in addition to outer CLI preparation.
Native and output storage are separately accounted. These are modeled payload
and ownership limits, not an operating-system RSS guarantee.

The five native work counters and the mask-visit allowance apply to the entire
stream. A new document, cohort, or flush does not renew them. Native reservations
and output guards remain owned through physical completion and writer flush.
Failed native execution or independent finalization returns no partial group
of successful results. Earlier fully delivered groups remain visible. A failed
output write or flush stops the stream and never appends another terminal frame.

Document results keep their existing extraction or source-task schema and
semantics. Outer
candidate provenance adds only the declared schedule:

```json
{
  "cohort_schedule": {
    "execution": "cross_document_constrained_cohorts",
    "declared_max_documents": 4
  }
}
```

Inner records identify the transport as `ordered-cohort-no-retry-v1`; the task
result's execution identifier remains unchanged. The declared limit is not an
executed group count. Require `run_complete` and a zero exit status for whole-run
success; a drained stream can still contain rejected documents.

## Library surfaces

- `NlpEngine::batch_int4_extract_cohorts` runs the ordered stream with bounded
  native windows. Its return separates `BatchSummary` from invocation-wide
  physical cohort statistics.
- `NlpEngine::batch_int4_source_cohorts` runs the same ordered transport for a
  fixed source-task family using `Int4SourceCorpusConfig`. Its
  `Int4SourceCohortBatchResult` likewise separates `.summary` and `.cohorts`.
  The source adapter reuses the extraction window memory quote and exact
  native budget types, while retaining the original source-task finalizers.
- `NlpEngine::execute_int4_extract_cohort` executes an already prepared finite
  collection of schema/source plans.
- `NlpEngine::execute_int4_source_cohort` executes prepared NER, keyphrase,
  summary, and passage-QA plans, including mixed source-task kinds.
- `NlpEngine::summarize_int4_source_cohorts` and
  `NlpEngine::reduce_int4_keyphrases_document_cohorts` execute native chunk
  windows directly into the existing complete-document summary/keyphrase
  reducers. Their `.document` retains the existing exact bullet/evidence union
  or global phrase ranking; `.cohorts` reports physical execution counts.
  They use `Int4SourceMapConfig` plus `Int4SummaryLimits` or
  `Int4KeyphraseLimits`. All candidates survive intermediate reduction and
  top-k applies only at the complete final root. The map's occurrence budget
  and the reducer's independent verification allowance remain separate and
  nonrenewable. Final output omits native token transcripts, while memory
  admission still covers every transient row and the complete evidence frontier.
- The finite-plan methods use `SourceLimits`: context and mask limits are per
  row, preparation covers **all** transferred plans and vocabulary/runtime
  state, and one invocation deadline/checkpoint budget covers the collection.
  The caller supplies the complete aggregate result cap.
- Lower-level task cohort adapters preserve the original independent semantic
  finalizers inside the native session. Each row must complete and validate;
  there is no unrestricted successful prefix or unconstrained fallback.

## Chunks of one long document

`candidate source-int4 extract-map`, `ner-map`, `keyphrases-map` (including
`--reduce-keyphrases`) and default `summarize-map` accept `--cohort-chunks N` to
apply native grouping to independently planned chunks of one complete document.
`N` must be 1 through 64 and cannot exceed `--max-chunks`. The default remains
serial.

The planner compiles every actual schema/source chunk before inference. Each
row retains its exact task identity, output limits, source grammar and causal
context. The final short window allocates its actual row count; it does not pad
or duplicate source chunks. Every value and source occurrence then passes the
existing independent validation and original-coordinate lifting. Raw extraction,
NER and keyphrase maps retain the ordered independent JSON or typed source-task
values. Default summaries retain the complete exact bullet/evidence union;
ranked keyphrases retain the complete exact phrase/evidence union. Both aggregate
modes perform deterministic document-wide ranking only after all chunks finish,
with top-k applied once at the final root. A later chunk or reduction failure
publishes no document result.

The pre-IO quote covers the entire declared simultaneous window. The host
checks all actual windows and retains componentwise peak native and decoder
workspace reservations across their physical completion. These are additional
to **all** prepared document plans, the full reduction frontier, verification
storage and the guarded final result. The 64 MiB transient cohort envelope cap
includes all per-chunk result ceilings plus metadata: a width of 64 with a
1 MiB per-chunk ceiling requires lowering `--max-result-bytes` or the width.
Whole-document work, mask and verification limits are never renewed by a new
window. The output guard survives final serialization and flush.

`NlpEngine::extract_int4_document_cohorts` uses
`Int4ExtractionMapConfig.limits.mapping.reduction.map_batch_chunks` as its
window width. Its result separates the original `.document` from `.cohorts`
statistics. The lower-level prepared document exposes
`cohort_requirements(context_limit)` and `execute_cohorts_with_control` for
embedding hosts that supply the complete ownership and memory reservations.
`NlpEngine::map_int4_source_cohorts` provides the equivalent source-task path
with `Int4SourceMapConfig.mapping.reduction.map_batch_chunks`. Its lower-level
prepared source map exposes the same two method names and source-specific
memory/budget/result types. Supported raw library tasks are NER, keyphrases
and independently cited summary chunks; the original task finalizers still
run inside the native session before original-coordinate map verification.
The CLI keeps the document under `output` and reports the declared width and
actual physical counts under `cohort_schedule`. Raw maps expose physical batch
counts in `output.mapped.map_batches`; aggregate summaries and ranked keyphrases
use `output.map_batches`. Per-chunk identities and source semantics are unchanged.
Width 1 explicitly selects the cohort path for comparison.

For example, retain twelve globally ranked phrases from complete native maps:

```sh
fnlp candidate source-int4 keyphrases-map report.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 \
  --reduce-keyphrases --document-keyphrases 12 --cohort-chunks 4
```

Summary and keyphrase aggregation retain their independent verification,
candidate/evidence and final-result limits. A small final top-k cannot shrink
the full transient native output or evidence frontier. Aggregate results omit
native token transcripts; those transient tokens remain funded until native
execution and reduction complete. The hosted aggregate methods described above
feed these reducers directly without retaining a separate raw-map document.

Question-aware maps, neural summary/answer synthesis and durable document jobs
retain their existing execution. Both direct argument parsing and owned job
admission refuse unsupported cohort requests. `summarize-map --cohort-chunks`
conflicts with `--synthesize-summary` and `--hierarchical-summary`; those modes
have separate reduction and synthesis budget contracts.

All source-derived masks and occurrence state stay document-owned. Structural
validity and exact source membership do not certify semantic accuracy or factual
support. Durable jobs retain their existing execution paths.

Focused regression source covers exact memory/KV admission, per-document budget
stability, unchanged whole-run work, complete result bounds, source separation,
and typed error isolation. Ignored full-model fixtures compare complete serial
and cohort NDJSON streams at widths 1, 2 and 4 for extraction and all four source
tasks, including malformed rows, task-switch rejection, flush/ID reuse, separate
QA passages and EOF without a trailing newline. Rust builds, tests, and
model/performance qualification remain pending
the repository's controller-selected DSR checkpoint.

Chunk-aggregate regression source additionally covers exact union/ranking
across widths and ragged tails, whole-batch identity/cardinality rejection,
late invalid evidence and cancellation, cumulative verification exhaustion,
complete candidate storage and guarded output/flush failures. An ignored
model-present fixture compares raw maps, aggregate summaries and global
keyphrase rankings at serial execution and cohort widths 1, 2 and 4, excluding
only physical scheduling fields. These fixtures remain unrun.
