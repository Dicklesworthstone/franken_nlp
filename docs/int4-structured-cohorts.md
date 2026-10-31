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
- `NlpEngine::answer_int4_document_cohorts` executes one immutable question
  over the complete source partition with `Int4SourceMapConfig<SourceQuestion>`.
  Its `.document` retains every independent answer, abstention and whitespace
  range in original order. `mapping.reduction.map_batch_chunks` counts source
  chunks per window; `.cohorts.chunks` counts only nonblank native passages and
  `.cohorts.cohorts` excludes all-whitespace windows. All original-coordinate
  citation checks and nonrenewable question verification budgets still apply.
  It does not select or synthesize a global answer.
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

### Neural synthesis after discovery cohorts

The prepared summary-synthesis, summary-hierarchy, answer-synthesis and
answer-hierarchy plans expose `cohort_requirements(context_limit)` and
`execute_cohorts_with_control`. Their native discovery chunks run in bounded
shared-layer windows, then enter the same evidence collection, final synthesis
and hierarchy reducers as serial discovery. In this discovery-only schedule,
hierarchy passes remain scalar;
every level still admits its complete frontier before its first forward.

The hosted entrypoints share one physical resource owner and use the same
task-specific configuration and hierarchy limits as their serial equivalents:

- `NlpEngine::synthesize_int4_source_summary_cohorts`
- `NlpEngine::synthesize_int4_source_summary_hierarchy_cohorts`
- `NlpEngine::synthesize_int4_document_answer_cohorts`
- `NlpEngine::synthesize_int4_document_answer_hierarchy_cohorts`

`Int4SourceMapConfig::synthesis_cohort_requirements(group_width)` quotes the
complete declared native phase peak. The host additionally reserves the full
source and private question capture, preparation, evidence frontier, verifier
storage and retained output. Summary output retains discovery transcripts plus
one final pass or every permitted hierarchy pass. Flat answer output retains
text and citations; hierarchical answer output also retains every additional
pass's native transcript. Each mode charges those actual output obligations.

Each result keeps the complete existing synthesis result under `.document` and
reports only discovery's physical grouping under `.discovery_cohorts`. The
generic envelope is `Int4SynthesisCohortRun<T>`; the four task-specific aliases
are `Int4SummarySynthesisCohortRun`, `Int4SummaryHierarchyCohortRun`,
`Int4QuestionSynthesisCohortRun` and `Int4QuestionHierarchyCohortRun` in their
respective `cohort` modules. Discovery statistics never include synthesis work
or charge question-map whitespace as a native row.

The complete native reservation is the componentwise maximum of all discovery
windows and one full-context scalar engine. Discovery engines physically drain
before synthesis allocates that engine. The scalar capacity must fit the
original task KV ceiling before discovery begins, even when a short map would
fit much less memory. Empty evidence skips scalar allocation and the final
model pass. The embedding host must also retain complete preparation, evidence,
verification, native transcript and final-output reservations through completion.

Final-pass and hierarchy work reserves are subtracted before discovery; early
completion or a smaller frontier never replenishes them. Shared verification
allowances, original-source occurrence checks, explicit no-evidence status and
whole-document failure behavior remain unchanged. This schedule does not widen
the evidence caps or make lossy hierarchy equivalent to full-context inference.

## Native hierarchy group cohorts

The separate `--cohort-hierarchy-groups N` option executes independent evidence
groups within one summary or answer hierarchy level through the native
shared-layer INT4 decoder. It requires the existing synthesis and hierarchy
flags. N must be in `1..=min(64, max_passes)`, where `max_passes` is the selected
`--summary-max-passes` or `--answer-max-passes`. It does not change those
semantic limits or infer a hierarchy from a flat synthesis request.

The hierarchy and discovery schedules are independent. Supplying only
`--cohort-hierarchy-groups` keeps discovery serial. Supplying both it and
`--cohort-chunks` uses the requested independent widths for the two phases.
Omitting both preserves the original serial execution. Only the two hierarchy
map routes and their `document-job start|resume` counterparts accept the new
option; single tasks, flat synthesis, other maps, batches, extraction jobs and
item-local jobs do not acquire a hierarchy schedule.

For hierarchy grouping with serial discovery:

```sh
fnlp candidate source-int4 answer-map report.txt \
  --question question.txt --synthesize-answer --hierarchical-answer \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 \
  --cohort-hierarchy-groups 4 \
  --answer-max-levels 8 --answer-max-passes 16 \
  --max-total-mask-node-visits 160000000000
```

Every level's complete ordered frontier is planned and authenticated before its
first forward. Native windows then run at most N independent compiled plans,
including any final short window. Each row retains its exact task identity,
question where applicable, source-constrained grammar, causal KV and semantic
finalizer. Validated results fold in original group order after the native
window drains; the next level is planned only after the current level
completes. A late invalid group, cancellation, exhausted work or nonprogress
discards the complete document result. Empty evidence produces zero hierarchy
statistics without allocating a hierarchy engine.

All permitted hierarchy passes and their five work axes, masks, evidence
verification and grouping-tokenizer allowance remain part of the original
complete-document budget. Grouping neither renews work nor selects fewer
groups. The hierarchy remains explicitly lossy: source quotations omitted by
intermediate summaries or answers can be lost.

Before discovery, admission prices the componentwise maximum of the selected
discovery phase and N full-context hierarchy rows, including decoder and row
result workspace. Each row must fit the original per-task KV ceiling. The
host also reserves `(N - 1) * preparation_reserve_bytes` for additional
simultaneous compiled plans, retaining the original full preparation owner,
source/question capture, complete evidence frontier, verification, native
transcripts and final output owners. Phase capacities are reused after physical
drain; they are not added as if both native phases ran simultaneously. These
are conservative ledger reservations, not measured RSS bounds.

The shared schedule is `Int4HierarchyCohortSchedule { max_groups,
discovery_cohorts }` in `long::cohort::hierarchy`. Prepared summary and question
hierarchies expose `hierarchy_cohort_requirements(context_limit, schedule)` and
`execute_hierarchy_cohorts_with_control`. Hosted callers use:

- `NlpEngine::synthesize_int4_source_summary_hierarchy_grouped`
- `NlpEngine::synthesize_int4_document_answer_hierarchy_grouped`

Each takes the existing configuration and semantic hierarchy limits, followed
by the schedule, kernel and cancellation token.
`Int4SourceMapConfig::hierarchy_cohort_requirements(group_width, schedule)`
quotes the native/decoder phase peak; `schedule.additional_preparation_bytes`
quotes the additional full preparation owners. Lower-level callers must retain
both charges, plus the original task workspace/output owners, through drain.

`Int4SummaryHierarchyGroupedRun` and `Int4QuestionHierarchyGroupedRun` keep
the original semantic run under `.document`, optional discovery statistics
under `.discovery_cohorts`, and hierarchy-only statistics under
`.hierarchy_cohorts`. The latter reports `cohorts`, `passes`, `group_steps`,
`projection_groups`, `planned_work` and `model_work`. Standalone CLI output
keeps its existing hierarchy protocol and `output` document, adding
`hierarchy_cohort_schedule` with execution
`"within_level_constrained_hierarchy_cohorts"`, `declared_max_groups` and
`physical`. Requested discovery grouping keeps its separate
`cohort_schedule`; private question/source identities are never added.

## Chunks of one long document

`candidate source-int4 extract-map`, `ner-map`, `keyphrases-map` (including
`--reduce-keyphrases`), `summarize-map` and `answer-map` accept `--cohort-chunks N` to
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

Default `answer-map` uses one exact private question for every nonblank chunk
and retains all independent answers and abstentions. Whitespace ranges remain
in source order with no native answer, KV allocation or work charge. Only the
nonblank plans within each logical source window form a native cohort; an
all-whitespace window performs no native call. The planner does not pack native
rows across an intervening source-window boundary. A wholly blank document
continues to refuse before inference.

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
counts in `output.mapped.map_batches`; independent QA uses the same field for
logical source windows. For QA, `cohort_schedule.physical.chunks` counts native
passages and `cohort_schedule.physical.cohorts` excludes all-whitespace windows,
so physical counts can be smaller. Aggregate summaries and ranked keyphrases
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

One-question document execution uses the same snapshot route:

```sh
fnlp candidate source-int4 answer-map report.txt \
  --question question.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 --cohort-chunks 4
```

Question cohorts retain the complete logical document frontier and the private
question's preparation/capture allowance even when most chunks are whitespace.
The `--max-qa-citations`, `--max-qa-evidence-spans` and `--max-qa-scan-steps`
allowances never renew across windows. Every citation is still checked against
its original passage; the question and passage manifest supply no citation
authority. The final QA output contains answers and citations without native
token transcripts. Those transient native results remain funded until execution
and reduction complete. Question text and paths stay out of public provenance.

### Synthesis command combinations

All four neural synthesis modes accept discovery cohorts. With only
`--cohort-chunks N`, later phases retain their existing schedules:

| Command | Synthesis options with `--cohort-chunks N` | Later phases |
|---|---|---|
| `summarize-map` | `--synthesize-summary` | One scalar pass over all verified quotations |
| `summarize-map` | `--synthesize-summary --hierarchical-summary` | Bounded scalar quotation-selection levels |
| `answer-map` | `--synthesize-answer` | One scalar answer over all verified passages |
| `answer-map` | `--synthesize-answer --hierarchical-answer` | Bounded scalar question-aware levels |

For example, combine discovery cohorts with native groups inside an explicit
summary hierarchy:

```sh
fnlp candidate source-int4 summarize-map report.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 --cohort-chunks 4 \
  --synthesize-summary --hierarchical-summary \
  --cohort-hierarchy-groups 2 \
  --summary-max-levels 8 --summary-max-passes 16 \
  --max-total-mask-node-visits 160000000000
```

`--memory-mib` must cover the complete declaration, including all discovery
plans and every permitted dependent pass. Work, mask, evidence and verification
ceilings retain their original scope; requesting cohorts never raises them.
Source conversion and independently pinned snapshots use the same route.
Omitting `--cohort-chunks` keeps discovery serial; omit both scheduling flags
for completely serial execution.

Synthesis output keeps its existing protocol and complete semantic result under
`output`, with discovery under `output.discovery`. The separate schedule reports
`execution: "cross_chunk_constrained_discovery_cohorts"`,
`declared_max_chunks`, and `physical`. Those counters include only discovery
native rows/windows and discovery work. They exclude all hierarchy passes
and question-map whitespace; all dependent work remains in the original
semantic receipt. Private question/source content is absent from provenance.

### Durable complete-document cohorts

Native chunk and hierarchy cohorts also run inside authenticated
complete-document jobs:

| Command | Supported cohort phase |
|---|---|
| `document-job start\|resume ner` | Independent entity maps |
| `document-job start\|resume keyphrases` | Independent maps or complete global ranking with `--reduce-keyphrases` |
| `document-job start\|resume summarize` | Raw maps with `--map-only`, deterministic selection, discovery for flat/hierarchical synthesis, or groups within explicit hierarchy levels |
| `document-job start\|resume answer` | Independent QA, discovery for flat/hierarchical synthesis, or groups within explicit hierarchy levels |
| `extract-document-job start\|resume` | Independent exact-schema extraction maps |

Each command above lives under `fnlp candidate source-int4`. Add
`--cohort-chunks N` to its existing complete job invocation, retaining the
protected job directory, secret, limits and explicit result-storage options
described in the README. N must be 1 through 64 and no greater than
`--max-chunks`. The selected schedule and width are authenticated private recipe
inputs: use the same option on resume. Omit it when resuming an existing serial
job. Explicit summary/answer hierarchies also accept
`--cohort-hierarchy-groups N`, bounded by `min(64, max_passes)`, independently
of discovery grouping. Both schedules and widths are authenticated; changing
either requires a separately started job. Existing serial and discovery-only
jobs keep their original flags and recipe bytes.

For example, after preparing the protected job directory, key, ID and limits
file described in the README:

```sh
fnlp candidate source-int4 document-job start summarize documents.ndjson \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 --max-chunks 8 \
  --synthesize-summary --hierarchical-summary --summary-max-passes 16 \
  --cohort-chunks 4 --cohort-hierarchy-groups 2 \
  --store-results --job-dir /private/document-job --job-id "$JOB_ID" \
  --key-file /private/document-job.key --limits document-job-limits.json \
  --materialize
```

Use `resume` with the same complete population and every original setting.
Here eight discovery chunks plus sixteen permitted hierarchy passes reserve
at most 48 billion mask visits at the default two billion per pass, within
the unchanged 128-billion document ceiling and README job limit. The standalone
examples use 64 discovery chunks and therefore explicitly raise that ceiling.

Every original document remains one durable attempt. Its complete model/mask
debit is persisted before inference, every required chunk and dependent pass
must complete, and the original semantic result is committed once. Cohort
windows and hierarchy levels do not become checkpoints, and failure never
publishes a partial document. All original task, evidence, verification and lifetime work bounds
remain in force. The complete original population is still required on resume.

Admission retains the full declared native/decoder peak for the invocation,
including short documents and ragged windows. Discovery-only synthesis modes
reserve the componentwise discovery/scalar phase maximum. Hierarchy grouping
instead includes its complete full-context group capacity in the phase maximum
and adds the extra full preparation owners described above. Population input,
private question or raw schema ownership, complete evidence frontiers, retained transcript arrays,
output serialization and journal storage keep their separate charges. No spare
scalar engine is allocated beside the cohort engines. Native owners survive
physical drain, spool acknowledgement and optional materialization.

The stored result remains the existing semantic document, without the standalone
map command's physical-statistics wrapper. Consequently, its original result
cap and spool/materialization format remain valid. Public stdout declares chunk
grouping through `chunk_schedule`, containing `execution: "native_chunk_cohorts"`,
`phase: "map"` or `"discovery"`, and `declared_max_chunks`. Requested hierarchy
grouping separately adds `hierarchy_schedule` with execution
`"native_hierarchy_group_cohorts"`, phase `"hierarchy"`, and
`declared_max_groups`. Neither declaration claims observed cohort counts or
exposes source, schema, question or result content.
Serial recipes and stdout omit the new optional fields.

Library callers opt in with
`Int4SourceDocumentJobPlanner::with_chunk_cohorts()` or
`Int4DocumentExtractionJobPlanner::with_chunk_cohorts()`. Configure the immutable
map width before constructing the planner. Both expose `uses_chunk_cohorts()`
and `cohort_requirements()`, and their existing hosted job entrypoints select
the requested schedule. Low-level integrations use the corresponding
`Int4SourceDocumentJobProcessor::new_cohorts` or
`Int4DocumentExtractionJobProcessor::new_cohorts` constructor while owning the
quoted capacity through complete processor teardown and durable acknowledgement.
For source hierarchies, call `with_hierarchy_cohorts(max_groups)` on the
immutable document planner, optionally after `with_chunk_cohorts()`.
`hierarchy_cohort_schedule()` identifies the combined declared policy,
`uses_native_cohorts()` recognizes either phase, and `cohort_requirements()`
returns their combined static peak. The existing
`NlpEngine::job_int4_source_document` dispatches the frozen policy and still
stores only complete semantic documents.

Item-local source/extraction jobs retain their existing single-item execution
and accept neither document scheduling option. New regression source covers every
document mode, authenticated schedule/width changes, prepared-owner isolation,
static and actual memory ceilings, unchanged task/work authority, private
metadata, both loading routes and early refusal. These fixtures remain unrun.

All source-derived masks and occurrence state stay document-owned. Structural
validity and exact source membership do not certify semantic accuracy or factual
support.

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

Question-specific regression source covers sparse native windows, all-whitespace
windows, ragged tails, all answers/abstentions, private-question input accounting,
invalid passage authority and complete Unicode occurrence lists. Its separate
ignored model fixture compares serial execution and widths 1, 2 and 4, checking
physical row/window counts against the original source partition while comparing
the complete QA document apart from physical schedule fields. That fixture
requires an explicit private question file in `FNLP_INT4_TEST_QUESTION` alongside
the pinned snapshot, model digest, kernel and memory settings, and remains unrun.

Synthesis regression sources cover full-context phase peaks before discovery,
all reserved work and mask allowances, shared quote-only continuations,
original-coordinate citation lifting, no-evidence scalar-allocation avoidance,
hierarchy transcript retention and discovery-only ledgers. Hosted and CLI
fixtures check complete private-input/workspace/output admission for all four
modes and unchanged semantic publication. These sources have been reviewed;
compilation, test execution, model parity and performance remain unrun pending
the controller-selected checkpoint.

Hierarchy-group regression sources additionally cover complete-level witness
checks, ordered windows and ragged tails, phase-specific statistics, original
work/verification ceilings, full-context peaks plus extra preparation owners,
QA transcript retention, unsupported CLI routes and authenticated durable
schedule changes. These new source fixtures and execution paths remain unrun;
neither this implementation nor its counters establish model parity or speedup.
