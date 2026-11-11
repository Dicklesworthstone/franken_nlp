# Grammar-first selected-row INT4 source tasks

The native grouped-INT4 decoder can explicitly project only the output rows
that the current JSON grammar permits. This uses the existing quantized model,
native session, selected-row kernels and source validator. Full-vocabulary
execution remains the default and the reference for qualification.

## Candidate CLI

One-off schema extraction exposes the strategy explicitly:

```sh
fnlp candidate source-int4 extract document.txt \
  --schema schema.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-new-tokens 128 --max-selected-rows 256 \
  --prefill-rows 4
```

The source-directory/manifest acquisition route also works. The optional
`--prefill-rows` changes prompt scheduling and composes with selected rows;
omitting it preserves sequential prefill. Add `--source-membership` for a
schema with annotated verbatim fields. The row cap must cover the complete
legal token set at every visited grammar state; this example cap is not a
promise that an arbitrary schema fits.

`--max-selected-rows` is available on one-off `extract`, `ner`, `keyphrases`,
`summarize` and `answer`, and on `extract-batch`, `ner-batch`,
`keyphrases-batch`, `summarize-batch` and `answer-batch`, including explicit
document cohorts and packed prompt passes. Complete-document extraction also
supports it on `extract-map`, `extract-corpus` and
`extract-document-job start|resume`, with serial or cohort chunk execution.
Other source-task maps/corpora, single-context durable extraction, redaction
and the semantic second-reader route retain their full-vocabulary contracts.
Without the option, these tasks retain full-vocabulary execution and their
existing identity. Native work in the result reports actual projected rows;
grammar-mask, task, source and complete-result limits are unchanged.

For example, the same source-bound NER task can opt into the strategy:

```sh
fnlp candidate source-int4 ner document.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-selected-rows 4096
```

The cap is a refusal boundary, not a prediction of how many rows a source
task will need. Summary prose and answer strings can admit wide token sets;
a small cap can refuse them. No task silently changes its schema or falls
back to the full head. NER and keyphrases retain exact source occurrences;
summaries and answers retain independently checked citations, passage
boundaries and their existing abstention semantics. A decoding failure does
not become an answer abstention.

### Resident extraction streams

The same fixed policy can process a bounded NDJSON corpus without reloading
the model for each record:

```sh
fnlp candidate source-int4 extract-batch documents.ndjson \
  --schema schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-new-tokens 128 --max-selected-rows 4096
```

Each record is `{"id":"document-1","text":"exact original source"}`.
The shared schema, grounding policy and row cap apply to the entire stream;
non-null `task_args` cannot replace them. Each document gets its own grammar,
source mask and causal state, while the model, vocabulary and engine remain
resident. All native work, masks, input/output limits and cooperative
checkpoints remain cumulative across records and `{"flush":true}` controls.
Whole-run success requires both `run_complete` and exit status zero.

Add `--cohort-documents N` for shared-layer execution of at most N independent
documents, and optionally `--prefill-rows P` for at most P total prompt and
already sampled feedback positions per native pass. Both caps are 1 through
64; P can be smaller than N. Without `--cohort-documents`, the stream remains
serial and `--prefill-rows` is unavailable. Selected-row batches explicitly
refuse a prepared item with a different
head policy. Host admission uses the fixed row cap to recover the exact
reserved output-token ceiling; dividing selected projections by the full
vocabulary size would underreserve token storage. Complete native results
still undergo independent schema, source, EOS, control and exact-work checks.

Library callers select the policy on the immutable
`Int4ExtractionBatchPlanner::with_selected_rows` factory before passing it to
`NlpEngine::batch_int4_extract`, `batch_int4_extract_cohorts` or
`batch_int4_extract_cohorts_packed`. `with_model` carries the declared policy
to another explicitly loaded model. Complete-document extraction uses the
same fixed policy as described below.

### Complete-document extraction

Use the same option when each original document spans several native contexts:

```sh
fnlp candidate source-int4 extract-corpus documents.ndjson \
  --schema schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-new-tokens 128 --max-selected-rows 4096 \
  --max-chunks 64 --cohort-chunks 4
```

For one document, use `extract-map document.txt` with the same schema, model
and projection options. Each chunk seals the selected strategy and exact cap
before its work is summed. A work ceiling can therefore admit selected-row
execution even when full-vocabulary projection would exceed it. Every chunk
must finish; a cap refusal never publishes a partial successful document.
The strategy composes with `--root-array-union`, whose exact schema-checked
record reduction and original-source evidence remain unchanged.

The row cap applies independently to each chunk at each output selection,
including accepting EOS. It never combines legal sets across chunks or drops
candidates to fit. The complete document result includes `max_selected_rows`
when selected projection is enabled. Its execution marker is
`grouped-int4-selected-row-exact-schema-document-map-v1`; the native chunk
cohort wrapper uses `grouped-int4-selected-row-exact-schema-document-cohorts-v1`.
Corpus provenance repeats the public cap; raw schema and source commitments
stay private. All five native work
axes, mask visits and original-coordinate verification allowances remain
cumulative across attempted documents and flush boundaries. Native KV and
decoder storage retain their complete existing reservations.

For retained work, `extract-document-job start` and `resume` accept the same
option. The authenticated recipe binds the selected task/native execution
versions and exact cap. Repeat the original cap on resume: adding, removing
or changing it refuses the stored job instead of mixing policies. Failed
attempts retain their durable work debit. Existing full-head jobs keep their
original recipe when the option is omitted.

For example, after preparing the protected job directory, key, ID and lifetime
limits, start a retained selected-row job with:

```sh
fnlp candidate source-int4 extract-document-job start documents.ndjson \
  --schema records.schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-chunks 8 --cohort-chunks 4 --max-selected-rows 4096 \
  --store-results --job-dir /private/extraction-job --job-id "$JOB_ID" \
  --key-file /private/extraction-job.key --limits extraction-job-limits.json \
  --materialize
```

A selected cap of 166144 still creates a distinct execution policy, even when
its conservative work equals the full head. It cannot reuse a full-head
preparation or resume a full-head job. Selection and chunk-cohort scheduling
compose in either builder order; changing either invalidates prior preparations.

Library callers select `Int4ExtractionBatchPlanner::with_selected_rows`
before `plan_document_with_control` or hosted one-document extraction.
`Int4ExtractionDocumentBatchPlanner::with_selected_rows` applies it to an
ordered complete-document stream, and
`Int4DocumentExtractionJobPlanner::with_selected_rows` binds it into a durable
document job. Each consuming document-factory setter invalidates previously
prepared documents; reconstruction checks every original chunk identity.
These routes do not expose packed prompt prefill through this option.

### Persistent CPU teams for a complete extraction document

The one-document `extract-map` command additionally accepts
`--compute-threads N`, with `N` from 2 through 64. One coordinator and `N - 1`
scoped workers serve every chunk and constrained output projection in the
document. The team uses the existing exact output-row partitioning; it does
not change a row's group reduction order. The ordinary single-threaded path
remains the default.

```sh
fnlp candidate source-int4 extract-map document.txt \
  --schema records.schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-selected-rows 4096 --compute-threads 4 --root-array-union
```

All chunks, identities and complete-document work preflight before the native
engine allocation and first forward. The team is formed once for the physical
invocation, and its additional worker workspace is included in admission before
private input IO. Runtime scheduler threads are additional to the selected
compute width. Original source coordinates, exact chunk JSON, the optional
root-array union, and every work/mask/verification allowance retain the existing
serial contracts. A completion becomes available only after the workers join;
the output reservation remains held through CLI serialization, write and flush.

The optional top-level `compute` receipt records the declared team width and
completed projection/row counts. It is separate from the semantic document and
its identities. With `--compute-threads 1` or no flag, that field is omitted.
The library entry point is `NlpEngine::extract_int4_document_with_threads`, which
returns `HostedInt4TeamResult<Int4ExtractionMapRun>` and shares the serial
planner, native execution and independent finalizer.

The document's explicit `--input-encoding safe-bpe-v1` policy can also be
combined with this team. Its complete-preparation tokenizer ceilings and
encoding identity remain in force, and admission adds one reusable encoder
workspace alongside one team workspace. This encoding choice changes prompt
tokens; the team itself only changes how output rows are scheduled.

This selection is currently limited to the one-document independent-chunk and
root-array routes. `--cohort-chunks`, packed prefill, `extract-object-map`,
document corpora and durable document jobs have separate scheduling contracts
and do not accept this team selection. Invalid widths, unsupported mixtures
and an unfunded complete memory quote refuse before schema/source/model IO.
Real-model serial/team equality and target-host performance remain separate
controller-run gates; the option does not claim a measured speedup.

### Resident source-task streams

The four built-in source tasks use the same explicit fixed projection policy:

```sh
fnlp candidate source-int4 ner-batch documents.ndjson \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-selected-rows 4096
```

For NER, keyphrases and summaries, `text` contains the original source. The
existing typed `task_args` or `--defaults` may change permitted task options
inside the fixed ceilings, but cannot override the selected-row cap or task.
For `answer-batch`, `text` is the question and typed arguments supply the
passages, options and budget. Its input shape differs from complete-document
`answer-corpus`, which supplies a source in `text` and a question in `task_args`.

The compiler seals the projection policy before each item's admission.
Injecting a dense plan into a selected stream, or a plan with another cap,
fails before debit or execution. Exact work validation covers all five native
axes and verifies the terminal EOS and excluded control IDs. Every original
task finalizer still checks its source spans, citations and schema.

Use `Int4SourceBatchPlanner::with_selected_rows` in the library or set
`Int4SourceCorpusConfig.selected_rows` for `NlpEngine::batch_int4_source`,
`batch_int4_source_cohorts` or `batch_int4_source_cohorts_packed`.
`None` retains dense execution. The host prices output-token storage from
the fixed cap even when actual legal sets are smaller. The same optional
`--cohort-documents N --prefill-rows P` combination works for all four tasks.
The cap can refuse wide prose states in summaries and
answers, and never causes silent pruning, retry or full-head fallback.

### Selected document cohorts

For example, combine all three independent bounds:

```sh
fnlp candidate source-int4 extract-batch documents.ndjson \
  --schema schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-new-tokens 128 --max-selected-rows 4096 \
  --cohort-documents 4 --prefill-rows 2
```

The row ceiling covers the complete legal set of one document at one output
selection. It is independent of the document window and packed token count.
Each document retains its own prompt, grammar, source evidence, causal KV,
task budget and selected execution identity. The scheduler never combines
legal token sets across documents. Every finalizer completes before a window
can publish success; a late refusal or cancellation yields no partial window.
Earlier completely delivered windows remain valid.

Finite prepared-plan extraction and source cohort APIs also accept selected
plans. All rows in a window must use the same projection strategy, but
independently sealed selected caps may differ. A mixed dense/selected window
refuses; no plan is converted to satisfy its neighbours.

`Int4ExtractionCohortMemoryLimits::requirements_for_execution` (also exposed
through `Int4SourceCohortMemoryLimits`) takes the fixed projection policy and
optional packed limits. Both CLI and host use this quote. Native KV, full
preparation owners, lookahead, complete output envelopes and token storage
remain funded. Generated-token ceilings are derived from planned projections
divided by the sealed cap, including terminal EOS. Selected scoring reuses one
legal mask and blocks of at most 32 rows inside the existing native head
workspace. Packed scratch adds exactly `prefill.validate()` once per live
window. These are checked ownership and payload models, not measured RSS.

## Library use

Seal the strategy before asking the host to admit the prepared extraction:

```rust
use franken_nlp::native_engine::constrained_int4::sparse::Int4JsonSparseLimits;

let prepared = prepared.with_selected_rows(Int4JsonSparseLimits {
    max_rows_per_step: 256,
})?;
```

The existing `NlpEngine::execute_int4_extract` and
`NlpEngine::execute_int4_extract_layer_major` entrypoints price the sealed
plan's work and dispatch its selected strategy. The layer-major variant also
requires its existing separate prompt-scratch reservation. Direct native
callers can use `constrained_int4::sparse::decode_json_int4_sparse` or
`sparse::prefill::decode_json_int4_sparse_layer_major` with the matching planned
work and native/KV budgets.

`PreparedInt4SourceTask::with_selected_rows` provides the same consuming
opt-in for built-in source tasks before hosted admission. The host's existing
serial and layer-major source entrypoints use the selected plan's work.
Its original typed finalizer stays sealed to the plan; callers cannot inject
a schema, substitute an extraction result, or relabel a full-head receipt.

`max_rows_per_step` must be in `1..=166144`. It is a complete legal-set ceiling:
if one selection has more legal rows, execution fails before that selection's
first projection. It never truncates the legal language or substitutes top-k
pruning. A wider schema can therefore require a larger ceiling. The grammar
mask still covers the complete vocabulary and retains its full traversal
charge.

## Execution and accounting

At every output step, the decoder materializes the complete grammar mask,
applies the sealed control exclusions and counts all legal rows. It then
projects every legal row in ascending-ID blocks of at most 32. Each logit must
be finite. Equal scores retain the lowest token ID, including signed-zero
ties. EOS is legal only at an accepting state and is explicitly projected.
Even a single legal non-EOS token is fed through the normal model before the
next selection.

The work ceiling uses `max_new_tokens * max_rows_per_step` output projections
and the unchanged worst-case forward-position and attention counts. The
completed receipt records actual projected rows and exact group-dependent
native work. Physical resident KV, grammar/source workspace and the existing
engine allocation remain separately admitted; this feature does not reduce
their memory reservation.

Both the execution identity and comparison policy bind the selected strategy
and row ceiling. Native results use
`grouped-int4-grammar-first-selected-rows-v1`; extraction results use
`grouped-int4-selected-row-schema-source-extraction-v1`; built-in source
results use `grouped-int4-selected-row-source-portfolio-v1`. A full-head result or
admission cannot be relabeled as selected-row execution. Cohorts retain these
per-row identities and result markers, with separate outer markers:

| Layer | Selected cohort | Selected packed cohort |
|---|---|---|
| Native | `grouped-int4-grammar-first-selected-rows-cohort-v1` | `grouped-int4-grammar-first-selected-rows-packed-cohort-v1` |
| Extraction | `grouped-int4-schema-source-extraction-selected-cohort-v1` | `grouped-int4-schema-source-extraction-selected-packed-cohort-v1` |
| Source tasks | `grouped-int4-source-portfolio-selected-cohort-v1` | `grouped-int4-source-portfolio-selected-packed-cohort-v1` |

`group_steps` counts forward passes. Selected `projection_groups` counts
physical head invocations for the bounded row chunks and can exceed the
forward-pass count. Both are checked against the completed per-document work;
neither is a speed measurement. Single-context durable jobs continue to refuse
injected selected plans before durable work journaling. Complete-document
jobs require their immutable factory and authenticated recipe to select the
strategy before any durable attempt is admitted.

Independent JSON validation, exact-number preservation, source-membership
evidence, complete result-size checks and the final cancellation check remain
inside the owned native session. Errors abort that session and drain its KV
state. An accepting grammar prefix alone is not a completed result.

## Evidence status

The retained model-free regression definitions cover complete row coverage,
bounded block tails, full-argmax comparison on small masks, ties, nonfinite
selected logits, cancellation, work mismatches, group tails, identity
separation, source spans, exact numeric boundaries and layer-major composition.
They have not been executed in this session.

Additional serial-stream definitions cover fixed policy propagation, injection
refusal, exact output-token reservation, failure poisoning and KV cleanup,
selected completion verification, parser scope, and early map/durable refusal.
New cohort definitions cover independent caps and identities, original source
finalizers, selected chunk counts, rejection of mixed strategies and wrong
markers, complete output-token reservation, packed scratch charged once, and
all five CLI combinations at exact process-memory boundaries. These
definitions are also **UNRUN**.

Complete-document definitions also cover authenticated resume, retained-input
replay, fixed memory and lifetime limits, and full-vocabulary-cap identity
separation despite equal planned work. These definitions remain **UNRUN**.

Compilation and regression execution await the repository's controller-owned
DSR checkpoint on the named production graph. Actual-model equality against
the full-vocabulary reference and target-host performance remain separate
pending gates. Unrequested logits are not computed or inspected, so this path
does not reproduce full-head rejection of a nonfinite *unrequested* row. No
model-parity, speedup or release-qualification claim follows from the reduced
projection ceiling alone.
