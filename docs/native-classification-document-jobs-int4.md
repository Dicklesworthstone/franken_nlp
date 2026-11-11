# Durable native INT4 classification of complete documents

`fnlp candidate source-int4 classify-document-job start|resume` applies the
existing complete-document classifier to an authenticated population. One
original document is one durable item: every chunk must finish and its aggregate
must validate before the result commits. A later invocation skips committed
documents and can explicitly retry unfinished documents under the same lifetime
work allowance.

This connects the complete-source classification in plan §§7.4 and 7.10 to
the owned-job boundary in §8.4. It uses the existing native chunk scorer,
[classification aggregate](native-classification-map-int4.md), and
[protected job storage](owned-jobs.md). Exclusive and independent multi-label
classification both support serial execution or explicit native chunk cohorts.
It requires `metadata-store` and `asupersync-runtime` on Linux x86-64 or AArch64.

## Fixed settings and original input

Provide one private defaults file, such as `labels.json`:

```json
{
  "labels": [
    {"id": "support", "description": "A request for product help"},
    {"id": "billing", "description": "An invoice, payment or subscription question"}
  ],
  "mode": "exclusive",
  "policy": {
    "minimum_candidate_weight_ppm": 0,
    "minimum_margin_ppm": 0
  }
}
```

Use `"multi_label"` for independent yes/no decisions. Mode and policy are
optional; omission selects exclusive classification and zero decision
thresholds. Labels are required, unique and ordered. Exclusive mode requires
at least two labels. The file cannot supply a document, task budget, prompt,
model identity, chunks or scores. Duplicate keys and unknown fields refuse.

The population is NDJSON of complete original source documents:

```jsonl
{"id":"document-1","text":"The entire first source, including its final section."}
{"id":"document-2","text":"The entire second source.","task_args":{}}
```

`task_args` may be absent, null or an empty object. Settings apply to the whole
population and cannot be overridden by a record. IDs must be unique across
the complete population; streaming flush controls are refused. The complete
record must fit the immutable stored input ceiling as well as the CLI's outer
input bound. The planner also checks the complete decoded source.

Supply the **complete original population on every invocation**, including
already committed items. The command admits original input in memory for the
invocation; it does not retain that input for future recovery.

## Start and resume

Prepare an existing protected job directory, a protected file containing
exactly 32 random raw secret bytes, and a random 128-bit job ID represented by
32 lowercase hexadecimal characters. The command does not generate the key
or create the directory. Follow the ownership and permissions requirements in
[owned job storage](owned-jobs.md).

For a small population, `classification-job-limits.json` can contain:

```json
{
  "max_items": 3,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 1048576,
  "max_snapshot_bytes": 4194304,
  "max_result_bytes": 16781312,
  "max_spool_bytes": 268435456,
  "max_materialized_bytes": 268435456,
  "max_journal_bytes": 67108864,
  "max_attempts": 6,
  "max_work": {
    "model": {
      "forward_positions": 262144,
      "projected_logits": 10000000000,
      "attention_pairs": 1000000000000,
      "projections": {
        "dot_products": 10000000000000000,
        "multiply_accumulates": 10000000000000000
      }
    },
    "mask_node_visits": 0
  }
}
```

These are explicit allowances, not measured costs or a guarantee that three
documents fit. The result cap includes the default 16 MiB semantic map plus
the 4096-byte allowance needed if chunk cohorts are selected at job creation.
The work allowance covers the entire job lifetime, including failed attempts.

```sh
fnlp candidate source-int4 classify-document-job start documents.ndjson \
  --defaults labels.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib "$MEMORY_MIB" \
  --group-width 128 --context-tokens 2048 --max-chunks 8 \
  --store-results --job-dir /private/classification-job \
  --job-id "$JOB_ID" --key-file /private/classification-job.key \
  --limits classification-job-limits.json --materialize
```

Use `resume` in place of `start`, preserving the original population, defaults,
model identity, group width, planning limits and job limits. Use `-` for stdin.
The explicit `--source-dir` and `--source-manifest` pair can replace the pinned
snapshot options. Both acquisition routes use the same model-bound factory.

An explicit `--save-snapshot NEW_FILE` follows fixed-settings and factory
admission, before population ingestion. It uses create-new storage. A completed
model snapshot can remain if later population authentication, execution or
report delivery fails.

## What resume authenticates

The keyed job contract binds the exact original input population, IDs and
order; the actual model and compiler identity; fixed labels, descriptions,
order, mode and local decision policy; task ceilings; all chunk limits;
complete-document native work and result ceilings; aggregate coverage policy;
and the selected serial/cohort schedule and requested cohort width. Numerical
output-row worker width is a separate physical selection and is excluded from
this recipe; changing `--compute-threads` on an explicit resume does not change
any semantic authority or renew attempts and work.

Job limits are immutable, including lifetime attempts, all five native work
counters, and storage capacities. A changed recipe or population refuses
before tail repair. Increasing an invocation's transport, memory or checkpoint
allowance does not increase the stored job authority. Embedding callers may
change host preparation or allocator reserves when the current complete quote
is admitted; native geometry and result envelopes still have to match.

Each original document is planned completely before its exact native work is
durably debited. The debit covers forward positions, projected logits,
attention pairs, dot products and multiply-accumulates for all chunks and all
label heads. Finite scoring performs no grammar-mask work, so
`mask_node_visits` must be zero. The five CLI work flags are per-document caps
and must fit the immutable whole-job allowance.

An interrupted document has no successful partial result. Retrying it starts
the complete document again and consumes another attempt plus new native work;
the previous debit remains. There is no chunk-level checkpoint and no automatic
retry. Already committed documents consume no additional native work on resume.

Uncommitted spool bytes or stages refuse by default. The resume-only
`--discard-uncommitted-tail` option explicitly authorizes recovery **after**
the original contract authenticates. Report-delivery failure may follow durable
progress; inspect the job or resume rather than assuming rollback.

## Compact input and resume

Add `--input-encoding safe-bpe-v1` to `start` and repeat it on `resume`, with
identical `--max-tokenizer-input-bytes`, `--max-tokenizer-output-tokens`,
`--max-tokenizer-pair-checks` and `--max-tokenizer-pair-bytes` values. These
limits cover one complete document attempt, including initial preparation,
all source/label probes and reconstruction for execution. They never reset
between those passes. Exhaustion refuses the document before native inference;
no source is truncated and no encoding fallback is selected.

The immutable private recipe binds the encoding kind and every ceiling before
population replay or repair, including retained-input replay. Changing the
policy refuses resume. Byte fallback remains the default, with its existing
recipe bytes unchanged. Serial, persistent numerical teams and chunk-cohort
execution use the fixed policy and admit exactly one transient encoder
workspace; numerical teams and chunk cohorts are mutually exclusive. The native work
ledger remains separate and retains the original lifetime semantics.
Compact mode supports pinned source input and pinned snapshot reload;
`--save-snapshot` is refused before IO.

For embedding, configure `Int4ClassificationPlanner::with_document_encoding`
before constructing the durable planner. The host derives the workspace from
the sealed policy; per-record arguments cannot replace it.

## Persistent numerical teams

Add `--compute-threads 4` to either `start` or `resume` to retain one coordinator
and three output-row workers for the complete invocation, with an additional
runtime thread. The same team serves all documents, chunks and label heads,
including every admitted durable attempt. Both exclusive and independent
multi-label modes preserve their ordinary full-vocabulary scores and complete
document aggregate. Explicit compact input retains the fixed attempt-wide
tokenizer allowance.

Width one preserves serial execution; widths 2..64 add one admitted worker
workspace and the corresponding process thread capacity before private IO.
Multiple workers cannot combine with any explicit `--cohort-chunks` selection,
including one. The original native, retained-result and durable-buffer quotes
remain in force, and all workers physically join before successful progress
returns. The original deadline continues through metadata publication.

Worker width is not part of the authenticated recipe. You may change it on an
explicit resume while preserving the original input population, labels, model,
encoding and semantic limits. Lifetime attempts and native work never renew.
Stdout adds only optional `compute_schedule` metadata with execution
`int4-output-row-team-v1` and `declared_compute_threads`; width one omits it.
Stored semantic results and their canonical bytes retain the serial contract.

Embedding callers use
`NlpEngine::job_int4_classification_document_with_threads(model, planner,
request, limits, kernel, compute_threads, reader, cancellation)` for widths
2..64, with a resource configuration that admits the selected team. The
existing `job_int4_classification_document` method retains serial execution.
Source review and regression definitions cover these boundaries; executable
validation and model comparison remain pending the required DSR checkpoint.

## Serial execution and chunk cohorts

Serial is the default. Add `--cohort-chunks N` at job creation to share native
decoder layers across a bounded window of complete chunks. Repeat the exact
option on resume. It changes physical scheduling and the stored result
envelope; it preserves the original per-chunk decisions and aggregate meaning.

The declared width must fit `N × heads_per_chunk <= 64`, even if the actual
document would have a shorter final window. Exclusive mode has one head;
multi-label mode has one head per label. For example, two independent labels
with `--cohort-chunks 3` require six simultaneous heads. Invalid geometry
refuses before model or population loading and is never silently narrowed.
Serial retains support for up to 128 labels. A cohort width of one still
executes the multi-label heads together and therefore permits at most 64.

Every chunk plan and every future native window is preflighted before native
execution. The complete map, retained chunk results, native state, original
population, journal, serialization and guarded output all have simultaneous
memory charges. The CLI conservatively quotes the largest admitted unknown
label geometry before population IO; the host quotes the actual fixed labels.
These resource claims model ownership, rather than measuring process RSS.

## Retained results and reporting

The serial stored result is the existing complete classification map. The
cohort stored result contains that map at `output` plus physical scheduling
metadata. Every source span, full-vocabulary chunk score and scalar-weighted
aggregate remains available. No native chunk failure is converted to an
abstention or omitted from a successful map.

The aggregate is `classification-source-scalar-mixture-v1`. It weights chunk
candidate distributions by original Unicode scalar counts and applies the
selected coverage and decision policies. It is uncalibrated and does not
claim equivalence to a single model pass over the full source. See the
[classification map guide](native-classification-map-int4.md) for the exact
exclusive and independent multi-label semantics.

Complete result guards remain held through canonical encoding, spool sync and
the journal's commit acknowledgement. `--materialize` publishes verified,
ordered `materialized.ndjson` only after all documents commit. Existing stored
job status, verification and materialization remain available without a model.

Stdout contains only public candidate provenance and authenticated progress
under `fnlp-candidate-int4-classify-document-job-v1`, with
`durable_unit: "complete_original_document"`. It omits source text, labels,
private policy values, scores, key bytes and private paths. The invocation's
original deadline covers configuration, model acquisition, execution and
bounded final reporting through write and flush. Blocking IO observes this
deadline cooperatively between operations. Exit zero is required for successful
report delivery; a writer failure can still leave partial metadata bytes.

## Embedding and validation scope

Construct `Int4ClassificationDocumentJobPlanner` from an actual
`Int4ClassificationPlanner`, `ClassificationBatchArgs`,
`Int4ClassificationMapLimits` and immutable `JobLimits`. Optionally select
`with_chunk_cohorts` with `Int4ClassificationMapMemoryLimits`. Call
`NlpEngine::job_int4_classification_document` with the resident model, factory,
`SourceJobRequest`, `JobHostLimits`, selected packed kernel, owned reader and
cancellation token. The low-level durable processor also supports explicitly
admitted serial or cohort backends.

Regression definitions cover private recipe/population authentication,
five-axis lifetime accounting, complete-result admission, cancellation,
resource ownership, CLI refusal before private input, bounded reporting and
real job storage with an explicitly synthetic executor. A separate ignored
actual-model fixture compares serial/cohort results and completed-job resume
without more work. Compilation, Rust test execution, native parity, crash
qualification, quality and performance remain pending the repository-required
controller-selected DSR and model-evidence gates. This route remains candidate
functionality and promotes no release or performance claim.
