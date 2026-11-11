# Durable native INT4 sentiment of complete documents

`fnlp candidate source-int4 sentiment-document-job start|resume` scores an
authenticated population of complete original documents and retains verified
results. One original document is one durable item. Every chunk and requested
dimension must complete before that document commits; a resumed invocation skips
committed documents and runs pending documents under the original lifetime work
allowance.

The command combines [complete-document dimensional sentiment](native-sentiment-map-int4.md)
with [protected owned-job storage](owned-jobs.md). It requires `metadata-store`
and `asupersync-runtime` on Linux x86-64 or AArch64. Its results remain candidate
evidence, with the mapper's explicit uncalibrated aggregation semantics.

## Fixed dimensions and complete input

An optional private `sentiment-defaults.json` selects dimensions and policy:

```json
{
  "axes": ["valence", "arousal", "dominance", "approach"],
  "policy": {
    "minimum_peak_weight_ppm": 0,
    "maximum_normalized_entropy_ppm": 1000000
  }
}
```

Omitting `--defaults`, or providing `{}`, selects all four dimensions and the
policy above. Axes must be nonempty, unique, and chosen from those four names;
their order is preserved. Defaults cannot supply documents, prompts, budgets,
model identity, native schedules or precomputed scores. Duplicate keys and
unknown fields refuse. The same settings apply to every original document and
authenticate on resume.

The population is NDJSON:

```jsonl
{"id":"document-1","text":"The entire original first document, including the final paragraph."}
{"id":"document-2","text":"The entire original second document.","task_args":{}}
```

`task_args` may only be absent, null or an empty object. Per-record policy or
axis overrides and streaming flush controls refuse. IDs are unique across the
whole population. Supply the **complete original population on every resume**,
including committed documents, with the exact original bytes and order. Input
is admitted in memory for the invocation; it is not persisted for later recovery.

`--max-input-bytes` is the outer record and source admission bound. The immutable
`max_input_bytes_per_item` in job limits must fit it, and becomes the planner's
complete-source ceiling. The full serialized record must also fit the immutable
item limit. Mapping preserves original byte and Unicode-scalar coordinates and
never truncates a document to make it fit.

## Start and resume

Prepare an existing protected job directory, a protected regular file containing
exactly 32 random raw secret bytes, and a random 128-bit job ID represented by
32 lowercase hexadecimal characters. See [owned job storage](owned-jobs.md) for
the ownership and permissions contract. The command does not create or adopt
job directories or generate keys.

An illustrative `sentiment-job-limits.json` is:

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
documents fit. Serial execution needs the configured `--max-map-result-bytes`
cap; chunk-cohort execution additionally needs 4096 bytes for its complete result
wrapper. The illustrated result limit covers either schedule at the default
16 MiB semantic result cap. `mask_node_visits` must be zero.

```sh
fnlp candidate source-int4 sentiment-document-job start documents.ndjson \
  --defaults sentiment-defaults.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib "$MEMORY_MIB" \
  --group-width 128 --context-tokens 2048 --max-chunks 8 \
  --store-results --job-dir /private/sentiment-job \
  --job-id "$JOB_ID" --key-file /private/sentiment-job.key \
  --limits sentiment-job-limits.json --materialize
```

Use `resume` in place of `start`, preserving the population, settings, model
identity, group width, planning limits, chunk schedule and job limits. Use `-`
for stdin. Source conversion through `--source-dir` plus `--source-manifest`
can replace the independently pinned snapshot route. Explicit
`--save-snapshot NEW_FILE` saves the admitted model before population ingestion;
that model file can remain if a later job or report-delivery step fails.

## Compact input and resume

Add `--input-encoding safe-bpe-v1` and repeat the same encoding policy on every
resume. `--max-tokenizer-input-bytes`, `--max-tokenizer-output-tokens`,
`--max-tokenizer-pair-checks` and `--max-tokenizer-pair-bytes` bind one complete
document attempt, including initial preparation, all chunk probes and exact
reconstruction for execution. The stored private recipe authenticates every
ceiling before replay or tail repair. A changed policy refuses resume, including
when `--use-retained-input` reads the authenticated original population.

Compact serial, persistent-team and chunk-cohort jobs admit one encoder
workspace. Numerical teams and chunk cohorts are mutually exclusive. Reaching a
tokenizer ceiling refuses the document before any native forward, with no
truncation or fallback. Per-pass result work remains deterministic; the private
attempt ledger includes both planning passes. Native work and failed attempts
retain their separate durable lifetime semantics. Byte fallback preserves the
existing recipe and output shape. Compact mode accepts pinned sources or a
pinned snapshot, and refuses `--save-snapshot` before IO.

Embedding callers select `Int4SentimentPlanner::with_document_encoding` before
constructing `Int4SentimentDocumentJobPlanner`; no record argument may override
that sealed policy.

## Persistent numerical teams

Add `--compute-threads 4` to `start` or `resume` for one coordinator and three
persistent output-row workers across the complete invocation, with an additional
runtime thread. Every original document, chunk and selected sentiment axis uses
that team. The existing scorer and reducer retain their original axis order,
decision thresholds, coverage, distributions, source spans and aggregate
statistics. Byte fallback and explicit compact input use their existing fixed
policies and work ledgers.

Width one preserves serial execution; widths 2..64 add one worker workspace
and the corresponding process thread capacity to pre-IO admission. Multiple
workers cannot combine with any explicit `--cohort-chunks`, including one.
All workers physically join before successful job progress is returned, and
the original invocation deadline continues through final metadata publication.

Numerical worker width is excluded from the authenticated recipe. An explicit
resume may select another admitted width while retaining the original inputs,
axes, thresholds, encoding, task limits and native lifetime work. Attempts and
failed-work debits never renew. Stdout adds optional physical `compute_schedule`
metadata (`int4-output-row-team-v1`, `declared_compute_threads`); width one omits
it and stored semantic results keep their serial format.

Embedding callers use `NlpEngine::job_int4_sentiment_document_with_threads`
with `compute_threads` after `kernel` and before `reader` for widths 2..64.
The existing method retains serial execution. Source review and regression
definitions cover admission, ownership, cancellation, output framing and recipe
invariance; compilation, executable tests and model comparison remain pending
the prescribed DSR production checkpoint.

## Work, memory and recovery

The five per-document native work flags cover all chunks and requested axes:
forward positions, projected logits, attention pairs, dot products and multiply
accumulates. Each ceiling must fit the corresponding immutable lifetime
`JobLimits.max_work` counter. Neither more documents, chunk cohorts, retries nor
resume multiply or renew those counters. Failed attempts remain debited.

Add `--cohort-chunks N` to share native decoder layers across complete independent
chunks of one document. `N` must fit `--max-chunks`, and the fixed selected axes
must satisfy `N * requested_axes <= 64` before model or population loading.

Optional `--max-axes 1..4` narrows the initial memory quote and task KV ceiling.
An explicit cap must satisfy `N * max_axes <= 64` before any configuration or
population is read, and fixed defaults may not select more axes than that cap.
Omitting the option preserves the original four-axis task ceiling and its
conservative memory quote; actual fixed-axis geometry is still checked after
defaults are parsed. Existing jobs created without the option remain resumable
without it.

For a new two-axis job, use defaults such as `{"axes":["valence","arousal"]}`
with `--max-axes 2 --cohort-chunks 3`. This admits six simultaneous native heads
instead of the twelve in the omitted-option quote. It does not increase any
work allowance. A cap below four requires compatible explicit defaults, since
omitting `--defaults` still selects all four axes.

The chunk-cohort schedule, exact chunk width, fixed axes and task budget are
part of the authenticated job recipe. Numerical worker width is separate.
Repeat the authenticated settings on resume, including `--max-axes` when
it was used at creation. Adding a narrower cap to an existing job would change
its authenticated task budget; it is not a way to rewrite stored authority.

Admission includes the full population, mapping plans, native execution,
retained results, complete publication, durable storage buffers and frozen
chunk identities needed during replay. A narrow job item ceiling retains the
more conservative initial input memory reservation. Work and result ceilings
refuse before original-population IO.

Interrupted documents have no partial durable result. An explicit resume can
retry an unfinished document from its beginning if original work and attempt
allowances permit. `resume --discard-uncommitted-tail` explicitly authorizes
discarding uncommitted spool bytes or stages after authentication; ordinary
resume refuses such tails. Committed records are retained and skipped.

Stdout contains bounded model and progress metadata, including committed count,
attempts and reserved work. It does not print source text, axes, policy, scores,
private paths or secret material. `--materialize` creates verified private
`materialized.ndjson` only after every original document commits. Existing
materialized files are not overwritten. Use the [job management CLI](job-management-cli.md)
to inspect or verify retained state.

One elapsed-time deadline spans configuration reads, model loading or saving,
job execution and metadata delivery. Report encoding must finish before any
stdout write. A timeout or flush failure returns a nonzero exit status; stdout
may already contain completion metadata when delivery or the final deadline
check fails. Check the process exit status even when a completion record was
written. Durable progress can exist when publication fails: authenticate the
stored job instead of assuming rollback.

The retained document output preserves per-chunk distributions and lineage,
plus scalar-weighted descriptive aggregates. Policy and estimated-coverage
thresholds govern whether a dimension publishes an estimate or abstains. This
does not establish full-context equivalence, calibrated psychological measures,
semantic accuracy or release certification.

## Embedding and validation scope

Construct `Int4SentimentDocumentJobPlanner` from an actual
`Int4SentimentPlanner`, fixed `SentimentBatchArgs`, `Int4SentimentMapLimits` and
immutable `JobLimits`. Optionally select `with_chunk_cohorts(chunks)`, then use
`NlpEngine::job_int4_sentiment_document` with the resident model,
`SourceJobRequest`, `JobHostLimits`, packed kernel, owned reader and cancellation
token. The implementation shares the native sentiment corpus processor and
existing authenticated `JobRunner`.

An explicitly ignored actual-model fixture covers one-axis and four-axis
multichunk UTF-8 populations, serial/cohort output and persisted-work equality,
completed resume without new work, changed-input/policy refusal, cancellation
ownership and resource drainage. It requires the controller's pinned source,
explicit memory and isolated private test directory. It has not been executed.
Compilation, Rust tests, native parity, crash qualification and performance
remain pending the prescribed DSR production checkpoint and separate model gates.
