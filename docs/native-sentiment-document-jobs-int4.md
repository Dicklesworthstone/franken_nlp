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

## Work, memory and recovery

The five per-document native work flags cover all chunks and requested axes:
forward positions, projected logits, attention pairs, dot products and multiply
accumulates. Each ceiling must fit the corresponding immutable lifetime
`JobLimits.max_work` counter. Neither more documents, chunk cohorts, retries nor
resume multiply or renew those counters. Failed attempts remain debited.

Add `--cohort-chunks N` to share native decoder layers across complete independent
chunks of one document. `N` must fit `--max-chunks`, and `N * requested_axes`
must not exceed 64. Actual fixed-axis geometry is checked before model or
population loading. The schedule and exact width are part of the authenticated
job recipe; serial and cohort schedules cannot be swapped during resume.

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
