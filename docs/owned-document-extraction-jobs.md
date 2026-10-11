# Resumable complete-document extraction jobs

`fnlp candidate job start|resume --task extract --schema FILE --document-config FILE`
extends explicitly retained extraction jobs to original documents larger than one
native context. Without `--document-config`, the existing single-context job route
and its shared-default/per-record schema behavior are unchanged.

This uses the existing owned-storage profile: `metadata-store` and
`asupersync-runtime` on Linux x86-64 or AArch64. Unsupported profiles refuse before
key, configuration, model or population IO. Candidate results remain
`non_authoritative`; this does not activate or qualify an artifact.

## Explicit start and resume

```sh
fnlp candidate job start documents.ndjson \
  --task extract --schema schema.json --source-membership \
  --document-config document.json \
  --model local.fnlpq --memory-mib 8192 --preparation-mib 1024 \
  --store-results --job-dir private/job --key-file private/key \
  --job-id "$JOB_ID" --limits job-limits.json --materialize
```

The directory must already be owner-only, the protected key file must contain
exactly 32 raw secret bytes, and `JOB_ID` must be the previously generated random
128-bit ID encoded as 32 lowercase hexadecimal characters. The command creates
neither the directory nor the key. Keep the complete original population for
resume: it is admitted in memory, not silently saved as an input spool.

To resume, change `start` to `resume` and supply the same original population,
key, ID, model identity, exact schema, semantic document settings and lifetime
limits. A torn/uncommitted tail is refused by default. The existing explicit
`--discard-uncommitted-tail` option permits cleanup only after authentication of
the original contract and every committed frame. No retry is automatic.

Memory options above are example admission inputs, not measured requirements or
a guarantee that a particular artifact and population fit. The host separately
admits resident weights, complete population, journal, preparation, maximum
map/reduction frontier, coordinate scans, native KV/scratch and guarded output.

## Document settings

`document.json` is a closed, at-most-16-KiB JSON object. Every field is required;
unknown fields, duplicate keys, partial settings and out-of-range limits fail.
For example:

```json
{
  "schema_version": 1,
  "max_chunks": 64,
  "max_chunk_bytes": 2048,
  "max_tokenizer_calls": 8192,
  "max_result_bytes": 16777216,
  "max_live_value_bytes": 33554432,
  "max_total_value_bytes": 67108864,
  "verification": {
    "max_fields": 4096,
    "max_matches": 16384,
    "max_scan_steps": 67108864
  },
  "reduction_reserve_mib": 64
}
```

`max_chunks` is 1 through 256. Chunk bytes are an additional bound, never a
substitute for the actual pinned schema/scaffold/source tokenization and context
check. The whole original text is still bounded by `--max-input-bytes`, and its
complete original NDJSON frame by `JobLimits.max_input_bytes_per_item`.

`max_result_bytes` covers the complete document result and must fit the immutable
job result cap. `--max-result-bytes` remains a native **per-chunk** limit. The
live/cumulative value limits bound the common reduction coordinator, not measured
heap allocation. `reduction_reserve_mib` is additional modeled host-memory
headroom; it is not a source of native work authority and is not a semantic recipe
field. Host memory headroom can be changed without renewing lifetime work.

Document mode requires a shared exact `--schema` and refuses `--defaults`.
Records may omit `task_args` or provide `{}`; they cannot override the schema,
grounding, model, chunking, verification or retry policy. Other task families
reject `--document-config` rather than ignoring it.

## Durable work and interruption semantics

The separate `job-limits.json` retains the existing `JobLimits` format. Its
`max_work` is immutable lifetime authority across **all** invocations; it is not
multiplied by input count, maximum chunks, transport lines or retry count. It
must fund the whole-document attempt ceilings selected by the document settings.

The per-attempt model ceiling is derived from independent native contexts times
maximum chunks, across all five work axes. It is not one fictitious giant-context
attention triangle. Mask work is similarly priced across all possible chunks.
Each attempt durably reserves that whole-document ceiling before any forward.
Unused work and interrupted attempts are not refunded.

Independent coordinate verification has a per-document allowance. The frozen
factory requires each allowance times `JobLimits.max_attempts` to fit its lifetime
verification ceiling. The CLI derives that ceiling with checked arithmetic.
Because the existing journal persists attempt count before execution, this bound
includes failed, cancelled and acknowledgement-uncertain attempts after restart;
it does not depend on an in-memory ledger surviving the process. No journal
format migration or implicit authority reset is needed.

One journal ordinal and one committed output frame represent one **complete
original document**. A late invalid chunk or cancellation produces no successful
partial document frame. On explicit resume, committed documents are skipped;
an interrupted uncommitted document is retried as a whole with a new debit.
There is no within-document chunk checkpoint or automatic retry loop.

Each result retains independent exact-schema JSON strings for every chunk.
With `--source-membership`, original-coordinate field evidence is retained too.
Required fields must be satisfiable independently in each chunk. Objects are
never merged, repaired with guessed nulls or advertised as cross-chunk reasoning;
source membership proves occurrence, not factual truth. Exact numeric literals
remain inside native JSON strings rather than passing through floating point.

The existing guarded-output transaction syncs a complete result frame before
committing its journal pointer. One resident native engine and one process-owned
invocation serve the entire pending population. Output guards survive the spool
sync and journal acknowledgement. Stdout contains metadata only; private results
stay in the explicitly protected spool or verified materialization.

## Library and validation status

The sealed factory is
`jobs::runner::extract::document::Int8DocumentExtractionJobPlanner`; the real native
processor is `Int8DocumentExtractionJobProcessor`. Hosted callers use
`NlpEngine::job_int8_extract_document` with a resident model, factory, vocabulary,
existing `SourceJobRequest`, `JobHostLimits`, explicit reduction reserve, owned
reader and cancellation token. The runner validates recipe/lifetime relationships
before create/resume storage access, including for lower-level embedders.

Regression definitions cover frozen schema/population/limit binding, real pinned
multi-chunk planning, failed-attempt verification bounds, all durable work axes,
complete-result admission, pre-storage refusal, cancellation, physical ownership,
CLI restrictions, exact schema text, checked scaling and unavailable-before-IO
behavior. Implementation-worker compilation, Rust tests, Actions, DSR and native
model execution were not run under the controller-owned validation policy. These
definitions and source review are not passing-test, inference-quality or release
qualification evidence.
