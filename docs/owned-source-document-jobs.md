# Resumable whole-document source tasks

This page describes the default independent-map mode. An optional `reduction`
object in the same document configuration now enables one selected cited summary
or one document-wide keyphrase ranking per durable result; see
[Durable document reduction](owned-document-reduction.md). Omitting that object
preserves the behavior and recipe described below.

`fnlp candidate job start|resume --task ner|keyphrases|summarize
--source-document-config FILE` runs the existing native source mapper over each
complete original document and retains one durable result per document. It adds
long-document execution to source jobs; omitting the option preserves their
single-context behavior. Extraction keeps its separate `--document-config`
option and shared-schema contract.

This is the existing owner-only Linux x86-64/AArch64 job profile with
`metadata-store` and `asupersync-runtime`. Unsupported builds reject the operation
before opening a key, configuration, model or input. Candidate results remain
`non_authoritative`; no artifact activation or release qualification is implied.

## Start and resume

```sh
fnlp candidate job start documents.ndjson \
  --task ner --source-document-config source-document.json \
  --model local.fnlpq --memory-mib 16384 --preparation-mib 1024 \
  --store-results --job-dir private/job --key-file private/key \
  --job-id "$JOB_ID" --limits job-limits.json --materialize
```

The existing directory must be owner-only, the protected key file must contain
32 raw secret bytes, and `JOB_ID` must be the previously generated random
128-bit ID encoded as 32 lowercase hexadecimal characters. No directory or key
is silently created. The complete original population is admitted in memory;
it is not automatically persisted as an input spool. Retain it separately.

To resume, change `start` to `resume` with the same original population, key,
ID, task/options, model identity, semantic document settings and lifetime limits.
Committed documents are skipped. An interrupted uncommitted document retries
as a whole with another work debit; no within-document chunk checkpoint or
implicit retry exists. The existing `--discard-uncommitted-tail` flag is an
explicit resume-only cleanup policy after authentication, not a bypass for
mismatched inputs, settings, or corrupt committed frames.

Use `keyphrases` or `summarize` instead of `ner` for those task families.
`answer` is intentionally refused: question-aware evidence selection is not the
independent source mapper, and source failures must never become QA abstentions.
`--schema`, `--source-membership`, and extraction's `--document-config` cannot be
combined with the new source option.

## Fixed options and document settings

Task defaults use the existing code-owned NER, keyphrase, or cited-summary
options. An explicit `--defaults FILE` may supply the matching complete
`SourceBatchArgs`, including its per-chunk budget, within the host's ceilings.
These options are fixed for the job. Records may omit `task_args` or supply `{}`;
they cannot replace the task, options, budgets, model, partitioning, or schema.
All original envelopes are authenticated and checked before job-file access.

`source-document.json` is a closed object of at most 16 KiB. Every field below
is required. Duplicate/unknown keys, partial settings, and invalid ranges reject.
Unlike extraction's configuration, this source configuration has no unused
`verification` field that could be mistaken for enforced authority.

```json
{
  "schema_version": 1,
  "max_chunks": 64,
  "max_chunk_bytes": 2048,
  "max_tokenizer_calls": 8192,
  "max_result_bytes": 4194304,
  "max_live_value_bytes": 16777216,
  "max_total_value_bytes": 67108864,
  "reduction_reserve_mib": 64
}
```

`max_chunks` is 1 through 256. Chunk byte limits supplement actual pinned source
tokenization, never a characters-per-token guess. The factory reserves the exact
code-owned task scaffold plus maximum output tokens before partitioning; every
resulting chunk also passes complete task compilation. `--max-input-bytes`
bounds one original document's UTF-8 text, while
`JobLimits.max_input_bytes_per_item` independently bounds its complete NDJSON
envelope. `--max-result-bytes` remains the per-chunk typed native result limit;
this configuration's `max_result_bytes` covers the entire document envelope
and must fit `JobLimits.max_result_bytes`.

The live and cumulative limits belong to the common reduction coordinator.
`reduction_reserve_mib` supplies additional modeled host memory, not native work
or a semantic recipe. Changing that headroom does not replenish lifetime work.
Memory values in the command are example admission inputs, not measured artifact
requirements. The host additionally prices the complete population, all retained
prompt/grammar plans, maximum native-result/reduction frontier, journal,
serialization/IO buffers, resident weights, full KV, scratch, and guarded output.
These are explicit reservation models, not OS-enforced RSS bounds.

## Durable work, identity, and result meaning

Each original document is fully planned before its durable attempt begins.
The complete map's actual planned work across all five native counters and mask
work is journaled before the first forward. Neither maximum chunks, input count,
epochs, nor retry count multiplies `JobLimits.max_work`. Preparation is model-free;
unused native work is not refunded and failed attempts remain charged on resume.

To avoid self-referential storage, preparation retains the exact original text,
complete ordered execution identities and work snapshot. Execution reconstructs
the map, checks that entire snapshot and its private factory ownership, and admits
every chunk identity before any forward. A changed final chunk or a prepared
object from another factory is refused, not accepted because its first chunk
looks valid. Native capacity checks price the entire allocated KV context.

One resident native engine serves all pending documents in one process-owned
invocation. A late chunk failure, cancellation, work violation or invalid result
cannot produce a successful partial document. One whole-result guard survives
canonical staging, spool sync and journal acknowledgement. Verified ordered
materialization uses the existing publication transaction. Stdout contains
metadata only; private results stay in protected job storage.

Results are source-ordered **independent chunk maps**, with exact original UTF-8
byte/scalar coordinates for entities, phrases and summary citations. They do not
claim globally resolved entities, cross-chunk phrase reranking, synthesized global
summaries, or equivalence to single-context inference. A source occurrence proves
membership, not factual truth or task-quality certification.

## Library and validation

Library hosts use `jobs::runner::source::document::Int8SourceDocumentJobPlanner`,
`Int8SourceDocumentJobProcessor`, and `NlpEngine::job_int8_source_document`.
The sealed planner freezes actual pinned tokenizer/template identities, task
options, compiler/source limits, effective partition settings, reduction limits,
and the exact immutable job contract. A factory does not grant memory or model
authority. Lower-level use still requires real admission and the durable runner.

Regression definitions cover all three pinned task families, full-map identity
and work reconstruction, cross-factory and last-chunk substitution, keyed
recipe/population binding, lifetime limits, atomic debits, cancellation, native
capacity, closed CLI/configuration, unsupported-profile refusal and host ownership.
Compilation, Rust tests, CI, DSR and native/model execution were not run by this
implementation worker under the controller-owned validation policy. Source/API
review and added definitions are not passing-test or inference-quality evidence.
