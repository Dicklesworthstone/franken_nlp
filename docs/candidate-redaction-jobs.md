# Retained native redaction jobs

`fnlp candidate redact --ndjson --store-results` adds authenticated durable
execution to the existing native redaction corpus path. Both single-context
and `--chunked` documents use the SAME native adapters as live NDJSON: source-
bound NER plus selected rules, transactional editing, and (by default) fresh
NER/rule verification of the actual edited text. No rules-only fallback,
per-document model reload, automatic retry or intermediate NER transcript is
introduced.

Retention is opt-in. The ordinary command and live NDJSON output are unchanged.
Retained mode writes only completion metadata to stdout. Redacted text and any
explicitly requested coordinate maps remain private content in protected job
storage. Neither redaction nor pseudonyms establish anonymity; model omissions,
Unicode obfuscations and entities split across chunk boundaries remain possible.

## Start and resume

Provide an existing owner-only directory, a protected regular file containing
32 RAW random job-secret bytes, a random 128-bit job ID, and immutable JobLimits
JSON. The directory and keys are not created automatically. See `owned-jobs.md`
for platform/storage requirements and `candidate-scored-jobs.md` for JobLimits.
The commands require `asupersync-runtime` and `metadata-store` on supported
Linux x86-64/AArch64 targets plus a compatible local current-candidate artifact.

```sh
fnlp candidate redact original.ndjson --ndjson --store-results \
  --model ./model.fnlpq --memory-mib 8192 \
  --job-dir ./protected-job --job-id "$JOB_ID" \
  --job-key-file ./job.key --job-limits ./job-limits.json

fnlp candidate redact original.ndjson --ndjson --store-results --resume \
  --model ./model.fnlpq --memory-mib 8192 \
  --job-dir ./protected-job --job-id "$JOB_ID" \
  --job-key-file ./job.key --job-limits ./job-limits.json --materialize
```

Use `--chunked` on BOTH commands for long documents. Original rules still scan
whole text; NER chunks preserve original source coordinates. Verification plans
new chunks from the transformed text. A successful durable record is an entire
completed document, never a subset of NER chunks. Verification is enabled by
default; an explicit `--no-verify` changes the frozen job contract and must be
preserved on resume. Clean verification means clean under the declared detector
union, not proof that every kind of PII has been removed.

Every invocation receives the COMPLETE original ordered NDJSON population:
`{"id":"item-1","text":"original text"}`. Original inputs are not saved for you.
Records may omit task_args or use `{}`; records cannot replace actions, detector
types, rules, verification, budgets, chunking or keys. Flush records are not
accepted in durable populations. Resume authenticates the original population,
model and private recipe before storage repair or model forwards, then skips
committed items. Failed attempts retain their lifetime work debit.

`--discard-uncommitted-tail` requires `--resume` and requests the existing
post-authentication repair of uncommitted spool/staging bytes. It never deletes
or rewrites committed results. `--materialize` publishes the fixed protected
`materialized.ndjson` only after all items commit. A failed invocation or stdout
write may follow durable progress: preserve the inputs/keys and explicitly
resume rather than assuming rollback. Storage is not promised to be encrypted.

## Pseudonym scope survives interruption

Pseudonymization retains the existing inherited-stdin key interface:

```sh
fnlp candidate redact original.ndjson --ndjson --store-results \
  --action pseudonymize --key-stdin --key-id pii-v1 --namespace project-a \
  --model ./model.fnlpq --memory-mib 8192 \
  --job-dir ./protected-job --job-id "$JOB_ID" \
  --job-key-file ./job.key --job-limits ./job-limits.json < ./pii.key
```

`pii.key` supplies raw 32..4096-byte caller-owned high-entropy pseudonym material;
`job.key` is the separate protected 32-byte job-authentication secret. Neither
has an argv key-bytes option. The original corpus must be a file when stdin is
used for the pseudonym key. Reuse the same options and both original keys for
resume; add only `--resume` and the desired repair/materialization operations.

The recipe freezes the ACTUAL key commitment and keyed namespace scope, not
just the public key ID or optional expected commitment. Changing key bytes
while keeping the same key ID, or changing namespace while keeping the key,
refuses resume. Raw key bytes, namespace text and source text are not serialized
into the job recipe. Only full256 pseudonyms are supported by this retained
route; no per-item truncated-pseudonym scope or implicit key rotation is used.

## Admission and lifetime ceilings

All native, mask, rule, edit, detector, source, chunk and reduction limits are
retained in the private authenticated recipe. The same process resource ledger
owns resident weights, full KV capacity, all map/NER intermediates, input
population, recipe, journal and output reservations. Output guards survive
spool synchronization and journal acknowledgement, not merely model completion.
Reservations are modeled ownership bounds, not observed RSS measurements.

`--max-requests` bounds the admitted job population. `--max-input-mib` bounds
invocation input transport and must accommodate the immutable snapshot cap.
The immutable `max_input_bytes_per_item` applies to the entire original JSON
envelope and must not exceed `--max-input-bytes` (or `--max-line-bytes`). This is
conservative: JSON syntax/escaping counts toward that cap. Larger transformed-
text planning allowances cannot be reused to admit larger original inputs.
`max_result_bytes` must cover `--max-result-bytes`; combined immutable spool and
materialization byte ceilings must fit `--max-output-mib`.

`--max-job-input-lines` includes blanks and defaults to 100000. Journal and
serialization RAM reservations default to 64 and 16 MiB and can be priced with
`--journal-memory-mib` and `--serialization-memory-mib`. These are additional to
their disk caps. Native/mask whole-invocation ceilings never renew within an
invocation; immutable JobLimits also bound lifetime work and attempts across
explicit resumes. Complete attempts are reserved before execution, including
verification whose exact prompt exists only after editing. No observed-work
refund or partial-document success is introduced.

## Grouped INT4 retained jobs and numerical worker width

`fnlp candidate source-int4 redact-job start|resume` retains complete-document
INT4 redaction through the same authenticated job runner. Its syntax uses
explicit `start` and `resume` subcommands, `--key-file` for the protected
job secret and `--limits` for immutable lifetime limits. The separate
`--key-stdin` option supplies a pseudonym key when that action is selected.
The job requires `metadata-store` and `asupersync-runtime` on Linux x86-64
or AArch64.

```sh
fnlp candidate source-int4 redact-job start documents.ndjson \
  --snapshot ./nanbeige.int4 --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 16384 --compute-threads 4 \
  --store-results --job-dir ./protected-job --job-id "$JOB_ID" \
  --key-file ./job.key --limits ./job-limits.json --materialize
```

The memory ceiling is illustrative. Supply the complete original ordered
population on resume unless explicit [retained-input replay](retained-inputs.md)
was selected. Both external input and retained replay support numerical teams.
Original bytes/order, model identity, detector/action/verification policy,
all semantic limits and the actual full256 key/scope still authenticate before
repair or inference. Each original document remains one durable result.

`--compute-threads N` defaults to one. Widths two through 64 keep one team
and native engine through the complete physical invocation: all pending
documents, original and edited-source NER, result serialization and durable
commit. One worker workspace and the process thread envelope are admitted
before private IO. They do not multiply with population size or verification
stages. The borrowed authenticated population and native processor drain under
the existing runner, journal and spool ownership; public success follows
physical worker join.

Physical worker width is excluded from the immutable recipe. A job started
at width four may explicitly resume at width two or one while keeping all
authenticated inputs and semantic settings the same. This does not renew
attempts, the five persisted native work axes or mask visits. Failed attempts
retain the original complete-document debit, including the fresh verification
envelope; chunk/stage boundaries never become resumable partial results.

Existing native chunk cohorts remain a separate schedule. They authenticate
their explicit width and context in the job recipe and must be repeated on
resume. Multiple numerical workers cannot combine with any
`--cohort-chunks`, including a width of one. Unsupported widths and mixed
schedules refuse before key, population or model IO.

Stdout keeps the metadata-only
`fnlp-candidate-int4-redact-job-v1` report. An optional `compute_schedule`
records the declared numerical width and schedule version; serial responses
omit it. No per-item physical projection count or speedup is fabricated.
Stored edited results, materialization and original serial recipe bytes are
unchanged. The same output-delivery and post-commit cancellation rules apply:
a later error does not imply that already committed work was rolled back.

The library entry point is
`NlpEngine::job_int4_redact_document_with_threads`, with the same arguments
as the serial method through the kernel, then explicit worker width, owned
reader and cancellation. It returns the existing `JobProgress`. Serial and
native-cohort entry points remain available; the implementation reuses the
existing batch processor, authenticated runner and storage format.

New source regression definitions cover admitted numerical widths, unchanged
lifetime budgets and key authority, retained/external population memory,
result guards and optional metadata. Recipe preservation and physical drain
were reviewed against the existing durable processor and shared dispatcher.
Compilation, executable regressions, model parity, crash-recovery execution
and performance remain **UNRUN** for this change pending controller-authorized
DSR validation.

## Validation scope

Added tests cover keyed scope/recipe mismatches, work and memory boundaries,
chunk/reduction limits, strict consent and no-IO failures, shared actual pinned
configuration, and metadata-only completion. They are test sources, not
execution receipts. Rust/DSR execution, native-model behavior, task quality,
numerical parity and performance have not been established for these changes.
