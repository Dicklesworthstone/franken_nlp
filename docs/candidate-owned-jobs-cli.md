# Candidate job start and authenticated resume

`fnlp candidate job start` and `fnlp candidate job resume` connect the existing
owned-job runner to actual process-hosted INT8 inference. Supported tasks are
`ner`, `keyphrases`, `summarize`, `answer` and user-schema `extract`. They require
both `asupersync-runtime` and `metadata-store` on the existing Linux x86-64 or
AArch64 owned-storage profile. Other builds expose help but refuse execution
before opening inputs/configuration/keys/models or creating a runtime.

These are candidate operations, not catalog activation, publisher authentication,
model-quality or durability qualification. One model and one native engine serve
the pending population. Existing `fnlp job status`, `verify`, `materialize` and
`schema` remain separate model-free management commands, unchanged by this route.

## CPU scheduling for start and resume

Add `--compute-threads 4 --prefill-rows 8` to either command to use one four-thread
CPU team across pending records and process up to eight prompt tokens per group.
Either option can be used independently. The thread count includes the coordinator.
Both counts accept 1 through 64. Omitting both uses sequential execution.

The schedule applies to single-context extraction, NER, keyphrases, summaries and
passage QA. For complete-document extraction and source maps it also applies to
each chunk. The worker buffers and prompt scratch share the process memory ceiling
with the model, document and journal. On resume you may change either count and
supply the original authenticated task settings. Committed records are skipped,
and previous attempts and work keep their lifetime charges. An explicitly selected
schedule appears in the completion report as `compute_threads` (for widths above
one) and/or `prefill_rows`.

## Explicit retention and setup

`--store-results` is required on both start and resume. Results are retained in
an owner-only spool; the key authenticates commitments and stored state, **not
encryption**. Private result text is present in the protected spool and, after
explicit publication, in `materialized.ndjson`. Stdout is a metadata report only.
The CLI creates neither directories nor keys and never silently adopts an
existing job on `start`.

Provision a fresh protected directory, 32-byte secret and random 128-bit job ID
using the host's trusted entropy source. For example, this out-of-band Python
setup fails if the example directory already exists and prints no key:

```sh
python3 - <<'PY'
import os
import secrets
from pathlib import Path
os.umask(0o077)
root = Path.home() / '.fnlp-job-example'
root.mkdir(mode=0o700)
(root / 'job').mkdir(mode=0o700)
with (root / 'key.bin').open('xb') as key:
    key.write(secrets.token_bytes(32))
    key.flush()
    os.fsync(key.fileno())
(root / 'id.txt').write_text(secrets.token_hex(16) + '\n')
PY
```

The key file contains exactly 32 **raw bytes**, not hexadecimal text. Keep the
original key, ID and limits. Never regenerate the key for resume. The underlying
storage reader rejects insecure/symlinked key storage and protects job files
using its existing owner-only, no-follow, exclusive-lock contracts. Creation or
publication failures are not automatically deleted, overwritten or retried.

Write a bounded `job-limits.json`. The following is an illustrative finite
allowance, not a capacity/performance promise; it reserves at most 100 items and
200 total attempts, leaving retry room from the outset:

```json
{
  "max_items": 100,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 1048576,
  "max_snapshot_bytes": 67108864,
  "max_result_bytes": 1048576,
  "max_spool_bytes": 134217728,
  "max_materialized_bytes": 134217728,
  "max_journal_bytes": 67108864,
  "max_attempts": 200,
  "max_work": {
    "model": {
      "forward_positions": 500000,
      "projected_logits": 4000000000,
      "attention_pairs": 1000000000000,
      "projections": {
        "dot_products": 1000000000000,
        "multiply_accumulates": 100000000000000000
      }
    },
    "mask_node_visits": 200000000000
  }
}
```

All five model-work axes, total mask visits, attempts and storage ceilings are
immutable **lifetime totals**, authenticated across every resume. Raising item
or transport limits does not multiply compute authority. Failed attempts keep
their debits; unused reservations are not refunded. An exhausted original
lifetime contract cannot be widened on resume. Choose retry/work headroom before
starting. The limit file is capped at 16 KiB and rejects duplicate/unknown fields.

## Start and resume a source task

Input uses complete original NDJSON envelopes:

```jsonl
{"id":"article-1","text":"Alice moved to Paris."}
{"id":"article-2","text":"Bob works at Acme."}
```

Unlike live batch mode, durable populations require globally unique IDs, not
an epoch window, and refuse `{"flush":true}`. Only empty LF/CRLF lines are
ignored. The entire population must fit its immutable limits and the process
memory admission. It is held in memory, not automatically persisted as input.
Retain the original file yourself for resume, including each envelope's exact
bytes before LF. Changes to order, IDs, text or per-record arguments are refused.

```sh
fnlp candidate job start corpus.ndjson --task ner --store-results \
  --job-dir "$HOME/.fnlp-job-example/job" \
  --job-id "$(cat "$HOME/.fnlp-job-example/id.txt")" \
  --key-file "$HOME/.fnlp-job-example/key.bin" --limits job-limits.json \
  --model ./model.fnlpq --memory-mib 8192
```

To continue after an interruption, use the **complete original population**,
original key/ID/limits and unchanged task/model recipe:

```sh
fnlp candidate job resume corpus.ndjson --task ner --store-results \
  --job-dir "$HOME/.fnlp-job-example/job" \
  --job-id "$(cat "$HOME/.fnlp-job-example/id.txt")" \
  --key-file "$HOME/.fnlp-job-example/key.bin" --limits job-limits.json \
  --model ./model.fnlpq --memory-mib 8192 --materialize
```

Committed items are not inferred again. A failed or interrupted pending attempt
requires a new debit under the same lifetime allowance. Default resume refuses
uncommitted/torn spool tails or output stages. Only the additional explicit
`--discard-uncommitted-tail` flag permits their truncation/removal, and only
AFTER the original contract and every journal-authorized frame authenticate.
That flag is absent from `start`. There is no blind repair/force-resume mode.

`--materialize` is optional and publishes the fixed protected
`materialized.ndjson` only after all items commit, using the existing verified,
synced, no-replace transaction. The file contains ordered native result records,
not the live candidate batch transport wrapper. A later model-free `fnlp job
materialize` can publish retained results without loading weights or originals.

## User schemas and task defaults

For schema extraction, use `--task extract --schema schema.json` in either
invocation, with an optional `--source-membership` flag for explicitly annotated
`x-fnlp-source=verbatim` fields. Schema files are capped at 64 KiB and remain
exact text, including whitespace and high-precision numeric constants. The
shared schema must remain unchanged on resume. Verbatim defaults compile against
each original source, never placeholder evidence. Source membership is not
semantic accuracy. Unsupported grammar constructs are refused by the existing
per-record compiler, not retried as unconstrained generation.

Alternatively, `--defaults FILE` supplies complete `SourceBatchArgs` or
`ExtractionBatchArgs`, capped at 1 MiB, or individual records supply complete
`task_args` with bounded budgets. Shared schema and defaults flags conflict.
The fixed NER/keyphrase/summary tasks use their usual options when defaults are
omitted. QA invents no evidence: supply passages through defaults or record
arguments. Extraction without a shared schema/default likewise requires record
arguments. A record override cannot change subsequent defaults, the fixed task,
model, score/grammar policy or host ceiling. Default KV authority must cover the
entire allocated context. Changed defaults or compiler/task limits invalidate
resume rather than changing the meaning of pending items.

## Compact input with authenticated resume

Single-context `extract`, `ner`, `keyphrases`, `summarize` and `answer` jobs
accept `--input-encoding safe-bpe-v1` on both `start` and `resume`. For example:

```sh
fnlp candidate job start corpus.ndjson --task extract \
  --schema schema.json --input-encoding safe-bpe-v1 \
  --max-total-tokenizer-input-bytes 67108864 \
  --max-total-tokenizer-output-tokens 4194304 \
  --max-total-tokenizer-pair-checks 268435456 \
  --max-total-tokenizer-pair-bytes 4294967296 \
  --store-results --job-dir /private/extraction-job --job-id "$JOB_ID" \
  --key-file /private/extraction-job.key --limits job-limits.json \
  --model ./model.fnlpq --memory-mib 8192
```

The directory, key, random job ID and limits must already have been provisioned
as described above. Resume uses the same arguments with `resume` in place of
`start`, and the original complete population or the existing explicit retained
input route. Repeat the original encoding policy and all tokenizer ceilings:
they are part of the authenticated recipe, so changing them refuses resume
before pending inference or recovery repair. Existing byte-fallback recipes
retain their exact serialized shape and remain the default.

Each record receives a bounded tokenizer allowance. For extraction, the exact
raw schema and original source share that allowance; the byte cap is the
document limit plus the 64 KiB schema ceiling, capped at the encoder's 1 MiB
maximum. Source-task records use the existing document and prompt-token bounds.
QA shares one allowance across its question, passage IDs and all supplied
evidence. `--max-tokenizer-pair-checks` and `--max-tokenizer-pair-bytes` bound
per-record merge work. Schema/source or QA segments share one record allowance,
and compiler reconstruction does not renew the invocation allowance. Repeated
preparation must fit both its per-call cap and the remaining invocation ledger.
Exhaustion never silently falls back to another encoding.

An additional cumulative ledger bounds all encoding preparation in the current
invocation, including failed and cancelled preparation:

| Flag | Default invocation total |
| --- | ---: |
| `--max-total-tokenizer-input-bytes` | 67,108,864 bytes |
| `--max-total-tokenizer-output-tokens` | 4,194,304 tokens |
| `--max-total-tokenizer-pair-checks` | 268,435,456 checks |
| `--max-total-tokenizer-pair-bytes` | 4,294,967,296 examined bytes |

All four totals must be positive and are accepted only with compact input.
Spent preparation remains spent when the factory hands its planner to the
native job processor; later records use the remaining allowance. A resumed
invocation receives the same declared tokenizer allowance. These are
**invocation totals**, while native/mask work and attempts remain the original
persisted lifetime totals. Neither a new invocation nor compact input refunds
those durable debits.

CLI admission includes the selected encoder's exact workspace before private
input. The actual prepared planner's policy, cumulative caps and workspace must
match that quote, and the host independently reserves it with either the
complete population or retained-record buffers through physical completion.
Preparation borrows the original cancellation controller, including source
index construction and final plan sealing. Typed cancellation causes remain
intact, and metadata publication checks the original deadline.

Compact completion reports add `input_encoding` and
`invocation_encoding_work_limits`; byte reports omit both. The report still
contains no private content or prompt-derived commitment. Encoding preserves
the original bytes and excludes template-control token IDs, while changing
the prompt token sequence. It does not establish identical model output,
numerical equivalence, quality or measured speed.

Complete-document jobs using `--document-config` or
`--source-document-config` continue to require byte fallback. They reject
compact input before configuration, key, model or population access; their
chunk-level accounting contract is separate.

Library callers configure `Int8ExtractionJobPlanner::with_document_encoding`
with the fixed policy and invocation totals. Source jobs use the existing
`SourceTaskPlanner::with_map_document_encoding` cumulative owner. Keep that
owner across preparation and native execution; the hosted job entry points
admit its workspace independently.

## Admission, cancellation and failure reporting

Preparation is admitted before key/config/tokenizer allocation. The CLI checks
configuration and a necessary population-memory floor before model loading;
the host still performs the actual independent admission. One existing process
runtime loads the explicit candidate file and compares all retained facts after
independent metadata/load opens. The sealed factory is built before weights.
The full population is ingested and its typed envelopes checked before job-file
access or native forwards, but **after model weights are loaded**. There is no
claim that a late malformed input avoids the model load.

Host limits price the complete in-memory population and indexes, planner/schema
state, journal RAM, serialization, owned IO, native KV/workspace and guarded
results. `--journal-memory-mib` defaults to 64 and is distinct from the database
file limit; `--serialization-memory-mib` defaults to 16. Native context and
preparation options match the existing source commands. These are modeled
ledger commitments, not measured or OS-enforced RSS limits. Owned handles cross
one blocking/scoped job invocation with no caller-thread stdin/stdout locks.

The invocation's deadline spans configuration, loading, ingestion, recovery,
planning, inference, commits and optional publication. Native checkpoints do
not reset per item. Explicit resume starts a new invocation deadline/checkpoint
allowance but never renews lifetime work/attempts. `--max-stream-mib` defaults to
64 and `--max-input-lines` to 100000, both per-invocation transport caps; blank
lines also consume them. Cancellation is cooperative, not OS-IO preemption.

Success is one bounded JSON line with candidate evidence/provenance and a
metadata-only report: operation, task, public job ID, items/committed/attempts,
reserved work, committed spool bytes and materialization state. No private
source, result, schema, key, path or prompt-derived commitment is printed.
Errors are fixed-code JSON on stderr with `durable_progress_may_exist=true`.
An error, including cancellation or failed stdout delivery, NEVER promises
rollback. Authenticate stored status or explicitly resume to reconcile a lost
acknowledgement. No second stdout record is appended after a failed write.

## Validation scope

Added command, configuration, no-IO, real pinned factory, work/memory admission,
recovery-policy and metadata/diagnostic regression sources. Compact additions
cover both operations for all five tasks, exact schema/source preservation,
all eight authenticated per-record/total policy axes, fixed identity under
shrinking remaining budgets, failed-preparation debits, all eleven typed
cancellation causes, byte recipe/report compatibility, and exact full-input
and retained-input memory boundaries. They do not simulate
neural success. Rust compilation/tests, full-model execution, crash/power-loss
qualification, measured memory/performance and controller DSR validation were
not executed for this checkpoint. All outputs retain the non-authoritative
candidate evidence grade; artifact, numerical, task-quality and release gates
are unchanged. Persistent input spooling and external-sort populations remain
separate work.
