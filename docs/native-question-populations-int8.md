# Complete-document INT8 question populations

`fnlp candidate answer-corpus` streams complete original documents with a
different question on each record through one resident INT8 model.
`fnlp candidate answer-job start|resume` uses the same document compiler and
native executor with authenticated retained results and explicit input replay.
These routes connect the question-answering contract in comprehensive plan
§7.7 with its complete-document and batch-first requirements.

Each document returns all independent passage answers, model abstentions and
whitespace ranges, with citations in original-document byte and Unicode-scalar
coordinates. Every required chunk must complete before the document succeeds.
The routes do not synthesize a global answer or select one answer from
conflicting passages. Exact citation checks establish source membership;
they do not establish semantic support, answer correctness or calibrated
answerability. These are uncalibrated candidate interfaces.

## Input and fixed policy

Use strict NDJSON records such as:

```jsonl
{"id":"who","text":"Alice visited Shanghai. Bob stayed in Paris.","task_args":{"question":"Who visited Shanghai?"}}
{"id":"where","text":"Alice visited Shanghai. Bob stayed in Paris.","task_args":{"question":"Where did Bob stay?"}}
```

`text` is the complete original source. It is not normalized, truncated to a
prefix, or replaced with generated text. `task_args` contains exactly one
required `question` string. Missing, null, duplicate or unknown fields and
array-valued arguments reject. Empty or whitespace-only questions reject.
Each nonblank source chunk receives that record's unchanged question.
An all-whitespace source has no native question task and is rejected.

`--options FILE` fixes one complete `AnswerOptions` object for the population;
omission selects the shipped defaults. For example:

```json
{"max_answer_scalars":2048,"max_citations":8,"max_quote_scalars":256}
```

Records cannot replace options, source passages, the model, budgets or execution
policy. The planner retains the exact original source/question and opaque task
commitments. Before native admission it reconstructs the complete plan and
compares every ordered execution identity and task commitment. A result is
accepted only after the existing complete-document task verifier checks the
finished output against that prepared request.

## Resident stream

```sh
fnlp candidate answer-corpus questions.ndjson \
  --model /models/nanbeige.int8.fnlpq \
  --memory-mib 16384 \
  --max-input-bytes 1048576 --max-question-bytes 16384 \
  --max-chunks 64 --max-new-tokens 256 \
  --prefill-rows 8 --compute-threads 4
```

The positional `-` reads stdin. The model path must identify an explicitly
chosen local current-candidate INT8 artifact. This command requires
`asupersync-runtime`; it does not require `metadata-store`.

The stream preserves record order. Exact `{"flush":true}` controls reset
duplicate-ID tracking without replenishing work budgets. Results use the
existing candidate-framed NDJSON protocol. Previously completed item frames
remain valid if a later record fails, but whole-stream success requires the
terminal `run_complete` frame and exit status zero. A failed document does not
publish its incomplete passage collection as a successful item. Terminal
completion follows physical worker join.

## Durable jobs

The durable command additionally requires `metadata-store` on Linux x86-64 or
AArch64. Use the existing job initialization workflow to obtain a protected
directory, job ID and owner-only key file. Result retention is explicit:

```sh
fnlp candidate answer-job start questions.ndjson \
  --model /models/nanbeige.int8.fnlpq \
  --memory-mib 16384 \
  --store-results --job-dir "$JOB_DIR" --job-id "$JOB_ID" \
  --key-file "$JOB_KEY_FILE" --limits qa-job-limits.json \
  --max-input-bytes 1048576 --max-question-bytes 16384 \
  --prefill-rows 8 --compute-threads 4 --materialize
```

Resume with the same complete original encoded population, model, options,
document policy and immutable job limits:

```sh
fnlp candidate answer-job resume questions.ndjson \
  --model /models/nanbeige.int8.fnlpq \
  --memory-mib 16384 \
  --store-results --job-dir "$JOB_DIR" --job-id "$JOB_ID" \
  --key-file "$JOB_KEY_FILE" --limits qa-job-limits.json \
  --max-input-bytes 1048576 --max-question-bytes 16384 \
  --prefill-rows 8 --compute-threads 4 --materialize
```

The population is frozen before execution, with IDs unique across the entire
population and no flush controls. Changing a question, source, record order or
encoded envelope changes the authenticated input. Resume authenticates the
population, model and fresh private recipe before repairing an uncommitted
tail or executing pending items. `--discard-uncommitted-tail` explicitly
permits the existing repair procedure after authentication; it does not repair
committed corruption. Completed items are authenticated and skipped.

To retain the exact input envelopes during creation, add
`--spool-input --max-retained-input-mib N`. Later resume can omit the input
positional argument and use
`--use-retained-input --max-retained-input-mib N`. These modes share the
existing [authenticated retained-input implementation](retained-inputs.md).
The retained population does not replace the model, options, key or immutable
limits. Original-input mode prices the complete population; retained mode
prices the authenticated index and bounded record/parser storage.

Stdout contains job metadata rather than source text or answers. Successful
items are retained in the protected result spool. `--materialize` publishes
the existing fixed `materialized.ndjson` only after the complete job succeeds.

## Work, memory and input encoding

Per-document limits cover the complete source/question pair, not one chunk:

| Authority | Relevant options |
|---|---|
| Original source and question | `--max-input-bytes`, `--max-question-bytes` |
| Complete chunk planning | `--max-chunks`, `--max-chunk-bytes`, `--max-tokenizer-calls` |
| Native work | The five `--max-document-*` native-work options |
| Grammar-mask work | `--max-mask-node-visits` per chunk; `--max-document-mask-node-visits` per document |
| Lifted QA verification | `--max-qa-citations`, `--max-qa-evidence-spans`, `--max-qa-scan-steps` |
| Complete answer and retained values | `--max-map-result-bytes`, `--max-live-value-bytes`, `--max-total-value-bytes` |

`answer-corpus` additionally accumulates all five native work axes, masks and
all three QA verification counters across the run. The five `--max-*`
native-work flags and `--max-run-mask-node-visits`, `--max-run-qa-citations`,
`--max-run-qa-evidence-spans`, and `--max-run-qa-scan-steps` set those ceilings.
Attempts reserve their complete allowance atomically. Failed execution and
flushes do not refund or reset it.

Durable jobs use the immutable `JobLimits` supplied by `--limits`. Native work
and mask counters remain spent across failures and resumes. The immutable
attempt cap conservatively bounds lifetime QA verification: the CLI derives
all three lifetime ceilings by multiplying the per-document allowances by the
maximum attempts. Rust callers must provide ceilings that cover the same
bound. Tokenizer work is not represented as a journal debit.

Byte-preserving control-excluding encoding remains the default. Compact
input uses `--input-encoding safe-bpe-v1` with finite per-call BPE limits and
the four `--max-total-tokenizer-*` limits. Each original source/question pair
gets one operation allowance spanning question encoding, chunk-sizing probes,
complete preparation and exact execution reconstruction. Failed probes retain
their consumption within that operation. Independent records receive separate
operations with the same fixed limits. A new durable invocation reconstructs
pending documents under the authenticated encoding policy; tokenizer totals
are invocation-local, not persisted lifetime counters. Compact encoding may
change tokenization and model output while preserving original source bytes.

`--prefill-rows N` selects bounded layer-major prompt execution.
`--compute-threads N` selects one persistent projection team across all records
and chunks. Both compose with compact input. Admission includes native KV and
scratch, prompt/team workspace, the complete source/question and identity
owners, native transcripts, citation-verification storage, the reduction
frontier and final output. Host and CLI share the same checked payload quote.
The memory ceiling models admitted ownership; it is not an observed process
RSS measurement. The example memory value must be adjusted to the selected
artifact and ceilings.

## Rust entry points and validation status

`Int8QuestionDocumentBatchPlanner` and
`NativeInt8QuestionDocumentBatch` provide the storage-neutral resident adapter.
Hosted callers use `NlpEngine::batch_int8_question_documents` and its
layer-major and persistent-team variants. Durable callers use
`Int8QuestionDocumentJobPlanner`, `Int8QuestionDocumentJobProcessor`, and
`NlpEngine::job_int8_question_documents` or its team variant. The public
`int8_question_document_corpus_requirements` and
`int8_question_document_job_requirements` helpers quote complete ownership
before private input or model access.

Focused regression definitions cover strict question-only input, original
source/question association, exact reconstruction, work and verification
ceilings, compact tokenizer ownership, admission, memory and resume contracts.
Source/API review and whitespace checks were performed. Compilation, executable
Rust tests, actual-model replay, task quality, throughput and the controller's
immutable-SHA DSR production checkpoint remain unrun. No model parity,
calibration or performance qualification is claimed.
