# Resident candidate source-task NDJSON CLI

`fnlp candidate batch` connects the existing INT8 source corpus implementation
to the executable. It loads one explicitly selected local current-candidate
model and retains one native engine, tokenizer/planner and vocabulary for the
whole bounded input stream. There is no per-document model reload or runtime
entry. Execution remains ordered and serial, not neural batch-M/GEMM execution,
a prefix-sharing optimization, an inference service or a durable job.

This is code-first integration. Compilation, Rust tests, real-model execution,
quality, performance and controller DSR qualification were not run during its
implementation. Every emitted event retains the candidate's non-authoritative
status; the command cannot activate a catalog or promote release evidence.

## A corpus invocation

```sh
fnlp candidate batch corpus.ndjson --task ner \
  --model ./model.fnlpq --memory-mib 8192 --max-requests 1000
```

Omit the input path or use `-` for stdin. `--task` is one of `ner`, `keyphrases`,
`summarize` or `answer`, fixed for the entire run. The first three tasks default
to their existing typed options and the explicit CLI task budget:

```jsonl
{"id":"article-1","text":"Alice moved to Paris."}
{"id":"article-2","text":"Bob works at Acme."}
{"flush":true}
{"id":"article-1","text":"A new epoch may reuse this ID."}
```

Input uses the existing `{id,text,task_args?}` batch protocol, not the single
request `answer` object's shape. For QA, `text` is the question, and `task_args`
must include the typed `answer` task options, finite budget and original
`passages`. No default evidence is invented. An optional local `--defaults FILE`
can supply a complete `SourceBatchArgs` object for the selected task, including
shared QA passages. Defaults JSON is capped at 1 MiB and parsed through the
central duplicate-key-rejecting boundary. Unknown fields, a different task,
invalid options or a budget exceeding the CLI ceiling are refused before
weight loading. Default KV authority must cover the entire allocated context,
not just the length of a particular document.

A per-document `task_args` replaces defaults for that document only; it cannot
change the selected task, runtime, model identity, thinking/tool mode or host
ceiling. Its full source/prompt/grammar validation runs under the same corpus
control and budgets. Failed source validation is a failed document, never a
successful answerability abstention or an automatic free-generation retry.

## Output and exit status

Every line is a completed candidate wrapper with public artifact provenance:

```text
{
  "protocol": "fnlp-candidate-batch-v1",
  "schema_version": 1,
  "scope": "real-artifact-current-candidate",
  "evidence": "non_authoritative",
  "model_id": "Nanbeige4.2-3B",
  "source_revision": "...",
  "source_root_sha256": "...",
  "logical_model_sha256": "...",
  "quant_recipe": "...",
  "task": "ner",
  "record": { ... original fnlp-item-local-batch-v1 event ... }
}
```

Actual output is NDJSON with no pretty-printing. The nested record preserves
its original canonical bytes: no parsing and rewriting of precise numbers or
JSON string values is performed by the wrapper. Events include `run_start`,
`doc`, `doc_error`, `flush`, and terminal `run_complete` or `run_error`. Successful
source results retain typed spans/citations and complete observed model work;
those structural checks do not constitute model-quality certification.

A protocol `run_complete` means input reached EOF and terminal delivery was
flushed. It does **not** assert that every document succeeded. The CLI exits
nonzero whenever the summary has any failed documents, or on a terminal native,
resource, cancellation or IO failure. Earlier completed document frames are not
retracted when later work fails. Inspect each nested event and its summary.

Each native event is staged in one bounded writer buffer. No prefix is written
until the complete event and whole-wrapper byte bound are checked. The existing
hosted output guard remains held through write and flush. Partial writes and
flush failures poison output permanently: neither a retry nor a second error
record is appended to an unknown transport prefix. A dropped incomplete buffer
is never flushed implicitly. No whole-corpus result buffer is accumulated.

## Bounds and ownership

Source-task host options match the single-document commands, including context,
constrained output tokens, typed-result bytes, grammar/mask limits and modeled
preparation memory. Additional defaults are 1000 nonempty input records, 1 MiB
per NDJSON line, 1024 MiB total input and 1024 MiB total output. `--max-input-bytes`
remains the source/question planning limit; `--max-input-mib` is the whole input
stream limit. `--max-line-bytes` includes task-argument JSON and syntax.

All emitted bytes, including provenance on every line, count toward
`--max-output-mib`. The runner's payload budget is reduced by a conservative
4096-byte allowance for each possible event, with extra terminal/failure slots.
Actual framing size and the actual emitted-byte sum are checked again by the
writer. A whole-output budget too small to reserve framing is rejected before
IO. Inner `max_result_bytes` never substitutes for the full transport budget.

Complete native work is bounded by the checked sum of independent per-request
context ceilings, including decoder/head projections and causal attention.
Mask visits, work, input/output bytes, requests, deadline and checkpoints do not
reset at a flush, failed document or early finish. Flush only acknowledges local
output delivery and resets the existing epoch's duplicate-ID set. It is not a
filesystem durability or remote-processing acknowledgement.

The process thread transfers owned buffered input and owned output handles to
one existing hosted blocking invocation. It holds no caller-thread stdin/stdout
locks that could block the worker. The runtime, resident model and process
ledger are the same implementations used by single candidate requests. Input,
output, native workspace and preparation charges survive until actual physical
completion. The CLI preparation reserve and the host's corpus-input reserve
conservatively overlap; these are explicit modeled commitments, not measured
RSS ceilings. No helper thread, second runtime, detachment or hidden network is
introduced.

Cancellation is cooperative. Native work and bounded planning share one elapsed
budget; input reads and filesystem calls are not safely preempted. Default
builds without `asupersync-runtime` expose help but refuse before reading input,
opening defaults/models or writing any stream output.
