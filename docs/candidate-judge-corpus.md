# Candidate bulk and retained judgments

`fnlp candidate score-batch --task judge` evaluates ordered NDJSON using one
resident strict-INT8 model and the existing complete-head judgment processor.
`fnlp candidate score-job start/resume --task judge` uses that SAME processor
with the existing authenticated population, journal, private result spool and
explicit resume lifecycle. There is no second judgment implementation.

These are current-candidate, non-authoritative execution routes. Judgments are
uncalibrated model outputs, not truth certificates. All three existing modes
are supported: pairwise, rubric and full-source faithfulness.

## Live evaluation

A defaults file supplies task data and an explicit policy, never the document,
TaskBudget, exact tokens, model identity or execution work. For example,
`comparison.json` can contain:

```json
{
  "mode": "pairwise",
  "criterion": "Prefer the answer that directly addresses the question.",
  "b": "Packages arrive on business days.",
  "policy": {
    "minimum_margin_milli": 0,
    "maximum_order_disagreement_milli": 1000
  }
}
```

The corresponding `answers.ndjson` contains the original candidate A texts:

```jsonl
{"id":"answer-1","text":"The package is due on Tuesday."}
{"id":"answer-2","text":"Delivery is scheduled for Tuesday morning."}
```

```sh
fnlp candidate score-batch answers.ndjson --task judge \
  --defaults comparison.json --model ./model.fnlpq --memory-mib 8192
```

The threshold values are illustrative policy declarations, not calibrated
recommendations. Pairwise thresholds are thousandths of natural-log score
ratios; they are NOT ppm probabilities. Both presentation orders must complete.

For rubric evaluation, defaults are `{mode:"rubric",rubric:{...},policy:{...}}`,
using the same `RubricDefinition` and `RubricPolicy` as `candidate judge`.
`text` is the document. Every criterion is retained in the complete result.

For faithfulness, defaults are `{mode:"faithfulness",claim:"...",policy:{...}}`,
using `FaithfulnessPolicy`. `text` is the ENTIRE original source. The planner
requires the full source to fit, checks every evidence window and never
substitutes a truncated source or retrieval excerpt. When there is only one
window, it reuses the whole-source head rather than scoring it twice.

No defaults file is required when each record supplies a complete `task_args`
object. This is the existing `JudgeBatchArgs` wire shape: mode-specific data,
explicit policy and `budget`. It is a complete replacement for defaults, not a
partial merge. All five budget fields must fit the host's task ceiling; record
JSON cannot increase input/output tokens, output bytes, grammar states or KV.
Missing defaults AND missing item arguments reject the record instead of
inventing a criterion, policy or successful empty judgment.

The live runner emits ordered candidate-framed events. A recoverable item
rejection does not erase earlier completed records; a failed item means a
nonzero command exit. Native corruption, admission failure and cancellation
remain terminal. Sink failure does not imply earlier writes were rolled back.

## Compact INT8 judgments over a live stream

All three judgment modes accept the explicit `safe-bpe-v1` input policy,
including serial, prompt-prefill, persistent-team and combined schedules:

```sh
fnlp candidate score-batch answers.ndjson --task judge \
  --defaults comparison.json --model ./model.fnlpq --memory-mib 8192 \
  --input-encoding safe-bpe-v1 --compute-threads 4 --prefill-rows 4
```

Byte fallback remains the default. Compact preparation preserves exact untrusted
criterion, candidate, document and claim bytes and excludes template-control
token IDs. Faithfulness still requires the complete source head to fit; all
evidence windows are encoded independently from their original UTF-8 spans.
Compact tokens do not replace source byte coordinates or authorize truncation.

Every record receives one complete-judgment tokenizer allowance:

| Option | Default |
| --- | ---: |
| `--max-tokenizer-input-bytes` | 1048576 |
| `--max-tokenizer-output-tokens` | 1048576 |
| `--max-tokenizer-pair-checks` | 4194304 |
| `--max-tokenizer-pair-bytes` | 67108864 |

The byte/token limits measure cumulative preparation work, independently of
`--max-input-bytes` for the original record text, `--max-line-bytes` for the
wire record and `--context-tokens` for a native head. Pairwise preparation
counts the criterion and both candidates; rubric preparation counts the document
and every criterion description. Faithfulness counts the claim and full source,
plus the entire source again across its evidence windows when the partition has
multiple windows. Limits never renew between heads, and refusal never falls
back to byte encoding.

A second ledger bounds the entire live invocation, including work spent on
rejected preparations:

| Option | Default |
| --- | ---: |
| `--max-total-tokenizer-input-bytes` | 67108864 |
| `--max-total-tokenizer-output-tokens` | 4194304 |
| `--max-total-tokenizer-pair-checks` | 268435456 |
| `--max-total-tokenizer-pair-bytes` | 4294967296 |

These totals do not renew at record or flush boundaries and do not grow with
`--max-requests`. A total may be smaller than the corresponding per-record
ceiling: preparation uses the remaining invocation allowance. When any work
axis has no invocation allowance left, further preparation is refused with a
terminal `WorkLimit`. An encoding refusal under a tightened remaining cap can
instead produce a recoverable planning rejection, as can a record's own limit.
Rejected preparation never restores work already spent. Earlier complete frames
remain delivered; cancellation remains terminal and never creates a successful
partial judgment.

The complete memory quote adds one derived encoder workspace and its invocation
ledger beside caller preparation, retained plans, stream buffers, native storage
and any selected scheduling workspace before private IO. The actual configured
planner supplies the encoding policy and template identity. Library callers use
`JudgeCorpusConfig::encoding_requirements` to validate that binding and quote
the workspace, and `NlpEngine::batch_int8_judge_scheduled` to supply explicit
invocation totals with optional prefill and team configuration.

Every compact candidate-framed event reports the exact `input_encoding` and
`input_encoding_work_limits`, including errors, flushes and `run_complete`.
Completed judgment results also retain the actual record's
`input_encoding_work` and selected policy. Byte-default output omits these
extensions. Changed prompt tokens do not establish identical model output,
tokenizer parity, task quality or measured speed.

Retained compact judgment recipes, invocation caps and resume requirements are
described in [candidate-scored-jobs.md](candidate-scored-jobs.md). The live
invocation ledger is not a persisted lifetime tokenizer-work receipt.

## Durable evaluation and resume

Retained jobs require `metadata-store` and `asupersync-runtime` on supported
Linux x86-64/AArch64 hosts, in addition to the explicit local artifact. Result
retention requires consent, an existing protected directory, a protected raw
32-byte key file, a random 128-bit job ID and immutable `JobLimits`.

```sh
fnlp candidate score-job start answers.ndjson --task judge \
  --defaults comparison.json --model ./model.fnlpq --memory-mib 8192 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key --limits ./job-limits.json

fnlp candidate score-job resume answers.ndjson --task judge \
  --defaults comparison.json --model ./model.fnlpq --memory-mib 8192 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key --limits ./job-limits.json \
  --materialize
```

`JOB_ID`, the key, directory and limits are caller-provisioned as documented in
`owned-jobs.md` and `candidate-scored-jobs.md`; neither command invents them.
Resume receives the COMPLETE original population, not just pending records.
The original secret, population, model identity, all defaults/policies and
scoring/planning limits authenticate before repair or native execution. Changed
criteria, comparison B, rubric weights, claims or thresholds are a different
job contract. Committed results are skipped; failed attempts retain their
lifetime work debit. There is no automatic retry or budget refund.

`--discard-uncommitted-tail` is an explicit resume-only repair choice, applied
by the common journal AFTER contract authentication. `--materialize` publishes
verified `materialized.ndjson` only after every item has committed. Standard
output is a metadata report, not the private judgments. Delivery failure can
follow durable progress and must not be interpreted as rollback.

## Resources and scheduling

All five native work ceilings cover the complete live invocation. Durable jobs
also retain the immutable lifetime `JobLimits.max_work` and attempt ceiling.
Independent criteria, comparison orders and evidence windows reuse one native
context; their work is summed, never discounted or renewed by a flush/record.

Preparation uses the invocation's real cancellation controller. Compact encoding
checks that controller during bounded preparation. Legacy byte tokenization is
checked around compilation, not preempted inside a tokenizer call. Output guards
survive complete-result serialization, external write and flush, or durable
spool synchronization and journal acknowledgement.

Both `score-batch --task judge` and `score-job start/resume --task judge` accept
`--prefill-rows 1..64` for explicit layer-major prompt processing and
`--compute-threads 1..64` for one persistent CPU team, including the coordinator.
The options can be combined; omitting both preserves serial execution. The
prompt schedule groups tokens within each prompt. It does not combine documents
or independent heads into one causal context. One selected team serves the
complete ordered invocation and physically joins before terminal publication.

Prompt scratch and fixed worker capacity are derived from native geometry and
reserved in the existing process ledger for the whole invocation. The complete
simultaneous memory commitment is checked before private IO. Neither schedule
enlarges any task or work ceiling or changes the sealed judgment identity.
A retained job freezes its prefill selection, including its absence, in the
private keyed recipe: repeat the original `--prefill-rows` selection on every
resume. Changed prefill selection refuses authentication. CPU-team width may
change between invocations within the admitted process-memory ceiling. Invalid
row/thread counts and insufficient combined memory refuse before input,
defaults, key or model IO; no unsupported schedule is ignored.

Reservations model admitted ownership, not measured RSS. Synthetic or pinned
planning fixtures are not native success, numerical equivalence, performance
or task-quality evidence. Rust compilation, regression execution, DSR and
full-model qualification have not been run for this implementation.

Compact INT8 regression definitions additionally cover complete-judgment work,
fixed invocation caps, exact additive memory, invalid flags before IO, actual
planner policy binding, optional provenance and all three modes across the four
schedules. They remain uncompiled and unexecuted source definitions.
