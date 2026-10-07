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

Preparation uses the invocation's real cancellation controller. Bounded legacy
tokenization is checked around compilation, not preempted inside a tokenizer
call. Output guards survive complete-result serialization, external write and
flush, or durable spool synchronization and journal acknowledgement.

Bulk judgment commands accept `--prefill-rows 1..64` for explicit layer-major
prompt processing. Omit the option to retain serial execution. Both
`score-batch --task judge` and `score-job start/resume --task judge` use the
same bounded schedule as the hosted `batch_int8_judge_layer_major` and
`job_int8_judge_layer_major` methods. This groups tokens within each prompt;
it does not batch documents or execute independent heads simultaneously.

Extra scratch is derived from native geometry and reserved in the existing
process ledger for the whole invocation. The schedule does not enlarge any
task or work ceiling. A retained job freezes its effective row schedule in
the private keyed recipe: use the original option (or its original absence)
on every resume. Changed schedules refuse authentication, rather than
replaying committed work with different settings. Invalid row counts and
classification/sentiment bulk requests with this option refuse before
input, defaults, key or model IO; no unsupported schedule is ignored.

Reservations model admitted ownership, not measured RSS. Synthetic or pinned
planning fixtures are not native success, numerical equivalence, performance
or task-quality evidence. Rust compilation, regression execution, DSR and
full-model qualification have not been run for this implementation.
