# Durable complete-source question jobs

`fnlp candidate source-int4 answer-document-job start|resume` runs complete
original documents with a distinct private question per record through the
existing authenticated job system. It reuses the same source/question planner,
native executor and independent citation finalizers as
[answer-corpus](answer-corpus.md).

The route requires `metadata-store` and `asupersync-runtime` on Linux x86-64 or
AArch64. These are implemented candidate interfaces. Compilation, Rust
regressions, actual-model execution, quality and performance qualification
remain pending the controller-selected DSR and model gates.

## Population and fixed policy

Each record has exactly this shape:

```jsonl
{"id":"who","text":"Alice visited Shanghai. Bob stayed in Paris.","task_args":{"question":"Who visited Shanghai?"}}
{"id":"where","text":"Alice visited Shanghai. Bob stayed in Paris.","task_args":{"question":"Where did Bob stay?"}}
```

`text` contains the complete original source, with no normalization or prefix
truncation. `task_args` is an object containing only the required exact question.
Missing, null, duplicate or unknown fields and array-valued arguments reject.
Questions cannot be empty or whitespace-only. A record cannot replace options,
passages, task policy, model, budgets or either native schedule.

The complete population is frozen before execution. IDs must be unique across
that entire population. Flush controls are not accepted. Ordinary resume
requires the same complete original encoded population, including question
bytes and JSON framing; rearranging or reserializing records changes the
authenticated input. Explicit retained-input replay is described below.

`--options FILE` selects one complete `AnswerOptions` object for every record
and resume. Omission uses shipped defaults. For example:

```json
{"max_answer_scalars":2048,"max_citations":8,"max_quote_scalars":256}
```

Options are validated before loading the model or opening the population.
Private source/question contents and options are absent from public provenance
and error messages; intentional answer and citation output may contain them.

## Starting and resuming

Use the existing job initialization/management workflow to obtain a protected
job directory, random job ID and owner-only key file. The command does not
create or silently adopt a directory or generate a key. Result retention
requires explicit consent:

```bash
fnlp candidate source-int4 answer-document-job start questions.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 32768 \
  --store-results --job-dir "$JOB_DIR" --job-id "$JOB_ID" \
  --key-file "$JOB_KEY_FILE" --limits qa-job-limits.json \
  --max-input-bytes 1048576 --max-question-bytes 16384 \
  --materialize
```

The source-conversion route uses `--source-dir` and `--source-manifest`
instead of snapshot arguments. An optional `--save-snapshot NEW_FILE` follows
fixed policy, model and complete-envelope admission; that snapshot remains if
later job execution fails.

Resume with the same task, model, mapping, options, schedules and immutable
limits:

```bash
fnlp candidate source-int4 answer-document-job resume questions.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 32768 \
  --store-results --job-dir "$JOB_DIR" --job-id "$JOB_ID" \
  --key-file "$JOB_KEY_FILE" --limits qa-job-limits.json \
  --max-input-bytes 1048576 --max-question-bytes 16384 \
  --materialize
```

Resume authenticates the actual model, exact complete population and fixed
recipe before repairing a tail or executing pending items. An uncommitted
spool tail is refused unless resume explicitly includes
`--discard-uncommitted-tail`; authentication still precedes that repair.
Committed corruption is not an uncommitted tail.

The exact question and source are bound through the job's keyed input
commitments. The private recipe binds the answer options, planning and map
configuration, synthesis and hierarchy limits, discovery/hierarchy schedules,
result and token ceilings, and lifetime allowances. It stores no default
question because every item owns its question. A changed question cannot reuse
a completed answer under the previous population.

## Three answer modes

| Mode | Selection | Result |
|---|---|---|
| Independent answers | Default | All independent chunk answers, abstentions and whitespace ranges, with original-coordinate citations |
| Flat evidence synthesis | `--synthesize-answer` | One additional native answer over all collected verified original-source quotations |
| Hierarchy | `--synthesize-answer --hierarchical-answer` | Bounded quote-selection levels followed by the final answer, with every pass retaining the original question |

Flat synthesis refuses a complete evidence frontier that exceeds its passage,
byte or actual context capacity; it never silently truncates to a subset.
Hierarchy is explicitly lossy: intermediate answers may select only part of
the original evidence. Generated answer prose never becomes factual evidence
for the next pass. Structural citation validity does not prove entailment,
recall, calibrated answerability or full-context equivalence.

Hierarchy options are shared with the live corpus:

```text
--answer-max-levels N
--answer-max-passes N
--answer-max-tokenizer-calls N
--answer-max-tokenizer-bytes N
```

The pass cap covers every additional native pass, including the final answer.
Every permitted pass reserves all five native work axes and its grammar-mask
allowance before discovery begins. Exhausted levels, passes, grouping work,
verification or context, an oversized quote, nonprogress or cancellation
prevents a partial successful document.

`--cohort-chunks N` shares native decoder layers across independent discovery
chunks. `--cohort-hierarchy-groups N` independently shares them within one
hierarchy level. Both widths are explicit, 1 through 64, bounded by the
corresponding chunk/pass cap and admitted without silent resizing. Every
window and level drains before the next dependent phase. Serial, discovery
cohorts, hierarchy cohorts with serial discovery, and both schedules use the
same underlying native task contracts. Repeat the selected schedules on resume.

`--compute-threads 2..=64` is a separate physical schedule: one persistent
INT4 output-row team spans the complete invocation, every pending source/question
pair, discovery map and permitted synthesis or hierarchy pass. Width `1`
retains serial execution. Team selection rejects chunk and hierarchy-group
cohorts before population or storage access. The shared pre-IO quote adds one
worker-scratch reservation without multiplying task KV or changing work caps.

The compute width is not part of the authenticated semantic recipe. A later
explicit resume can select a different compatible width while retaining the
exact input/options/model binding, stored result bytes and cumulative native,
mask and attempt debits. Compact input keeps its existing complete-document
scope and supported modes; neural synthesis and hierarchies remain byte-input.
The invocation still drains workers before returning, including after a
durable commit whose completion acknowledgement is lost to cancellation.

## Lifetime work and complete memory admission

The shared `JobLimits` JSON is the sole lifetime authority for item count,
attempts, all five native work axes, masks, encoded input, snapshot and stored
output sizes. It uses the existing [owned-job format](owned-jobs.md); this
command introduces no new journal or spool format.

The command's five work flags and `--max-total-mask-node-visits` bound one
complete document across discovery and every continuation. The job file's
`max_work` bounds all attempts over the entire lifetime. A failed attempt
keeps its exact complete prepared work debit. Restart, shorter physical
windows, output abstention and changing invocation transport limits do not
refund or replenish it.

Both verification pools have three axes: fields, occurrence spans and scan
steps. Because these are not journal counters, the recipe derives each
lifetime ceiling as the corresponding per-document ceiling multiplied by the
immutable total `max_attempts`, using checked arithmetic. The durable adapter
verifies this relationship before execution. Overflow rejects. Discovery,
evidence selection and final citation lifting share the same per-document QA
verification pool.

Input bounds remain distinct:

| Bound | Authority |
|---|---|
| `--max-input-bytes` | Outer admission ceiling that must cover the stored per-item cap |
| `JobLimits.max_input_bytes_per_item` | Entire encoded record before LF, including source, question, escaping, IDs and framing |
| `--max-question-bytes` | Exact decoded question ceiling |
| Task/context limits | Actual question, source, template, schema, manifest and reserved output token capacity |
| `--max-stream-mib` and `--max-input-lines` | Per-invocation transport ceilings; these do not renew lifetime work |

The decoded source and question are each bounded, and their actual combined
length is checked again before planning. Their independent maxima need not
both fit in the same stored record: a large source with a short question and
a short source with a long question can share one fixed policy. The encoded
record ceiling still applies to every real item.

Before population IO, CLI and host use the same pure memory quote. It includes
the complete authenticated population, journal/serialization declarations,
all source and question representations, frozen/reconstructed identities,
complete evidence frontier, simultaneous hierarchy preparation, the native
phase peak, and retained pass token vectors through durable acknowledgement.
Each task retains its original single-context KV budget; a wider cohort does
not enlarge that task's context. The quote is a modeled allocation contract,
not a measured RSS guarantee.

## Explicit retained-input replay

Result retention and input retention are separate choices. To retain the exact
private input population at creation, add:

```text
--spool-input --max-retained-input-mib N
```

To resume from that authenticated input spool, omit the population positional
argument and use:

```text
--use-retained-input --max-retained-input-mib N
```

The resume path does not open a population file or read supplied stdin.
Using the wrong operation, supplying a population alongside retained-input
selection, omitting the retention cap, or combining both modes rejects before
IO. Exact current model, policy and lifetime limits remain required.
Retaining input is explicit storage of private source and question bytes;
ordinary start/resume does not gain that consent. See
[retained-input replay](retained-inputs.md) for framing, verification and
recovery semantics.

## Completion and library API

One complete source/question result is one durable item. Chunks and hierarchy
passes do not commit separately. Successful results remain guarded through
canonical serialization, result-spool synchronization and journal
acknowledgement. `--materialize` publishes only after every item has committed.
It does not replace an existing materialized file.

Stdout reports metadata under
`fnlp-candidate-int4-answer-document-job-v1`, including the answer mode and
selected schedules. A nonzero return can follow durable progress; inspect or
authenticate resume rather than assuming rollback. Cancellation retains its
cause, and a publication deadline does not become successful abstention.

Library callers select schedules on
`Int4QuestionDocumentBatchPlanner` before wrapping it in
`Int4QuestionDocumentJobPlanner::new`. The thin
`Int4QuestionDocumentJobProcessor` implements the existing
`DurableBatchProcessor` contract. Use
`NlpEngine::job_int4_question_documents` with the pinned vocabulary,
`SourceJobRequest`, `JobHostLimits`, reduction reserve, kernel and an owned
reader. The public `hosted::int4_question_document_job_requirements` exposes
the same pure quote; explicit retained-input modes use the shared
`JobOpenMode` variants.

`NlpEngine::job_int4_question_documents_with_threads` adds an explicit team
width and requires a compatible process thread envelope. The dispatcher owns
the additional worker scratch until physical join. CLI progress metadata
adds `compute_schedule` for this route, while durable semantic results and
serial report shapes remain unchanged. Thread selection is not a measured
performance claim.

Regression definitions cover real owned-storage replay with explicitly
synthetic transport output, strict question contracts, lifetime accounting,
source/question mismatch, corruption, memory quotations, cancellation and
report publication. Those definitions are not evidence of native inference
quality or performance, and remain UNRUN pending the project gates.
