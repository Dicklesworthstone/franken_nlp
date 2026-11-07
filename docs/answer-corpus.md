# Complete-source question-answering corpora

`fnlp candidate source-int4 answer-corpus` answers a different question for each
complete original source in an ordered NDJSON stream. One resident model,
source compiler, execution policy and set of run ceilings serve the stream.
Long sources are partitioned with the actual question, task schema and prompt
capacity. This route requires `asupersync-runtime`; retained-job storage is not
required.

## Input and execution

Every document record supplies `id`, `text` and an object-valued `task_args`
containing exactly one `question` string:

```jsonl
{"id":"who","text":"Alice visited Shanghai. Bob stayed in Paris.","task_args":{"question":"Who visited Shanghai?"}}
{"flush":true}
{"id":"where","text":"Alice visited Shanghai. Bob stayed in Paris.","task_args":{"question":"Where did Bob stay?"}}
```

`text` is the complete original source. The question retains its exact Unicode,
whitespace and line endings after JSON decoding. Each source/question pair
receives a separate binding to all of its ordered native chunk identities.
Caller IDs are output metadata; they do not provide source or question authority.

Missing or null `task_args`, missing or duplicate `question`, unknown fields and
non-object argument values are refused. Empty or whitespace-only questions are
refused during preparation. Records cannot supply passages, change task options,
select another model or raise work, verification or memory ceilings.

Run against an independently pinned local snapshot:

```bash
fnlp candidate source-int4 answer-corpus questions.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 32768 \
  --max-requests 1000 \
  --max-input-bytes 1048576 \
  --max-question-bytes 16384
```

`-` or an omitted input path reads stdin. The explicit source-conversion route
uses `--source-dir` and `--source-manifest` instead of snapshot arguments. An
optional `--save-snapshot NEW_FILE` follows fixed options, model and complete
envelope admission. The snapshot remains if a later population record fails.
Every real source/question pair is planned when that record arrives; setup
does not substitute a sample source or question.

## Independent answers and explicit synthesis

The default preserves every independent chunk answer, model abstention and
whitespace-only range. Citations are independently verified and lifted to
original-document byte and Unicode-scalar coordinates. Different chunk answer
texts remain available; their exact string agreement is not confidence,
contradiction resolution or a global answer selection.

Add `--synthesize-answer` to produce one additional answer over **all collected,
verified verbatim source quotes**:

```bash
fnlp candidate source-int4 answer-corpus questions.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 32768 \
  --synthesize-answer \
  --max-synthesis-passages 32 \
  --max-synthesis-evidence-bytes 16384 \
  --cohort-chunks 4
```

All collected evidence must fit the declared passage and byte ceilings and the
actual question, canonical passage manifest and native context. These are
independent checks: a byte ceiling alone does not guarantee context fit. An
oversized evidence frontier fails the complete document without truncating it
to a top-k subset. Generated discovery answer prose never becomes factual
evidence. Original-coordinate citations remain attached to the final answer.

No collected evidence, a final model abstention and an answered result remain
distinct outcomes. An inference, verification or resource failure is never
converted into successful abstention.

`--options FILE` selects one complete `AnswerOptions` JSON object for the entire
stream. For example:

```json
{"max_answer_scalars":2048,"max_citations":8,"max_quote_scalars":256}
```

Omitting the file uses the shipped options. Unknown, duplicate or missing
option fields are refused. Options are read and validated before model loading
or population input; per-record questions do not replace them.

## Bounded answer hierarchies

Add `--hierarchical-answer` with `--synthesize-answer` to reduce an evidence
frontier through bounded quote-selection levels before the final answer:

```bash
fnlp candidate source-int4 answer-corpus questions.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 32768 \
  --synthesize-answer --hierarchical-answer \
  --answer-max-levels 8 --answer-max-passes 16 \
  --cohort-chunks 4 --cohort-hierarchy-groups 2
```

Each level plans its complete evidence frontier into groups that fit the actual
question, passage manifest and context. Each pass retains the record's exact
question. Only independently verified original-source quotations selected by
these passes enter the next level; generated answer prose remains output data.
Every final citation lifts to its original source coordinates.

This mode is explicitly lossy: intermediate selection may omit relevant
quotations. It does not promise equivalence to full-context inference or to a
single flat evidence pass. `--answer-max-levels` includes the final level, and
`--answer-max-passes` includes every additional native pass across all levels.
`--answer-max-tokenizer-calls` and `--answer-max-tokenizer-bytes` bound grouping
work across the complete document. Exhausted bounds, an oversized individual
quote, nonprogress, cancellation or verification failure yields no partial
successful document.

All permitted passes reserve their full work and masks **before discovery**.
The shared quote prices their question copies, identities and retained native
transcripts; the output guard keeps those transcripts charged through delivery.
The independent and flat-synthesis modes retain their existing result shapes.

## Native discovery and hierarchy cohorts

Without `--cohort-chunks`, discovery uses a serial native engine. Explicit
`--cohort-chunks N` shares decoder layers across independent discovery chunks.
N must be 1 through 64, no greater than `--max-chunks`, and fit the full native,
decoder and result quote. The selected width is not silently reduced.

Each row keeps the same original task KV, prompt, source evidence and finalizer.
Whitespace chunks retain their original ranges without fabricated native
answers or model work. All discovery windows drain before the evidence phase.
A flat synthesis uses one scalar pass. With hierarchy enabled,
`--cohort-hierarchy-groups N` independently shares native layers within each
level; its width must be 1 through 64 and no greater than the complete pass
ceiling. Hierarchy cohorts can run with serial discovery, and discovery cohorts
can run with serial hierarchy passes. Every complete level drains before the
next begins. The host admits the componentwise native peak across phases,
including the full continuation context, plus each simultaneous preparation
owner. Documents remain sequential and ordered.

The final semantic document keeps the existing independent-question,
question-synthesis or question-hierarchy wire shape. Public candidate provenance
records the answer mode, discovery schedule and optional hierarchy schedule;
physical scheduling does not replace the semantic result with a wrapper.

## Bounds and whole-run accounting

These input ceilings have separate meanings:

| Flag | What it bounds |
|---|---|
| `--max-input-bytes` | Decoded bytes of each complete original source |
| `--max-question-bytes` | Decoded bytes of each exact question |
| `--max-line-bytes` | Entire encoded record before LF, including JSON escaping and a trailing CR |
| `--max-total-input-bytes` | All input bytes, including blank and oversized records |
| `--max-requests` | Nonempty records, including malformed records and flush controls |
| `--max-map-result-bytes` | Complete semantic result for one original document |
| `--max-total-output-bytes` | All output bytes, including provenance and terminal frames |

Chunk planning never truncates a document to satisfy `--max-chunks`. The question,
schema, template, source encoding and reserved output all participate in actual
context admission. A question can fit its byte ceiling and still fail the native
prompt limit.

The five native work flags bound the **entire stream**:

- `--max-forward-positions`
- `--max-projected-logits`
- `--max-attention-pairs`
- `--max-dot-products`
- `--max-multiply-accumulates`

Every admitted attempt reserves its complete planned discovery work and, when
requested, the full future synthesis or hierarchy envelope before execution. Failed attempts
keep those debits. Increasing the permitted record count does not multiply the
native work ceilings.

Masks and the two independent verification pools also have run ceilings:

| Per-document allowance | Corresponding whole-run allowance |
|---|---|
| `--max-total-mask-node-visits` | `--max-run-mask-node-visits` |
| `--max-source-fields` | `--max-run-source-fields` |
| `--max-source-evidence-spans` | `--max-run-source-evidence-spans` |
| `--max-source-scan-steps` | `--max-run-source-scan-steps` |
| `--max-qa-citations` | `--max-run-qa-citations` |
| `--max-qa-evidence-spans` | `--max-run-qa-evidence-spans` |
| `--max-qa-scan-steps` | `--max-run-qa-scan-steps` |

The complete per-document verification allowances are reserved before an
attempt. Discovery, synthesis, hierarchy levels and citation lifting share the
same QA pool. Neither flush, a new
record, a shorter native window nor a failed attempt renews these counters.
`--max-checkpoints` and the hosted deadline likewise span the invocation.

CLI and hosted admission use one pure memory quote. It prices bounded transport
and ID storage, both preparation owners, the complete source, all simultaneously
retained question copies and chunk identities, map and evidence frontiers,
temporary native transcripts, native/cohort workspace and guarded semantic
output. The memory ceiling is a modeled reservation contract, not an operating
system RSS measurement.

## Completion and privacy

Every event carries `protocol: "fnlp-candidate-int4-answer-corpus-v1"`, public
model provenance, `source_mode: "complete_document"`,
`question_source: "per_record"` and the selected `answer_mode`. The existing
semantic event remains under `record`. Private question strings, options, input
paths and source text are not copied into provenance or error messages. Answer
text and source citations are intentional task output and may be sensitive.

Require **`run_complete` and exit status 0** for whole-run success. Earlier
completed records remain valid when a later record fails. Writer failure
poisons the stream; it does not trigger a replay or a second terminal frame.
An exact flush acknowledges the local writer and resets the transport ID epoch
after prior work drains. It does not reset work or establish durable retention.
The host retains reader, writer, model, compiler and memory ownership through
physical cleanup.

This workflow performs no retrieval and has no retained-job or automatic retry
mode. Structural citation membership does not certify semantic support,
completeness, calibrated answerability or full-document equivalence. Compiled,
model-present, performance and quality qualification remain subject to the
controller-selected project gates; authored fixtures are not execution evidence.

## Library entry points

Library callers use
`batch::source::grouped_int4::question::Int4QuestionDocumentBatchConfig`,
`Int4QuestionDocumentBatchPlanner`, and `NativeInt4QuestionDocumentBatch`.
The planner binds the actual model, source compiler, options, budgets and
physical schedules. `QuestionDocumentArgs` contains the required exact question.
`config.hierarchy` selects the bounded hierarchy; call `with_chunk_cohorts()`
and/or `with_hierarchy_cohorts(max_groups)` before preparing documents. A schedule
change invalidates earlier prepared ownership. The public memory limits and
hosted quote accept independent discovery and hierarchy group widths.

`NlpEngine::batch_int4_question_documents` accepts the factory, pinned vocabulary,
owned reader/writer, `CorpusLimits`, reduction reserve, kernel and cancellation
token. `hosted::int4_question_document_corpus_requirements` exposes the same pure
quote used by the command. Neither API requires metadata-store or invents a
durable population contract.

## Authenticated durable populations

For retained results and resumable complete source/question items, use
[`answer-document-job start|resume`](answer-document-jobs.md). That command
reuses this same question compiler, answer modes and physical schedules through
the owned job runner. It adds explicit result retention, authenticated lifetime
work, whole-population identity and optional retained-input replay. The live
`answer-corpus` command keeps its ordered-stream contract.
