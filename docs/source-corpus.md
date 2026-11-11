# Resident complete-source task corpora

`fnlp candidate source-int4 source-corpus` applies the existing complete-document
NER, keyphrase, summary and fixed-question pipelines to an ordered NDJSON
stream. One model stays resident for the invocation. Every input record owns
one complete original source and receives one complete semantic document
result, after all of its discovery and selected reduction passes finish.

The CLI requires `asupersync-runtime`. The live core and host have no
`metadata-store` dependency. Retained `document-job` workflows continue to
use their existing authenticated recipes and storage protocol.

## Commands and task modes

| Command | Default result | Explicit extensions |
|---|---|---|
| `source-corpus ner` or `ner-corpus` | Independent entity results for every chunk, with original-source evidence | `--cohort-chunks N` |
| `source-corpus keyphrases` or `keyphrases-corpus` | Independent chunk keyphrases | `--reduce-keyphrases` retains the exact candidate union and applies final global ranking once |
| `source-corpus summarize` or `summarize-corpus` | Deterministic exact cited-bullet union | `--synthesize-summary`; optionally `--hierarchical-summary` |
| `source-corpus answer --question FILE` | Independent answers to one exact fixed question throughout every source | `--synthesize-answer`; optionally `--hierarchical-answer` |

The existing [`answer-corpus`](answer-corpus.md) route accepts a separate
question in each record. It remains available with its own input contract.
Use `source-corpus answer` when one question is fixed across the population,
including when explicit hierarchical answer synthesis is required.

All four fixed-task commands accept `--options FILE` for the selected task's
complete per-chunk typed options. Exact reducers and synthesis settings use
the same flags and semantics as the corresponding single-document map
commands. No source-task behavior is reimplemented in the CLI.

## Input and examples

A fixed-task population contains complete original UTF-8 sources:

```json
{"id":"report-1","text":"The complete first original document."}
{"id":"report-2","text":"The complete second original document.","task_args":{}}
{"flush":true}
{"id":"report-3","text":"The next complete original document.","task_args":null}
```

`task_args` may only be absent, `null`, or an empty object. Per-row options,
questions, schemas, budgets and schedule overrides are refused. IDs are
unique within the bounded current epoch. A completed flush releases the
epoch's ID bookkeeping; it does not renew execution allowances.

The examples use an independently obtained model-content pin and an
operator-selected memory ceiling. The ceiling must admit the declared
context, maximum chunk window, complete evidence frontiers and every
permitted synthesis pass.

Extract globally ranked keyphrases across complete sources:

```sh
fnlp candidate source-int4 keyphrases-corpus documents.ndjson \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib "$MEMORY_MIB" --group-width 128 \
  --reduce-keyphrases --document-keyphrases 12 \
  --cohort-chunks 4
```

Run explicit hierarchical summary synthesis:

```sh
fnlp candidate source-int4 summarize-corpus documents.ndjson \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib "$MEMORY_MIB" --group-width 128 \
  --synthesize-summary --hierarchical-summary \
  --cohort-chunks 4 --cohort-hierarchy-groups 2
```

Ask one exact private question of every complete source:

```sh
fnlp candidate source-int4 source-corpus answer documents.ndjson \
  --question question.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib "$MEMORY_MIB" --group-width 128 \
  --synthesize-answer --hierarchical-answer \
  --cohort-chunks 4 --cohort-hierarchy-groups 2
```

Source conversion also works: replace the snapshot and pin flags with
`--source-dir DIR --source-manifest FILE`. Optional `--save-snapshot NEW_FILE`
runs only after the fixed task, actual-model factory and maximum document
envelope have been admitted. The saved model may remain after a later
document fails; the stream is not collected or pre-read to create the snapshot.

## Complete-document semantics

Original bytes are losslessly partitioned using the real pinned tokenizer
and exact typed task/template capacity. Before a document's first native
forward, the complete original discovery frontier is planned, its identities
are frozen, and its native/resource requirements are checked. Execution
reconstructs the plan and compares the sealed source, ordered identities and
work receipt.

NER and raw keyphrases preserve independent chunk outputs. Ranked keyphrases
merge the complete exact candidate/evidence union before final selection.
Default summary reduction preserves every candidate through intermediate
reduction and applies the final bullet cap once at the root.

Flat summary or QA synthesis consumes all independently verified original
source quotes from discovery. It refuses when that complete evidence cannot
fit the final context. Explicit hierarchies partition evidence into bounded
context-sized groups and pass only verified selected original quotes into
later levels. Generated answer or bullet prose never becomes factual source
evidence.

Hierarchies are explicitly lossy. Exhausted pass/level/grouping limits,
oversized atomic quotes, nonprogress, native errors or failed verification
produce no successful partial document. Citation membership establishes
original-source lineage; it does not establish semantic entailment, recall,
contradiction resolution or equivalence to full-context inference.

## Native schedules

Documents execute in input order. `--cohort-chunks N` shares existing native
decoder layers across independent discovery chunks within a document.
`--cohort-hierarchy-groups N` independently shares them across independent
groups in an explicitly enabled hierarchy level. Discovery remains serial
unless its own cohort flag is supplied.

`--compute-threads 2..=64` selects one persistent INT4 output-row team for the
whole bounded invocation. The coordinator and workers serve every document,
map, exact reducer and selected neural synthesis or hierarchy pass, including
across flush boundaries. Documents still finish in input order. Width `1`
keeps the existing serial route. This physical schedule preserves each row's
arithmetic, semantic identity and cumulative work limits; it cannot combine
with chunk cohorts or hierarchy-group cohorts.

For example, add `--compute-threads 4` to a corpus command without either
cohort flag. Compact `safe-bpe-v1` input can use the team on its existing
supported map/reducer modes. Neural synthesis and hierarchies retain their
existing byte-input restriction. One additional worker-workspace reservation
is admitted before private IO and stays charged through physical worker join;
the native KV/context and complete evidence-frontier charges remain separate.

Each row keeps its own prompt, grammar, source language, causal state and
typed finalizer. The complete current level finishes before the next level
starts. Chunks and hierarchy groups do not become partial results or durable
checkpoints. Neither flag increases task authority or work allowances, and
neither changes the semantic document envelope.

The host reserves the complete declared native phase peak and decoder
workspace before execution. It uses the existing native kernels and closed
serial/cohort/team backends; no caller-supplied executor or automatic schedule
selection participates.

## Nonrenewable work and memory

The five native flags bound the **entire stream**, including discovery and
all reserved synthesis passes:

- `--max-forward-positions`
- `--max-projected-logits`
- `--max-attention-pairs`
- `--max-dot-products`
- `--max-multiply-accumulates`

The existing `--max-total-mask-node-visits` bounds each complete document.
`--max-run-mask-node-visits` additionally bounds the whole invocation.
Independent `--max-run-source-fields`,
`--max-run-source-evidence-spans` and `--max-run-source-scan-steps` cover
original-source verification. `--max-run-summary-scan-steps` and
`--max-run-keyphrase-scan-work` cover the separate deterministic reducers.

These allowances are committed atomically before an execution attempt.
Independent verification and reducer accounting conservatively reserves the
selected per-document cap. A later failure retains its debit. More
`--max-requests`, a larger cohort, or a flush does not fund more native or
verification work; exhausted allowances refuse subsequent execution.

A pure quote shared by the host includes stream/JSON/ID buffers, fixed
question or option allocation capacity, complete source storage, frozen
chunk identities, all map/reducer/synthesis frontiers, simultaneous hierarchy
plans, the native phase peak and output staging. Static CLI bounds precede
private configuration IO. The exact typed-task quote is checked before model
acquisition. The resident model and CLI preparation lifetime are admitted
alongside the hosted invocation. These are explicit resource-ledger
reservations, not a measured operating-system RSS guarantee.

`--max-input-bytes` bounds each decoded source; `--max-line-bytes` also
accounts for JSON escaping and framing. Total transport, native work,
checkpoint and elapsed-time ceilings are independent. One elapsed deadline
spans configuration reads, model acquisition, optional persistence, corpus
execution and final publication. Blocking IO observes it cooperatively.

## Results and library use

Every emitted frame includes
`protocol: "fnlp-candidate-int4-source-corpus-v1"`, public actual-model
provenance, the selected document mode and declared native schedules. The
original semantic event remains under `record`. Private options, the fixed
question and filesystem paths are not added to provenance.

A complete run requires **both `run_complete` and exit status zero**. A
drained protocol can contain rejected records. Earlier completed documents
remain valid if a later record or terminal write fails. Guarded output,
reader/writer ownership and native reservations survive delivery and physical
cleanup. There is no retained job or implicit input spool.

Explicit teams add `compute_schedule` to public provenance, with the selected
schedule and declared width. The semantic `record` stays unchanged, and serial
events omit this field. Terminal `run_complete` is deferred until every worker
has joined; cancellation or a failed terminal write still makes the invocation
fail. No measured speed or projection-count claim follows from selecting a team.

Library callers use:

- `batch::source::grouped_int4::document::Int4SourceDocumentBatchPlanner`
  and `Int4SourceDocumentTask` for the immutable actual-model-bound factory.
- `Int4SourceDocumentBatchLimits` for independent invocation-wide ceilings.
- `NativeInt4SourceDocumentBatch` for the closed native processor.
- `hosted::int4_source_document_corpus_requirements` for the pure complete
  memory quote.
- `NlpEngine::batch_int4_source_documents` for one owned resident stream.
- `NlpEngine::batch_int4_source_documents_with_threads` for the same stream
  with a compatible installed process thread envelope and one persistent team.

The durable source-document runner reuses the same task representation,
native planning, receipt verification, execution backend, workspace formulas
and output admission. Existing durable recipe bytes and public prepared
`JobWork` remain unchanged.

## Evidence status

Core, host, parser and runtime regression definitions cover mode selection,
fixed private configuration, original-source seals, all lifetime axes,
snapshot/source equivalence, phase peaks, malformed arguments, unsupported
strategy refusal and output ownership. Source review was performed, and the
edits were recovered through GitHub after the workspace runner disconnected.

Compilation, executable Rust tests, native parity, task quality and target-host
performance remain **UNRUN** pending the repository's controller-selected
immutable-SHA DSR production gate. Selected-row projection remains limited to
its explicitly supported one-off and serial extraction-batch routes.
