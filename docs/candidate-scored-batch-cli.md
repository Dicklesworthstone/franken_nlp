# Resident candidate scoring corpora

`fnlp candidate score-batch` runs one finite-scoring task over a bounded NDJSON
stream using one resident INT8 model and one native engine. It uses the existing
candidate scorer, typed task finalizers, process resource host and ordered batch
transport. It does not generate labels as free text or retry failed parsing.
This is serial resident execution, not parallel/GEMM batching or a durable job.

The binary needs `asupersync-runtime`, an explicit local compatible candidate
artifact, and an explicit process memory ceiling. No model download, catalog
activation or publisher/fidelity/quality certification is performed.

## Classification with shared labels

```sh
fnlp candidate score-batch corpus.ndjson --task classify --defaults labels.json \
  --model ./model.fnlpq --memory-mib 8192 --max-requests 1000
```

Example `labels.json`:

```json
{
  "labels": [
    {"id": "complaint", "description": "A dissatisfied customer reporting a problem"},
    {"id": "question", "description": "A request for information"}
  ],
  "mode": "exclusive",
  "policy": {"minimum_candidate_weight_ppm": 0, "minimum_margin_ppm": 0}
}
```

`mode` and `policy` may be omitted to use the values shown. Exclusive mode needs
at least two labels. `multi_label` uses an independent yes/no head for each label,
not a probability distribution across labels. IDs and descriptions are bounded
exact data; they are not executable templates. The complete candidate language,
including terminal EOS, is scored with full-vocabulary denominators.

Input records use the existing transport:

```jsonl
{"id":"ticket-1","text":"The parcel arrived damaged. Please replace it."}
{"id":"ticket-2","text":"When will my order ship?"}
{"flush":true}
```

Without a defaults file, each classification record must supply a complete
`ClassificationBatchArgs` in `task_args`: `labels`, `mode`, `policy` and `budget`.
Such arguments replace defaults only for that record. Budgets can narrow, never
increase, the CLI task ceiling. Defaults files intentionally omit `budget` and
`document`; the CLI supplies the budget. No placeholder document stands in for
future corpus records during configuration validation.

## Independent-axis sentiment

```sh
fnlp candidate score-batch reviews.ndjson --task sentiment \
  --model ./model.fnlpq --memory-mib 8192
```

Omitted defaults select all four axes (valence, arousal, dominance, approach),
minimum peak weight 0 and maximum normalized entropy 1000000. An optional local
settings file may contain `axes` and `policy` with those named policy fields.
Every axis retains its complete five-bin candidate distribution, descriptive
moments and estimated/abstained decision. These are uncalibrated conditional
candidate weights, not correctness confidence or psychological measurements.

Sentiment record `task_args` contains only `axes` and `budget`. It cannot change
the run's policy, score space, EOS, tokenizer/template or model. The whole axis
bundle succeeds or fails together; a native failure is never neutral sentiment,
a successful abstention or a partial response. Per-record axes overrides do not
change subsequent defaults. Logical KV is cleared between independent heads and
documents, while the resident model and native allocations remain shared.

## Compute, memory and transport bounds

### Compact classification and sentiment records

The INT8 stream accepts `--input-encoding safe-bpe-v1` for `--task classify` and
`--task sentiment`:

```sh
fnlp candidate score-batch reviews.ndjson --task sentiment \
  --model ./model.fnlpq --memory-mib 8192 --input-encoding safe-bpe-v1
```

The default remains `byte-fallback`. Compact input preserves exact untrusted
bytes and excludes template-control token IDs. It can fit more ordinary source
text into the selected context; it does not truncate documents or imply the same
model output, quality or measured speed. INT8 `--task judge` rejects compact input
before defaults, source or model IO. Retained INT8 `score-job` commands continue
to use their existing byte-only recipes and do not accept these options.

Each record gets one bounded tokenizer ledger shared by its document and every
canonical label-metadata segment across all scoring heads. Sentiment reuses the
one encoded document for all requested axes. `--max-input-bytes` bounds the
cumulative untrusted bytes within a record, and context minus reserved candidate
depth bounds its cumulative untrusted output tokens. The per-record work caps
are `--max-tokenizer-pair-checks` (default 4194304) and
`--max-tokenizer-pair-bytes` (default 67108864). Exhaustion fails that record;
there is no per-head renewal or silent byte fallback. Native work remains one
whole-run allowance across all records and flushes.

Preparation admission includes the selected encoder's derived workspace beside
retained prompt and stream storage. The hosted INT8 classification and sentiment
APIs independently reserve that workspace from the explicit planner policy.
Library callers select it with `ClassificationPlanner::with_document_encoding`
or `SentimentPlanner::with_document_encoding` and use the existing hosted corpus
methods. Lower-level embeddings use the named
`Int8ClassificationBatchPlanner::new_with_document_encoding` or
`Int8SentimentBatchPlanner::new_with_document_encoding` constructors and must
admit the policy's workspace themselves. Legacy constructors remain byte-only.

The exact policy binds every prepared identity and appears as `input_encoding`
on every compact-stream event, including failures, flushes and an empty terminal
`run_complete`. Omitting the option preserves the prior wire shape. Source and
label content remains untrusted even when marker tokens are excluded.

### Whole-run bounds

The command reuses the single-request scored options for context, candidate
length, result size, memory and time. It exposes no generation, schema or mask
options. All FIVE native work flags are WHOLE-RUN totals:
`--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs`,
`--max-dot-products`, and `--max-multiply-accumulates`. Raising `--max-requests`
never multiplies compute authority. Defaults therefore may exhaust work before
1000 records; set explicit totals appropriate to the corpus. A larger forward
allowance can also require a larger modeled preparation reserve.

Independent heads reuse the largest live context rather than pretending their
summed forward work fits one continuous context. Whole KV capacity is admitted.
Neither early finishes, document failures nor flush epochs refund native work.
Classification preparation now borrows the actual corpus cancellation controller,
including its head compilation. Sentiment preparation and execution likewise use
one shared run controller. Cancellation remains cooperative; blocking IO and
in-progress bounded tokenizer operations are not forcibly preempted.

Transport defaults are 1000 nonempty records, 1 MiB per line and 1024 MiB each
for total input and output. `--max-input-bytes` bounds a document, while
`--max-input-mib` bounds the whole stream. Defaults JSON is capped at 1 MiB and
rejects duplicate keys, unknown fields and injected identities or documents.

Every record uses the existing `fnlp-candidate-batch-v1` provenance wrapper and
retains its exact canonical inner event. Provenance bytes count toward the wire
budget. Output guards stay held through write and flush; partial writes poison
transport without retries or appended terminal records. A completed stream with
any failed documents still exits nonzero. Prior complete frames are not retracted.

Owned input/output handles cross one actual hosted blocking invocation without
caller-thread stdio locks. Preparation, IO, weights, native workspace and output
use the existing process ledger; these are modeled commitments, not an OS RSS
limit or a measured memory claim. Without the runtime feature, execution refuses
before opening defaults/models or reading/writing the stream. Existing source
and extraction `candidate batch` commands are unchanged.

## Grouped INT4 scoring streams

The separate `fnlp candidate source-int4 score-batch` route supports
`--task classify|sentiment|judge` with one resident source-converted model or
independently digest-pinned INT4 snapshot. Judgment preserves complete pairwise,
rubric or faithfulness results. Defaults and replacement `task_args` retain the
existing strict schemas. This command accepts byte fallback and explicit
`--input-encoding safe-bpe-v1`; compact input keeps its own bounded per-record
tokenizer ledger.

Serial execution remains the default. `--compute-threads N` selects a
persistent output-row team, and `--prefill-rows M` selects bounded layer-major
processing within each prompt. These options can be combined: one team serves
all prompt morsels, candidate continuations, documents and flushes. Thread and
prompt-row counts are each bounded by 1 through 64. The separate
`--cohort-heads N` schedule groups simultaneous scoring heads from complete
documents and cannot combine with prompt morsels or a nonserial team. Invalid
row counts or incompatible schedules fail before defaults, input or model IO.

For example:

```sh
fnlp candidate source-int4 score-batch reviews.ndjson \
  --task sentiment \
  --snapshot ./model.int4.snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 16384 --group-width 128 \
  --prefill-rows 32 --compute-threads 4
```

This uses the existing strict INT4 prompt executor for every independently
prepared head. Longer prompts use finite morsels and the final short morsel
remains bounded. It does not reorder task prompts, combine documents into one
causal context, or retain private prefixes for later requests. Candidate trie
traversal, terminal EOS, full-vocabulary normalization and task finalization use
the existing scoring path. The native result markers and conservative work
reservations are unchanged by the physical row count.

The host admits one validated extra prompt workspace alongside resident weights,
the full live KV context, ordinary scratch, prepared inputs, optional encoder
workspace and complete guarded output. With a team, it additionally reserves
every worker's fixed input/output capacity for the declared prompt-row count.
The team and all these buffers are reused across heads, records and flushes;
increasing the population does not multiply them or grant more compute. A short
final prompt morsel does not reduce the declared capacity, and single-row
candidate continuations use the same team without growing its buffers. All five
`--max-*` native work flags still bound the entire stream. Cancellation uses the
original invocation controller. A failed head cannot publish a partial scored
bundle or turn a native failure into an abstention. Output guards continue
through ordered writes and flushes. The terminal `run_complete` is published
only after every worker physically joins, with the original cancellation and
checkpoint allowance still authoritative.

All events, including errors, flushes and an empty `run_complete`, use the
`fnlp-candidate-int4-score-batch-v1` candidate wrapper. An explicit prompt
schedule adds `prefill_schedule` with:

| Field | Meaning |
| --- | --- |
| `execution` | `grouped-int4-layer-major-causal-prefill-v1` |
| `max_batch_rows` | The selected bounded prompt row count |
| `admitted_workspace_bytes` | The exact validated additional prompt payload reservation |

Omitting `--prefill-rows` omits this field and preserves the default wire shape.
When a CPU team is selected, the existing `compute_schedule` appears alongside
`prefill_schedule`; the two declarations retain their independent meanings.
The metadata describes selected physical configuration; it is not measured
execution, parity or speed evidence. Complete semantic records retain their
original contents. Require the terminal `run_complete` and exit status zero for
whole-stream success.

Library callers use
`NlpEngine::batch_int4_scored_layer_major(model, planner, limits, kernel, prefill, reader, writer, cancellation)`
with the existing sealed `Int4ScoredBatchPlanner` for any of the three tasks.
To share a persistent team, use
`NlpEngine::batch_int4_scored_layer_major_with_threads(model, planner, limits, kernel, compute_threads, prefill, reader, writer, cancellation)`.
The lower-level `NativeInt4ScoredBatch::new_layer_major` requires its embedding
caller to admit the additional validated workspace. Existing serial and CPU-team
constructors preserve their behavior.

Regression source definitions cover exact combined scratch quotes, invalid
schedules and memory refusal before IO, per-record tokenizer coexistence,
metadata compatibility, and mixed prompt-batch/single-row worker lifetimes with
post-join publication under the original quota. Controller-only full-model
fixtures compare serial, team, prefill and combined schedules across all three
tasks, both encodings and flush epochs. These definitions have not been
executed in this editing session. Kernel parity, model quality and target-host
performance remain separate qualification requirements.

## Evidence scope

Every emitted event retains `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`. Added parser, resource and real pinned planning test
sources do not provide neural-success fixtures. Rust compilation/tests, model
runs, numerical parity, quality/performance and controller DSR qualification were
not executed in this implementation session. The native backend and artifact
retain their existing candidate evidence status.
