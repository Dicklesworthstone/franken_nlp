# Resident candidate classification and sentiment corpora

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

## Evidence scope

Every emitted event retains `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`. Added parser, resource and real pinned planning test
sources do not provide neural-success fixtures. Rust compilation/tests, model
runs, numerical parity, quality/performance and controller DSR qualification were
not executed in this implementation session. The native backend and artifact
retain their existing candidate evidence status.
