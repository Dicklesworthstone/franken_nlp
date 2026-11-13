# Durable native classification and sentiment jobs

`fnlp candidate score-job start` and `resume` connect the existing finite-score
engines to the authenticated owned-job journal. This closes the gap between a
live `score-batch` pipe and a corpus that can survive an interrupted invocation
without rerunning committed records. It does not add a second scorer, runtime,
model or generated-label shortcut.

The command requires `asupersync-runtime` and `metadata-store` on Linux x86-64 or
AArch64, and an explicitly selected compatible local candidate INT8 `.fnlpq`.
Other builds refuse before opening a key, defaults file, input or model. The
candidate remains non-certified and non-authoritative. Scoring is full-vocabulary
candidate/EOS scoring, not calibrated correctness confidence.

## Start and resume

Supply a fresh random 128-bit job ID as 32 lowercase hex characters, an existing
protected owner-only directory, and a separately protected regular file containing
exactly 32 RAW random secret bytes. A text/hex encoding of the secret is not the
key-file format. The command does not create a directory, generate a key, save
original input, silently adopt existing storage or retry automatically.

The examples below assume `JOB_ID` is that generated ID, `protected-job` is the
prepared directory, `protected.key` is the raw key, and the settings/limits files
have been prepared. Do not use a fixed example ID or key for real jobs.

```sh
fnlp candidate score-job start corpus.ndjson \
  --task classify --defaults labels.json \
  --model ./model.fnlpq --memory-mib 8192 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key \
  --limits ./job-limits.json --materialize
```

After interruption, repeat the SAME invocation with `resume` instead of `start`:

```sh
fnlp candidate score-job resume corpus.ndjson \
  --task classify --defaults labels.json \
  --model ./model.fnlpq --memory-mib 8192 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key \
  --limits ./job-limits.json --materialize
```

Resume requires the COMPLETE original ordered population, not just pending
records. It authenticates the original identity, recipe, population, limits and
committed result frames. Already committed records are read from the protected
spool, never regenerated to reconstruct output. Failed or interrupted attempts
remain charged; a new processor cannot reset lifetime work or attempt ceilings.

By default an uncommitted tail or reserved output stage causes refusal. Only
`resume --discard-uncommitted-tail` authorizes its repair, and only after the
original contract and committed frames authenticate. It does not relax mismatch
checks, overwrite committed records or refund work. Report delivery failure may
follow durable progress: do not infer rollback or blindly restart as a new job.

## Exact input and shared scoring settings

Input is bounded UTF-8 NDJSON with unique IDs across the whole population:

```json
{"id":"doc-1","text":"The agreement was signed on Friday."}
{"id":"doc-2","text":"The team announced a product update."}
```

Omit the input path or use `-` for stdin. Preserve the original input separately
for resume. Blank lines count toward the transport limit. Live-batch flush
commands are not durable population records. The existing bounded in-memory
snapshot/manifest design remains in use; this is not a disk-backed input index
or an unbounded corpus stream.

Classification defaults use exactly the same parser as `candidate score-batch`:

```json
{
  "labels": [
    {"id":"legal","description":"Contracts and legal agreements"},
    {"id":"product","description":"Product releases and updates"}
  ],
  "mode":"multi_label",
  "policy":{"minimum_candidate_weight_ppm":0,"minimum_margin_ppm":0}
}
```

`exclusive` computes one complete exclusive-label decision; `multi_label`
computes independent binary decisions for every label. There is no softmax over
independent label heads. All labels, descriptions, ordering, mode and policy are
part of the frozen private recipe. A changed default description refuses resume
even when every document is byte-identical. Omitting classification defaults is
allowed only when records supply the complete typed classification `task_args`.

Use `--task sentiment` for independent affect dimensions. Omitting defaults uses
the existing full axis set and existing uncalibrated policy. A sentiment defaults
file may specify `axes` and `policy`, with the same schema as `score-batch`.
Its score mode, EOS, anchors and policy are bound by the ACTUAL pinned planner's
template digest, not a caller-provided second description. One complete axis
bundle is one durable item; a failed axis never becomes a committed partial result.

Both settings formats reject duplicate keys, unknown fields and injected
`document`, `budget` or model/identity settings. Per-record `task_args` uses the
existing typed batch schema and is part of the exact original population; it may
narrow admitted task limits but cannot increase them. Overrides do not mutate
later records or the defaults. Labels, descriptions, documents and resulting
scores remain private data, not telemetry.

## Limits are part of the replay contract

`--limits` is the existing complete `JobLimits` JSON. A small-corpus example is:

```json
{
  "max_items":100,
  "max_id_bytes":128,
  "max_input_bytes_per_item":65536,
  "max_snapshot_bytes":1048576,
  "max_result_bytes":1048576,
  "max_spool_bytes":16777216,
  "max_materialized_bytes":16777216,
  "max_journal_bytes":1048576,
  "max_attempts":200,
  "max_work":{
    "model":{
      "forward_positions":131072,
      "projected_logits":100000000,
      "attention_pairs":100000000000,
      "projections":{
        "dot_products":100000000000,
        "multiply_accumulates":1000000000000000
      }
    },
    "mask_node_visits":0
  }
}
```

These are illustrative ceilings, not measured requirements or a promise that
100 documents will fit. Actual document lengths, candidate sets and head counts
consume real work. Choose limits for the intended corpus BEFORE starting.
Scoring has no grammar masks, so a zero mask allowance is valid; all five native
counters are still reserved durably before any native execution callback.

The scoring CLI exposes `--max-candidate-tokens` rather than generation length,
plus the existing `--max-forward-positions`, `--max-projected-logits`,
`--max-attention-pairs`, `--max-dot-products` and `--max-multiply-accumulates`.
These native adapter ceilings, candidate limits and task/planning ceilings are
also frozen in the recipe; changing them on resume is not a way to extend the
job. Lifetime `JobLimits` additionally include all previously charged attempts.
Increasing `--max-input-lines` or `--max-stream-mib` does not multiply work
allowances. Seed, generation, free-form instruction, schema and grammar-mask
flags are not accepted by `score-job`.

`--max-result-bytes` is the complete native task ceiling and must fit immutable
`max_result_bytes` in `JobLimits`, even when some records request smaller outputs.
`--max-input-bytes` bounds task planning; `max_input_bytes_per_item` separately
bounds the complete original NDJSON record including its arguments. Shared
per-record settings still carry real budgets. There is no silent truncation.

## Persistence and process ownership

Stdout contains a non-authoritative candidate provenance wrapper and metadata:
task, operation, job ID, committed/item/attempt counts, reserved work, spool size
and materialization status. It does NOT stream IDs, labels, documents or scores.
With `--materialize`, the existing verified ordered publication produces
`materialized.ndjson` only after all results commit. That file and the spool
contain private native output, not anonymized data. Existing stored-job status,
verification and materialization APIs remain available without model inference.

The host reserves the complete bounded input population, journal allocations,
configuration/graph preparation, serialization staging, full resident KV and
scratch storage. The real process admission authority supplies each output guard,
which remains alive through spool sync and journal acknowledgement. Owned readers
and configuration survive the blocking invocation; native state drops before its
resource guards. Reservations are modeled ledger commitments, not measured or
OS-enforced RSS bounds. Blocking I/O and bounded compiler calls are cooperative,
not preempted mid-operation.

The CLI validates settings, immutable result ceilings and a necessary population
memory floor before weight loading. It then loads the selected model once. Full
population ingestion and replay authentication occur inside the hosted invocation
after weights are resident and before job-file repair or the first native forward.
This path does not claim pre-weight authentication of the complete corpus.

For embedding, the concrete entry points are `NlpEngine::job_int8_classify` and
`NlpEngine::job_int8_sentiment`, taking the existing resident model, pinned planner
Arc, corpus configuration, `SourceJobRequest`, `JobHostLimits`, owned reader and
cancellation token. The durable adapters reuse `JobRunner` and existing native
batch processors; no public fake-native or replacement-admission hook is added
to these hosted entry points. Existing `candidate job` source/extraction and
`candidate score-batch` live streaming semantics are unchanged.

## Validation scope

Twenty regression definitions were added: eight core recipe/planning tests, two
host admission/capture tests, six CLI tests (including a feature-dependent no-I/O
case), and four real pinned-planner/runtime-configuration tests. They cover exact
private defaults, 41 scorer/work-limit mutations, classification heads, sentiment
policy/axes, cancellation, typed defaults, mode isolation, full KV/output ceilings
and metadata-only reporting. They do not inject fake neural successes.

This editing session reviewed source APIs, ownership/cancellation paths and the
committed diffs. No Rust compilation, Cargo tests, repository execution scripts,
model runs, numerical qualification or performance gates were run; executable
validation remains with the designated controller. No passing-build, fault-recovery
execution, task-quality, calibration or throughput result is claimed here.

## Compact input for grouped INT4 jobs

The grouped INT4 route is
`fnlp candidate source-int4 score-job start|resume --task classify|sentiment|judge`.
It uses the same authenticated storage protocol, with an explicit local source
closure or a digest-pinned INT4 snapshot. Its three tasks accept
`--input-encoding safe-bpe-v1`; this does not enable compact input on the INT8
commands documented above.

For example, with an existing protected job directory and key and immutable
limits as described above:

```sh
fnlp candidate source-int4 score-job start corpus.ndjson \
  --task classify --defaults labels.json \
  --snapshot ./model.int4.snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 16384 --group-width 128 \
  --input-encoding safe-bpe-v1 \
  --max-tokenizer-input-bytes 65536 \
  --max-tokenizer-output-tokens 8192 \
  --max-tokenizer-pair-checks 4194304 \
  --max-tokenizer-pair-bytes 67108864 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key \
  --limits ./job-limits.json --materialize
```

These tokenizer limits are per complete record and separate from its NDJSON
byte ceiling, native context capacity and lifetime job work. One cumulative
ledger covers the source and metadata, every pairwise order, every rubric
criterion, and independently encoded faithfulness evidence windows. Original
source bytes and UTF-8 byte/scalar evidence coordinates remain exact. Oversized
or exhausted requests fail without silently switching encoding or returning a
partial scored bundle.

Omitting the byte and token flags uses `--max-input-bytes` and per-head prompt
capacity respectively. The pair-work defaults are 4,194,304 checks and 67,108,864
examined bytes. Each configured ceiling is bounded by the existing safe-BPE
validator. The planner's modeled encoder workspace is admitted exactly once
alongside the retained population, prepared heads, native KV and guarded output.

Use the same policy and all four caps for `resume`. They are committed in the
immutable private recipe even when two configurations happen to produce the
same prompt tokens. Recipe mismatches refuse before uncommitted-tail repair.
The same binding applies to `--spool-input` and `--use-retained-input`.
Tokenizer work is bounded for each preparation, including a preparation on an
explicit resume; native work and attempts remain persisted lifetime charges.

Library callers first configure the actual `Int4ClassificationPlanner`,
`Int4SentimentPlanner` or `Int4JudgePlanner` with
`with_document_encoding(policy)`, then construct
`Int4ScoredJobPlanner::new_with_document_encoding` and pass it to
`NlpEngine::job_int4_scored`. The factory exposes its immutable
`document_encoding()` and `encoding_workspace_bytes()` for host admission.
The existing `Int4ScoredJobPlanner::new` remains byte-only and refuses compact
planners. Byte mode omits the new recipe and metadata field, preserving existing
authenticated job bytes.

Compact prompt tokenization is an explicit input profile. Byte preservation
and control-token exclusion do not establish semantic injection resistance,
native-output equivalence, task quality or a measured speedup. Regression
definitions cover recipe mutations across all five record shapes, admission,
metadata compatibility and authenticated recovery; execution and model
qualification remain pending the controller-selected DSR checkpoint.

## Persistent workers for grouped INT4 jobs

The grouped INT4 `score-job start` and `resume` commands also accept
`--compute-threads N`. Widths 1 through 64 include the coordinator; one is the
existing serial route. Runtime threads are additional. This option applies
to classification, sentiment and all three judgment modes, with either byte
fallback or explicit `safe-bpe-v1` input.

For example, add the physical width to the protected-job invocation above:

```sh
fnlp candidate source-int4 score-job start corpus.ndjson \
  --task classify --defaults labels.json \
  --snapshot ./model.int4.snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 16384 --group-width 128 --compute-threads 4 \
  --input-encoding safe-bpe-v1 \
  --max-tokenizer-input-bytes 65536 \
  --max-tokenizer-output-tokens 8192 \
  --max-tokenizer-pair-checks 4194304 \
  --max-tokenizer-pair-bytes 67108864 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key \
  --limits ./job-limits.json --materialize
```

One fixed team spans population authentication, every pending scored record,
all of its heads, durable attempts and commits, and optional materialization.
The existing strict INT4 projection executor divides numerical output rows
among workers. Input order, scored-language semantics, both pairwise orders,
result validation and journal acknowledgement keep their existing authority.
This command does not expose a native cohort or prompt-prefill schedule.

The process envelope admits the selected children and their modeled scratch
before execution. The host retains the input, private recipe and key owner,
native KV and scratch, and guarded result staging through physical completion.
The original cancellation cause and checkpoint/deadline allowance propagate
through the team. Every worker joins before the public API can return progress.
The CLI stages bounded metadata only after that return, then checks its original
deadline before and after delivery. Durable progress can precede a later
cancellation, output error or deadline failure; failure never implies rollback.

The width is excluded from authenticated semantic identity. To resume, change
`start` to `resume` and keep the original complete input or explicitly select
retained-input replay. The width may change, including between serial and team
execution; defaults, task policies, encoding caps and lifetime limits must not.
Already committed records remain skipped. Interrupted attempts and all five
native work axes remain charged.

Serial metadata keeps its previous shape. Nondefault widths add only:

```json
{
  "compute_schedule": {
    "execution": "int4-output-row-team-v1",
    "declared_compute_threads": 4
  }
}
```

This is selected physical configuration, not a projection counter or measured
speedup. Library callers use
`NlpEngine::job_int4_scored_with_threads(model, planner, request, limits, kernel, compute_threads, reader, cancellation)`.
The explicit threaded API accepts widths 2 through 64 that fit the installed
process envelope. Use `job_int4_scored` for serial execution; the same immutable
planner and owned storage request work in either route.

Six additional regression definitions cover width and memory admission,
original cancellation causes, charged input lifetime, bounded metadata and
delivery deadlines, plus controller-only lifecycle and pinned-model fixtures.
The actual-model fixture compares complete stored classification, sentiment,
pairwise, rubric and faithfulness results at widths 1, 2 and 4 separately for
byte and compact input, then replays authenticated retained input at another
width without new attempts or work. All executable checks remain **UNRUN**
pending the designated controller. No native parity, crash qualification,
task-quality or throughput result is asserted by this source change.
