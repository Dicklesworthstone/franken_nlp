# Complete-source INT4 faithfulness populations

The candidate INT4 path can check a stream of source/claim pairs with one
resident model. Each source is complete original text, including material
that might contradict the claim. The native planner partitions only that
source, repeats the exact claim for every chunk, and retains every chunk's
full-vocabulary NLI judgments and original-coordinate evidence.

This extends the existing `faithfulness-map` implementation to an ordered
population. It uses the same native map and explicit evidence reduction.
The reduction is uncalibrated: it does not establish full-context equivalence,
semantic correctness, or a factuality probability. A verified quote proves
membership in the original source.

## Ordered corpus command

Save an explicit policy as `policy.json`, for example:

```json
{
  "minimum_candidate_weight_ppm": 700000,
  "minimum_margin_milli": 100,
  "evidence_window_bytes": 256,
  "max_evidence_windows": 31,
  "max_evidence_spans": 31
}
```

These are illustrative descriptive thresholds, not a recommended calibrated
decision policy. The policy file is a strict object limited to 16 KiB. Missing,
duplicate or unknown fields are refused. A policy cannot contain a source,
claim, task budget, prompt or model identity.

Supply `claims.ndjson` with one original source and one exact claim per row:

```jsonl
{"id":"shipment-1","text":"The shipment arrived on Tuesday.","task_args":{"claim":"The shipment arrived on Tuesday."}}
{"id":"shipment-2","text":"The shipment arrived on Wednesday, after a delay.","task_args":{"claim":"The shipment arrived on Tuesday."}}
```

Run the live corpus:

```sh
fnlp candidate source-int4 faithfulness-corpus claims.ndjson \
  --policy policy.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-input-bytes 1048576 --max-chunks 64 \
  --max-requests 1000 \
  > results.ndjson
```

The source-directory route uses `--source-dir` and `--source-manifest` instead
of the snapshot flags. Optional `--save-snapshot NEW_FILE` follows fixed
model/policy/envelope admission. The snapshot may remain if a later record is
rejected. Acquisition is explicit; this command does not download a model.

Each row requires `task_args: {"claim": "..."}`. Claims may vary across rows.
Source and claim must both be nonempty, and their combined decoded UTF-8 byte
length must fit `--max-input-bytes`. The transport separately bounds each
encoded NDJSON line with `--max-line-bytes`, including escaping and trailing
CR. Row arguments cannot override the fixed policy, task budget or schedule.

Output uses `fnlp-candidate-int4-faithfulness-corpus-v1` around the existing
ordered batch events. Each success contains the complete source map result.
Serial results retain the native map directly; cohort results retain its
`output` plus physical scheduling counters. Every chunk, uncertain window,
contrary witness and original byte/scalar span remains available. A failed
chunk produces no success for that original document.

An exact `{"flush":true}` control drains the current ID epoch before IDs may
be reused. It resets no execution budget. Earlier complete success frames
remain usable after later rejection or failure. Whole-run success requires
**both `run_complete` and exit status zero**; the footer alone does not hide
rejected records. The command returns failure if writing or flushing fails.

## Work and scheduling

The five native flags bound the **whole live corpus**:

| Flag | Charged work |
| --- | --- |
| `--max-forward-positions` | All native forward positions across sources, chunks and evidence heads |
| `--max-projected-logits` | Every full-vocabulary candidate projection |
| `--max-attention-pairs` | All attention pairs |
| `--max-dot-products` | All grouped native projection dot products |
| `--max-multiply-accumulates` | All grouped native multiply-accumulate work |

A prepared document must fit the remaining invocation ledger. Admission
debits every axis before execution replans, admits or forwards it. An admitted
failed attempt is never refunded. Fatal execution failure prevents reuse;
preparation refusals can remain record-local when the engine is still clean.
Flush, another chunk, another document or another cohort never replenishes
these counters. The cooperative deadline and checkpoint allowance also span
the invocation.

Chunks execute serially by default. `--cohort-chunks N` explicitly shares
native layers across up to `N` independent source chunks within a document.
Every actual ragged window must fit **at most 64 simultaneous evidence
heads**. Total heads across a long document may exceed 64. All future windows
are checked before its first native forward; an oversized window refuses
without resizing, dropping evidence or falling back to serial execution.

The pure allocation quote prices `min(32 × N, 64)` heads. This conservative
peak does not itself reject widths above two: the actual number of evidence
heads depends on each source chunk. Other limits still apply. In particular,
the bounded cohort-output quote includes each chunk result plus its wrapper;
64 chunks at the default 1 MiB per-chunk cap exceed that output ceiling.
Narrow `--max-result-bytes` when appropriate for the intended result size;
actual oversized results still fail explicitly.

The live command requires `asupersync-runtime`. It rejects generation,
sampling, prompt overrides, thinking, tools, selected-row projection,
`--prefill-rows`, `--cohort-heads` and cross-document cohorts. Full-vocabulary
scoring remains part of the NLI result contract.

## Durable start and resume

`faithfulness-document-job start|resume` retains one complete-source judgment
per original record. It accepts the same `policy.json` and source/claim
population shown above. Supply the **complete original population on every
resume**, including committed records in their original order. Claims, source
bytes and IDs authenticate along with the fixed policy, model and recipe.
Durable populations require unique IDs and do not accept live flush controls.

The command requires `metadata-store` and `asupersync-runtime` on Linux x86-64
or AArch64. Prepare an existing owner-only job directory, a protected file
containing exactly 32 random raw secret bytes, and a random 128-bit job ID
encoded as 32 lowercase hexadecimal characters. The
[owned-job guide](owned-jobs.md) describes these storage prerequisites. The
command does not generate keys or create or adopt a job directory.

An illustrative `faithfulness-job-limits.json` is:

```json
{
  "max_items": 3,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 1048576,
  "max_snapshot_bytes": 4194304,
  "max_result_bytes": 16781312,
  "max_spool_bytes": 268435456,
  "max_materialized_bytes": 268435456,
  "max_journal_bytes": 67108864,
  "max_attempts": 6,
  "max_work": {
    "model": {
      "forward_positions": 262144,
      "projected_logits": 10000000000,
      "attention_pairs": 1000000000000,
      "projections": {
        "dot_products": 10000000000000000,
        "multiply_accumulates": 10000000000000000
      }
    },
    "mask_node_visits": 0
  }
}
```

These are explicit allowances, not measured costs or a guarantee that a
population fits. `mask_node_visits` must be zero because this route scores
the complete fixed candidate language. The stored item byte limit must fit
`--max-input-bytes`; it bounds both the entire serialized record and the
planner's combined source-plus-claim capability. The complete population
must also fit its snapshot and transport bounds.

Start with explicit retention:

```sh
fnlp candidate source-int4 faithfulness-document-job start claims.ndjson \
  --policy policy.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --context-tokens 2048 --max-chunks 64 \
  --store-results --job-dir /private/faithfulness-job \
  --job-id "$JOB_ID" --key-file /private/faithfulness-job.key \
  --limits faithfulness-job-limits.json --materialize
```

Use the same invocation with `resume` in place of `start` to continue pending
records. Repeat the exact policy, model/group identity, task and planning
limits, map settings, schedule and immutable job limits. Committed records
are authenticated and skipped; an unfinished source is retried from its
beginning if the original attempt and work ceilings permit it. No partial
source result is committed.

For this durable command, the five native flags bound **each complete
document**, and every one must fit the corresponding **lifetime** counter in
`JobLimits.max_work`. The immutable job ledger is the authority across
documents, failures and resumes. Increasing attempts or restarting never
replenishes work. The per-invocation timeout and cooperative checkpoint
allowance govern the current call without changing those persisted debits.

Optional `--cohort-chunks N` uses the same actual ragged-window checks as the
live corpus. The exact schedule binds the authenticated recipe. Serial
stored results require the `--max-map-result-bytes` cap; cohort stored results
need an additional 4096 bytes for their wrapper. The illustrated stored cap
covers either schedule at the default 16 MiB semantic result cap.

Stdout uses `fnlp-candidate-int4-faithfulness-document-job-v1` and contains
only public model and job progress metadata. It does not emit source, claim,
policy, scores, private paths or secrets. `--materialize` publishes verified
private `materialized.ndjson` after all items commit; it never overwrites an
existing materialization. The [job management CLI](job-management-cli.md)
can inspect and verify retained state.

Ordinary resume refuses uncommitted spool bytes or stages. Explicit
`resume --discard-uncommitted-tail` authorizes their removal **after** model,
recipe and original-population authentication. Changed source or claim bytes
refuse before this repair. Durable progress can already exist when a later
timeout or metadata write/flush fails, so check the exit status and
authenticate stored state rather than assuming rollback.

## Library surfaces

Construct an `Int4JudgePlanner` from the actual resident model, then wrap it:

```rust
use franken_nlp::batch::judge::grouped_int4::Int4FaithfulnessDocumentBatchPlanner;

let planner = Int4FaithfulnessDocumentBatchPlanner::new(
    judge_planner,
    policy,
    mapping_limits,
    whole_run_work,
)?;
// Optional explicit schedule:
let planner = planner.with_chunk_cohorts(3)?;
```

`NlpEngine::batch_int4_faithfulness_documents` takes the planner, resident
model, `CorpusLimits`, explicit kernel, owned reader/writer and cancellation
token. `int4_faithfulness_document_corpus_requirements` quotes its complete
hosted reservation before IO. `Int4FaithfulnessDocumentArgs` contains only
the required claim; `BatchDocument.text` is the original source.

Preparation is sealed to its planner. Execution recompiles the source/claim
and compares the full identity list, geometry and planned work with the
prepared descriptor before forwarding. The executor uses the existing map
for both serial and cohort schedules. No caller-supplied scorer or fabricated
chunk result enters the native path.

The durable library surface is
`jobs::runner::scored::grouped_int4::faithfulness_document::Int4FaithfulnessDocumentJobPlanner`
with `NlpEngine::job_int4_faithfulness_document`. It reuses the corpus
processor and the existing authenticated population/journal runner. Every
complete source and its claim is one durable result. The recipe binds policy,
map settings, actual task/planning limits, schedule and result envelope; the
job binds the original population, model and immutable lifetime work. Failed
attempt debits survive restart. Private content is retained only through
explicit result storage; model and recipe identities do not permit implicit
input spooling.

## Ownership and evidence status

Memory admission includes the complete native map, original source and claim,
all retained chunk results, transport or authenticated original population,
and 32 KiB per maximum chunk for frozen identities during replanning.
Output guards remain owned through stream delivery or durable serialization,
spool synchronization and journal acknowledgement. IO and native state stay
inside the physical invocation until it drains. Blocking IO observes
cancellation cooperatively.

Focused regression definitions cover strict per-record claims and policy,
Unicode byte limits, all five lifetime counters, changed-claim authentication,
ragged geometry, output ownership, memory coexistence, cancellation and
restart behavior. Compilation and test execution remain **UNRUN** in this
session under the repository's controller-owned DSR policy. Native parity,
semantic quality, performance and release qualification require their
separate evidence; this implementation does not assert them.
