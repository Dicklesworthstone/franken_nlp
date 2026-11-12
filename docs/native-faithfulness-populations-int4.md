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
  --compute-threads 4 \
  --max-input-bytes 1048576 --max-chunks 64 \
  --max-requests 1000 \
  > results.ndjson
```

The source-directory route uses `--source-dir` and `--source-manifest` instead
of the snapshot flags. In byte mode, optional `--save-snapshot NEW_FILE` follows fixed
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

With `--compute-threads N`, one admitted numerical output-row team spans the
entire corpus, including every source, chunk, evidence head and flush epoch.
The default `1` uses the existing serial route; `2` through `64` select one
coordinator plus `N - 1` persistent children. Documents and claims keep their
original order. The team joins before the terminal `run_complete` frame can
be published. The optional top-level `compute_schedule` field records
`execution: "int4-output-row-team-v1"` and `declared_compute_threads`; it is
omitted for width one. This is physical scheduling metadata, not a fabricated
projection count, speed measurement or new judgment score. The wrapped inner
events and complete-source results keep their existing contracts.

## Bounded compact source and claim encoding

Both population commands accept `--input-encoding safe-bpe-v1`. Repeat the
same selection and all four work limits on `faithfulness-document-job resume`:

| Flag | Bounded work within one document attempt |
| --- | --- |
| `--max-tokenizer-input-bytes` | Total UTF-8 bytes consumed across every encoding call |
| `--max-tokenizer-output-tokens` | Total tokens produced across every encoding call |
| `--max-tokenizer-pair-checks` | Total BPE pair inspections |
| `--max-tokenizer-merge-bytes` | Total bytes processed by BPE merges |

The shared allowance covers **both preparation and execution reconstruction**.
Each pass includes claim-capacity preparation, accepted and rejected source
sizing probes, and every complete chunk judgment with its independently
encoded evidence windows. It does not renew for another chunk, evidence head
or native cohort. A document whose preparation fits can still be refused
during reconstruction if the remaining allowance is insufficient. The native
attempt debit has already been charged at that point and is not refunded.

These tokenizer ceilings apply per document attempt. They are separate from
the live corpus's five whole-run native ceilings and the durable job's
persisted native lifetime and attempt limits. A complete map's encoding work
receipt describes its own planning pass so preparation and reconstruction can
compare the same sealed result; the in-memory attempt ledger carries the
cumulative consumption across both passes. It is not a new persisted lifetime
tokenizer counter. `--max-tokenizer-calls` separately bounds chunk-sizing
probes. Original source/claim bytes still obey `--max-input-bytes`, and each
actual prompt must fit the native context.

Compact token counts choose chunk boundaries without altering the original
UTF-8 source or exact claim. Evidence windows are encoded from their original
byte spans, independently of the enclosing chunk's token sequence. Evidence
byte/scalar coordinates and counterevidence remain complete. Tiny ASCII
evidence windows are supported; an indivisible scalar or CRLF that cannot fit
the declared evidence byte cap refuses the document.

One bounded encoder workspace is admitted alongside map, transport and
retained-plan ownership before private IO. Its size is not multiplied by the
number of documents, chunks, evidence heads, cohorts or numerical workers.
The selected full policy appears on every live event and on durable progress
and error metadata. It also binds execution identity and the immutable job
recipe: changed limits or a changed encoding refuse authenticated resume.
Physical worker width may still change under the existing scheduling rules.

Byte fallback remains the default. Omitting compact input preserves the old
byte-mode recipe and omits optional encoding provenance. Compact input is
supported with independently pinned snapshot loading; `--save-snapshot` is
refused before private IO in compact mode. Compact prompts can change model
output; this feature makes no judgment-parity, quality or speed claim.

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

### Numerical workers and native chunk cohorts

Numerical workers parallelize output rows within the existing INT4
projections. The planner still uses the exact complete source and claim,
full-vocabulary NLI scores, fixed evidence policy, original evidence
coordinates and the same complete-source reduction. All five native work
ceilings and the original invocation deadline retain their meaning. Both byte
fallback and explicit bounded compact input use the same scheduling rules.

Worker scratch is admitted once for the complete invocation before private
policy, model or population IO. The CLI also prices one runtime thread in
addition to the selected compute width, for a process ceiling of `N + 1`
threads. Worker width does not multiply native work allowances, evidence-head
capacity, result limits or population memory. The memory quote still includes
all existing model, preparation, map, transport, output and cleanup owners.

Multiple numerical workers cannot combine with **any explicit**
`--cohort-chunks`, including `--cohort-chunks 1`. Such combinations, widths
outside `1..=64`, prompt prefill and evidence-head cohort overrides are
refused before private IO. Select one numerical team or the existing native
chunk-cohort schedule for an invocation.

Chunks execute serially by default. `--cohort-chunks N` explicitly shares
native layers across up to `N` independent source chunks within a document.
Every actual ragged window must fit **at most 64 simultaneous evidence
heads**. Total heads across a long document may exceed 64. All future windows
are checked before its first native forward; an oversized window refuses
without resizing, dropping evidence or falling back to serial execution.

By default, the pure allocation quote prices `min(32 × N, 64)` heads. This
peak does not itself reject widths above two: the actual number of evidence
heads depends on each source chunk. Other limits still apply. In particular,
the bounded cohort-output quote includes each chunk result plus its wrapper;
64 chunks at the default 1 MiB per-chunk cap exceed that output ceiling.
Narrow `--max-result-bytes` when appropriate for the intended result size;
actual oversized results still fail explicitly.

### Smaller fixed-policy head ceilings

Both population commands accept optional `--max-evidence-heads H`, with
`1 <= H <= 32`. This narrows the admitted **total NLI heads per source
chunk**, including the whole-source head. Set it only when the fixed policy
fits that ceiling:

| Fixed `max_evidence_windows` | Required head ceiling |
| --- | --- |
| `1` | `1`, because the single window reuses the whole-source head |
| `2` through `31` | `max_evidence_windows + 1` |

`max_evidence_spans` limits retained witnesses and does not lower the number
of scored heads. A conflicting policy is refused before model loading or
population input. No source window is removed to satisfy the ceiling.

The smaller ceiling is used during the initial memory admission, including
retained plans and the per-chunk task KV budget. Cohorts then reserve
`min(H × N, 64)` physical heads. Serial execution continues to reserve one
physical KV row. The actual complete ragged schedule is still checked before
native execution; a window above 64 heads refuses even when the allocation
quote was capped at 64. All five work allowances, complete result limits and
source/claim byte ceilings retain their original meaning.

For a fixed policy with `max_evidence_windows: 1`, `max_evidence_spans: 1`
and `evidence_window_bytes: 256`, the options
`--max-evidence-heads 1 --cohort-chunks 8 --max-chunk-bytes 256` admit
eight native heads per full cohort. The whole source must still fit the
configured number of chunks and every exact claim-aware prompt must fit the
context. Narrowing the ceiling can make this invocation fit a memory limit
that the default 32-head allowance would refuse.

Omitting the option preserves the previous 32-head task budget and memory
quote. The one-shot `faithfulness-map` command retains its existing behavior
and does not accept this population-only option.

The live command requires `asupersync-runtime`. It rejects generation,
sampling, prompt overrides, thinking, tools, selected-row projection,
`--prefill-rows`, `--cohort-heads` and cross-document cohorts. Full-vocabulary
scoring remains part of the NLI result contract.

## Durable start and resume

`faithfulness-document-job start|resume` retains one complete-source judgment
per original record. It accepts the same `policy.json` and source/claim
population shown above. Supply the **complete original population on every
resume** by default, including committed records in their original order. Claims, source
bytes and IDs authenticate along with the fixed policy, model and recipe.
Durable populations require unique IDs and do not accept live flush controls.

Explicit input retention is also available: start with `--spool-input
--max-retained-input-mib N`, then resume with `--use-retained-input
--max-retained-input-mib N` and omit the population argument. Retained creation
stages one envelope at a time, authenticates the complete original population
and seals its spool before inference. Retained replay authenticates the whole
spool before repair, then lends one pending record at a time. Both modes keep
the bounded keyed item index and strict per-record parser workspace; logical
snapshot and lifetime limits still apply. See the
[retained-input guide](retained-inputs.md) for consent, disk caps and interrupted
creation behavior. Numerical teams support both external and retained inputs.

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
  --compute-threads 4 \
  --context-tokens 2048 --max-chunks 64 \
  --store-results --job-dir /private/faithfulness-job \
  --job-id "$JOB_ID" --key-file /private/faithfulness-job.key \
  --limits faithfulness-job-limits.json --materialize
```

Use the same invocation with `resume` in place of `start` to continue pending
records. Repeat the exact policy, model/group identity, task and planning
limits, map settings, exact input-encoding policy, native chunk schedule and immutable job limits. Committed records
are authenticated and skipped; an unfinished source is retried from its
beginning if the original attempt and work ceilings permit it. No partial
source result is committed.

The physical `--compute-threads` width may change on explicit resume. For
example, use width four for the start above and width two on a later resume,
with the same authenticated inputs and other required settings. Numerical
width is excluded from the job recipe and execution identity; it cannot
renew attempts or lifetime work. One team spans the complete invocation,
including input authentication, every source/chunk/evidence head, result
commit and optional materialization. All workers join before successful job
progress is returned. The metadata response adds the same optional
`compute_schedule` description as the live corpus, omitted for width one.

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

An explicit `--max-evidence-heads` also binds the authenticated job recipe.
Repeat the exact option on every resume. Even explicit `32` is a distinct
opt-in recipe from omission, although its task and memory allowances are
equal. Resume jobs created without this option by continuing to omit it;
their existing recipe bytes remain unchanged. A smaller ceiling never
increases or resets the immutable lifetime work counters.

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

To choose compact input, call `.with_document_encoding(encoding)?` on the
`Int4JudgePlanner` before constructing either population planner. Both expose
`.document_encoding()` for the exact immutable policy. When quoting memory
outside the hosted route, call `.with_input_encoding(planner.document_encoding())?`
exactly once on the returned `Int4FaithfulnessMapMemoryRequirement`.
The hosted routes apply that addition themselves.

`NlpEngine::batch_int4_faithfulness_documents` takes the planner, resident
model, `CorpusLimits`, explicit kernel, owned reader/writer and cancellation
token. `int4_faithfulness_document_corpus_requirements` quotes its complete
hosted reservation before IO. `Int4FaithfulnessDocumentArgs` contains only
the required claim; `BatchDocument.text` is the original source.

`NlpEngine::batch_int4_faithfulness_documents_with_threads` takes the same
arguments with explicit `compute_threads` immediately after the kernel. It
requires `2..=64`, accepts no native chunk-cohort setting, and retains one
admitted team across the complete stream. The scalar method remains the
width-one API. Pure document/corpus quotes retain their existing semantic
meaning; team dispatch additionally admits one worker-scratch envelope and
the declared process-thread capacity.

Both population planner types provide `.with_max_heads(H)?` and
`.max_heads()`. The builder validates the fixed policy and the prepared
descriptor remains bound to the admitted ceiling. Use
`int4_faithfulness_document_corpus_requirements_with_max_heads` with that
same ceiling for a smaller pre-IO corpus quote; the original quote function
continues to use 32. For native map quotes,
`Int4FaithfulnessMapMemoryLimits::requirements_with_max_heads` narrows serial
plan ownership, and `cohort_requirements` uses its explicit head ceiling for
both plans and native state. These pure quote functions require the caller
to enforce the corresponding policy and prepared-head capability; the
population planners and hosted routes perform those checks.

Preparation is sealed to its planner. Execution recompiles the source/claim
and compares the full identity list, geometry and planned work with the
prepared descriptor before forwarding. The executor uses the existing map
for both serial and cohort schedules. No caller-supplied scorer or fabricated
chunk result enters the native path.

The durable library surface is
`jobs::runner::scored::grouped_int4::faithfulness_document::Int4FaithfulnessDocumentJobPlanner`
with `NlpEngine::job_int4_faithfulness_document`. Its threaded counterpart is
`NlpEngine::job_int4_faithfulness_document_with_threads`, with explicit
`compute_threads` immediately after the kernel and before the owned reader.
It requires `2..=64` and rejects native chunk cohorts. Both reuse the corpus
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
spool synchronization and journal acknowledgement. Document planners and
native state finish their work inside the physical invocation and drop before
worker join. The model and worker scratch remain charged through physical
worker drain; corpus IO remains charged through post-join terminal delivery.
Public return waits for that drain, including cancellation, writer errors and
unwind. Blocking IO observes cancellation cooperatively. A terminal
cancellation or metadata-delivery failure can follow a durable commit;
explicit authenticated resume reconciles that state.

Focused regression definitions cover strict per-record claims and policy,
Unicode byte limits, all five lifetime counters, changed-claim authentication,
ragged geometry, output ownership, memory coexistence, cancellation and
restart behavior. Team regression definitions additionally cover width and
cohort refusal before IO, scratch accounting, process-thread admission,
physical drain and terminal delivery, unchanged inner framing, all typed
cancellation causes, and real model-bound recipe/identity invariance across
widths one/two/four/64 for start/resume. Recipe fixtures distinguish omitted
evidence-head caps from explicit caps, including explicit 32. Compact-input
regression definitions cover cumulative preparation/reconstruction budgets,
changed-policy resume authentication, retained replay, pre-IO memory admission,
exact policy provenance and byte-mode compatibility. These are
source definitions; they do not establish numerical parity or throughput.
Compilation and test execution remain **UNRUN** in this
session under the repository's controller-owned DSR policy. Native parity,
semantic quality, performance and release qualification require their
separate evidence; this implementation does not assert them.
