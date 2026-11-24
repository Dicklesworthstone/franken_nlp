# INT8 extraction with semantic field review

`fnlp candidate extract-semantic` runs constrained extraction and then reviews explicitly configured scalar claims against the complete original source, using one resident INT8 model and one owned invocation. The result retains the original extraction unchanged and adds per-field diagnostics.

This current-candidate route requires `asupersync-runtime` and an explicitly selected local INT8 artifact. The second reader is the same model: its output is correlated, experimental, and uncalibrated. It does not repair fields, decide acceptance, or certify correctness.

## Complete example

The checked-in [schema](examples/int8-semantic/schema.json), [options](examples/int8-semantic/options.json), and [source](examples/int8-semantic/source.txt) demonstrate a revenue field and an exact source quote. The claim rule says what the integer means; the system does not infer a field's meaning from its name.

```sh
fnlp candidate extract-semantic docs/examples/int8-semantic/source.txt \
  --model /models/candidate-int8.fnlpq \
  --memory-mib 16384 \
  --context-tokens 2048 \
  --max-new-tokens 128 \
  --max-input-bytes 512 \
  --max-result-bytes 16384 \
  --schema docs/examples/int8-semantic/schema.json \
  --source-membership \
  --semantic-options docs/examples/int8-semantic/options.json
```

These are input and configuration examples, with no asserted model output or quality result. The options explicitly opt into the experimental reader and provide a revision, typed claim paths, judge thresholds, evidence coverage, both-stage budgets, semantic limits, and all five complete-request model-work ceilings. Options and schema must be bounded local files; the source may instead come from stdin.

The example's claim rule renders `prefix + exact JSON scalar + suffix`. For a revenue value of `1250000`, that means `Acme reported revenue of 1250000 US dollars for 2025.` Strings remain quoted JSON scalars, and decimal values are not converted through binary floating point. A `property` path step identifies a literal schema property; `each_item` applies a rule to the scalar leaves of an array. Every configured path and scalar kind must match the schema before model loading. Duplicate rules, unknown fields, duplicate JSON keys, and rules on verbatim fields are refused.

## Reading the result

The candidate response's `output.extraction.result.output.json` is the original exact JSON string. `output.semantic.fields` contains JSON pointers and one of:

- `entailed`: the configured claim receives an entailment result under the explicit judge policy.
- `contradicted`: the judge finds contrary source evidence under that policy.
- `unsupported`: the judge does not find support for the claim under that policy.
- `not_checked`: the field was not given an accepted semantic assessment; the associated reason explains why.

The example's `quote` field uses `x-fnlp-source: "verbatim"`. Its independent source-membership validation remains part of extraction, and semantic review reports `verbatim_membership_only`. Null values, fields without claim rules, empty containers, and judge abstentions also retain explicit `not_checked` reasons. Missing optional fields are not invented. A failed or cancelled stage returns no successful combined result; it is never translated into `unsupported`.

`same_model_correlated` and `experimental_uncalibrated_diagnostic_only` remain visible in the receipt. Evidence quotes and original-source coordinates allow inspection, but source membership alone does not prove entailment.

## Bounds and execution

The options file is read first under caller preparation admission. The CLI then prices the declared maximum source, retained field plans and source copies, native context, intermediate results, and combined output before reading schema, source, or model metadata. It rechecks the quote against the consumed plan before weights are loaded. The source and every dynamic judge must fit the explicit context and evidence policy; no tail is silently dropped.

The example uses a 2,048-position context. Its judge `max_kv_bytes` is `369098752`, the complete resident bf16 KV allocation for that context (`2048 * 180224`). Change this field with `--context-tokens`; both stages must admit the same reused native context. The example's attention ceiling is `17716740096`, covering its 4,096 declared forward positions at the full 2,048-position context across 44 KV slots and 48 query heads; it is a finite work allowance, not a measured usage claim. `--max-result-bytes` bounds the original extraction, while `limits.max_output_bytes` in the options bounds the combined semantic result. The extraction bound must also fit `limits.max_input_json_bytes`.

The five native ceilings cover forward positions, projected logits, attention pairs, projection dot products, and projection multiply-accumulates. The maximum extraction reservation remains charged even when extraction stops early. Actual values determine the review prompts, so every dynamic judge is compiled and preflighted after extraction and before the first review forward. All judges run within the remaining complete-request allowance.

This CLI route keeps full-vocabulary projections. Serial execution, sequential prompt processing and byte-fallback input remain the defaults; compact input, prompt morsels and worker teams are explicit choices. The command does not accept selected-row or sampling flags. The deadline spans preparation, loading, extraction, review, checked serialization, write, and flush; output and preparation charges remain held through delivery. Memory values describe the process resource ledger, not a measured operating-system RSS limit.

## Rust API

Prepare a real `PreparedInt8SemanticExtraction` from a pinned extraction plan, judge planner, and explicit `Int8SemanticOptions`. Use `Int8SemanticHostLimits::requirements` for the default complete quote, `requirements_with_encoding` for compact input, or `requirements_with_schedule` for an explicit `Option<Int8PrefillLimits>`. The host consumes the plan, reuses one strict INT8 engine for both stages, and preserves typed cancellation causes.

The default entry point is `NlpEngine::execute_int8_semantic_extract`.
`execute_int8_semantic_extract_layer_major` adds bounded prompt morsels;
`execute_int8_semantic_extract_with_threads` adds one persistent row team; and
`execute_int8_semantic_extract_layer_major_with_threads` combines them.
Serial entries return a guarded `HostedOutput<Int8SemanticExtractionRun>`;
team entries return `HostedInt8TeamResult<Int8SemanticExtractionRun>` with the
same guarded result and compute receipt. Caller preparation, complete retained
review data, coordinator prompt scratch, child workspace and output all require
separate admission.

## Compact input across extraction and every review

Add `--input-encoding safe-bpe-v1` to encode the original schema/source and
every generated claim and evidence prompt compactly while preserving their
decoded bytes and excluding template-control token IDs. One allowance covers
the complete operation: the work already spent preparing extraction is carried
into semantic planning, and none of the four ceilings renews between fields.

```sh
fnlp candidate extract-semantic docs/examples/int8-semantic/source.txt \
  --model /models/candidate-int8.fnlpq --memory-mib 16384 \
  --context-tokens 2048 --max-new-tokens 128 --max-input-bytes 512 \
  --max-result-bytes 16384 --schema docs/examples/int8-semantic/schema.json \
  --source-membership --semantic-options docs/examples/int8-semantic/options.json \
  --input-encoding safe-bpe-v1 \
  --max-tokenizer-input-bytes 65536 --max-tokenizer-output-tokens 8192 \
  --max-tokenizer-pair-checks 1000000 --max-tokenizer-pair-bytes 8000000
```

The cumulative byte/token ceilings are distinct from the original document
byte cap and from each native context. The pair ceilings count comparisons and
examined bytes across both stages. All fields still compile and preflight
before review inference. Exceeding a ceiling returns no combined result; there
is no renewed per-field allowance or fallback.

Compact mode defaults the caller preparation reservation to 1,024 MiB.
The host separately prices one transient encoder workspace beside all retained
source copies, prompts, field results, exact JSON and output storage. Existing
field-count and output limits remain in force.

`output.input_encoding` records the exact policy and actual cumulative work.
Byte-default output omits that extension. Library callers configure the pinned
extraction and judge with the same policy and quote
`Int8SemanticHostLimits::requirements_with_encoding` before execution; a
mismatched policy or changed ceiling is refused. The original source bytes,
schema numbers and extraction JSON remain unchanged. Compact prompt tokens
differ from byte fallback, so this is not a tokenizer-parity, identical-output,
model-quality or measured-speed claim.

Regression definitions cover policy mismatch, all four cumulative ceilings,
retained original work, exact schema/source preservation, byte-default output,
and additive admission. Compilation, execution, actual-model behavior and
performance remain subject to the repository's controller-owned DSR gates.

## Prompt morsels and one persistent review team

Add `--prefill-rows 8` to process bounded groups of prompt tokens throughout
extraction and every judge head. Add `--compute-threads 4` to admit one
persistent numerical row team for the complete operation. Both flags accept
values from 1 through 64; the width includes the coordinator and runtime
threads are additional. They compose with either input encoding.

The same native engine and worker team stay attached from original extraction
through every complete review. Field count does not create new teams or new
model-work allowances. Every dynamic judge still compiles and preflights
before the first review forward. Full-vocabulary scoring, exact extraction
JSON, evidence coordinates, semantic status rules and all original native
ceilings are unchanged.

Coordinator prompt scratch and fixed child workspace are priced separately,
before schema/source/model IO. Neither replaces full resident KV, retained
field plans, source copies or output storage. Cancellation and deadline
checks span the whole operation, and no combined result escapes until every
worker has physically joined. The guarded output remains owned through
serialization and flushing. Width one omits a compute extension; a team adds
the existing top-level `compute` receipt with the admitted width and completed
projection/row counters.

Regression definitions cover invalid geometry, source and preparation
retention, exact memory boundaries, independent coordinator/child admission,
schedule mismatch before IO, and unchanged serial publication. They do not
establish actual-model parity or throughput. These explicit schedules remain
candidates pending controller-owned DSR validation.

## Resident semantic extraction populations

`fnlp candidate extract-semantic-batch` applies the same fixed schema and
explicit review policy to an ordered NDJSON stream using one resident INT8
engine and compiler pair. Each successful record contains the unchanged
extraction and its complete field review. Non-null `task_args` is refused
before encoding: a record cannot replace the schema, grounding, review policy
or budgets. Ordinary batch flush markers retain their transport meaning and
do not renew work.

For a one-record run, place the original example source in this envelope:

```jsonl
{"id":"report-1","text":"Acme reported revenue of 1250000 US dollars for 2025."}
```

Then run:

```sh
fnlp candidate extract-semantic-batch reports.ndjson \
  --model /models/candidate-int8.fnlpq --memory-mib 16384 \
  --context-tokens 2048 --max-new-tokens 128 --max-input-bytes 512 \
  --max-result-bytes 16384 --schema docs/examples/int8-semantic/schema.json \
  --source-membership --semantic-options docs/examples/int8-semantic/options.json \
  --compute-threads 4 --prefill-rows 8
```

The stream uses the `fnlp-candidate-int8-semantic-extraction-batch-v1`
protocol. Combined results retain exact extraction JSON strings and the same
four field states described above. The terminal completion frame follows
physical worker join; a failed extraction or review never produces a
successful extraction-only record.

The semantic options' five-axis `max_model_work` limits one complete
extraction-and-review attempt. Population work has separate ceilings:

| Flag | Invocation allowance |
| --- | --- |
| `--max-total-forward-positions` | All native forward positions |
| `--max-total-projected-logits` | Full-vocabulary projected logits |
| `--max-total-attention-pairs` | Native attention work |
| `--max-total-dot-products` | Projection dot products |
| `--max-total-multiply-accumulates` | Projection multiply-accumulates |
| `--max-total-mask-node-visits` | Extraction grammar-mask work |

Each native default is the options file's allowance for **one complete
attempt**. The mask default is the extraction's per-attempt mask allowance.
Increase the totals explicitly for a larger population; increasing the request
count does not multiply them. Before either native stage, the processor spends
the full declared attempt allowance. Early EOS, failed review, cancellation,
and flushes do not refund that reservation.

Compact populations retain the per-document `--max-tokenizer-*` allowance
across schema/source encoding and every field review, and additionally use the
four `--max-total-tokenizer-*` invocation ceilings. Actual consumed tokenizer
work stays charged across documents, discarded preparation, failed or
cancelled review planning, and flushes. A prepared item is sealed to its own
processor invocation, so moving it to another processor cannot evade that
debit. The encoding receipt reports the document's actual work under the
original declared policy, not the smaller temporary allowance remaining at
one planning step.

Serial, bounded-prefill, persistent-worker and combined schedules share the
same task and work contracts. The complete quote prices all retained field
plans, source copies, transport, one encoder and ledger, full native context,
prompt scratch, worker buffers and combined output before private input or
model IO. Actual compiler and vocabulary objects, readers, writers and output
authority remain owned through physical join, including failed work.

Library callers use `batch::extract::semantic::int8` for the sealed planner
and native processor. The pure quote is
`hosted::corpus::semantic::int8_semantic_corpus_requirements`;
`NlpEngine::corpus_int8_semantic_extract` and
`corpus_int8_semantic_extract_scheduled` run owned readers and writers through
the existing bounded transport. The scheduled entry accepts optional
`Int8PrefillLimits` and `(Int8Kernel, usize)` team settings.

The population implementation and source regression definitions are present.
Compilation, executable regressions, independently pinned model equivalence,
task-quality evaluation and target-host performance remain controller-owned
validation gates.

## Retained semantic extraction jobs

Use `fnlp candidate extract-semantic-job start|resume` for authenticated
populations with complete extraction-and-review results, durable work debits,
explicit retained input and optional materialization. See the
[INT8 semantic job guide](candidate-semantic-jobs.md) for setup, bounded examples,
resume authentication, compact invocation accounting and physical scheduling.
