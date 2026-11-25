# Candidate finite-scoring judgment

`fnlp candidate judge` connects the existing native strict-INT8 judge to the
executable. The input is one bounded JSON object, with `mode` set to `pairwise`,
`rubric` or `faithfulness`. Every mode requires an explicit policy. The command
uses complete finite-candidate scoring with full-vocabulary denominators and
terminal EOS, not free generation followed by a JSON parser or label coercion.

```sh
fnlp candidate judge request.json --model ./model.fnlpq --memory-mib 8192
```

Omit the input path or use `-` for stdin. The binary needs `asupersync-runtime`
and a compatible explicitly selected local current-candidate artifact. The
command does not download or activate models, authenticate a publisher, or
promote artifact, numerical-fidelity or task-quality evidence.

## Bounded compact INT8 input

All three one-context judgment modes accept explicit compact input:

```sh
fnlp candidate judge request.json --model ./model.fnlpq --memory-mib 8192 \
  --input-encoding safe-bpe-v1 --compute-threads 4 --prefill-rows 4
```

Omitting `--input-encoding` keeps `byte-fallback`. The compact policy preserves
the exact untrusted UTF-8 bytes while excluding template-control token IDs.
It applies to the criterion and both candidates in pairwise mode, the document
and every criterion description in rubric mode, and the claim, full source and
every original-byte evidence window in faithfulness mode. A byte offset never
becomes a slice into compact token IDs. Both presentation orders, every rubric
criterion and all required faithfulness heads must still complete.

One tokenizer allowance covers the complete judgment, with these defaults:

| Option | Default | Scope |
| --- | ---: | --- |
| `--max-tokenizer-input-bytes` | 1048576 | Sum of untrusted bytes encoded across the complete preparation |
| `--max-tokenizer-output-tokens` | 1048576 | Sum of untrusted tokens produced across the complete preparation |
| `--max-tokenizer-pair-checks` | 4194304 | All BPE pair comparisons in the same preparation |
| `--max-tokenizer-pair-bytes` | 67108864 | All examined BPE pair bytes in the same preparation |

These limits are independent of `--max-input-bytes`, which bounds the original
request JSON, and `--context-tokens`, which bounds each native head. In
faithfulness mode the source is encoded for the whole-source head and, when
there is more than one evidence window, again across the complete evidence
partition. No allowance renews between segments or heads. Exhaustion refuses
the complete request before inference; it never switches back to byte input
or drops evidence.

The selected policy and all four ceilings bind the actual planner's template
identity. Serial execution, prompt prefill, a persistent CPU team and their
combination use that same semantic plan. Before private input or model IO, the
CLI admits one complete transient encoder workspace beside the retained plans,
native storage and selected schedule's workspace. Invalid or unused encoding
flags refuse before IO. Planning preserves the original terminal cancellation
cause and never returns a partial judgment.

Compact responses add `input_encoding` with the exact policy and
`input_encoding_work` with the actual complete preparation counters. Byte
responses omit both fields. Compact input changes the prompt token sequence;
these routes make no identical-output, tokenizer-parity, task-quality or
measured-speed claim. Native context, model work and result limits remain
separate requirements.

## Pairwise comparison: both presentation orders

```json
{
  "mode": "pairwise",
  "criterion": "Prefer the answer that directly addresses the question.",
  "a": "The package is due on Tuesday.",
  "b": "Packages arrive on business days.",
  "policy": {
    "minimum_margin_milli": 0,
    "maximum_order_disagreement_milli": 1000
  }
}
```

The native planner compiles both A/B and B/A presentations. Both must complete;
the finalizer maps their log odds back to the original A/B identities before
combining them. Output preserves both complete score sets, order disagreement,
combined margin, candidate-conditional weight and prefer-A/prefer-B/tie/abstained
decision. Threshold units are thousandths of natural-log score ratios, not ppm
probabilities. The values above are an illustrative explicit policy, not a
calibrated recommendation. No failed order becomes a tie or abstention.

## Rubric: every independent criterion

Rubric requests contain `mode: "rubric"`, `document`, `rubric` and `policy`.
The rubric is the existing `RubricDefinition` shape:

* `schema_version`: 1; a bounded `revision` identifier;
* `declared_origin_digest`: the caller's 64-lowercase-hex SHA-256 declaration;
* `scale_maximum`: an integer from 1 through 10;
* `criteria`: one through 16 objects with `id`, `description` and `weight`.

Criterion IDs must be unique valid identifiers; descriptions must be nonempty.
Weights are integers in 1 through 1000000. The caller's origin digest and
revision are retained as declarations, not verified publisher provenance. The
CLI does not invent them or label a user rubric as a qualified shipped preset.

The explicit policy contains `minimum_peak_weight_ppm` and
`maximum_normalized_entropy_ppm`, each in 0 through 1000000. Every criterion is
scored independently over the complete integer scale 0 through `scale_maximum`.
The existing finalizer computes the weighted aggregation; it does not omit
failed or abstained criteria to manufacture an aggregate success. Results keep
all criterion distributions, weights, diagnostic means and decision disclosures.

## Faithfulness: whole source plus all evidence windows

```json
{
  "mode": "faithfulness",
  "source": "Alice moved to Paris. Bob stayed in Berlin.",
  "claim": "Alice moved to Paris.",
  "policy": {
    "minimum_candidate_weight_ppm": 0,
    "minimum_margin_milli": 0,
    "evidence_window_bytes": 128,
    "max_evidence_windows": 31,
    "max_evidence_spans": 31
  }
}
```

The source and claim remain exact UTF-8 data. The full-source head must fit the
admitted context; this command does not silently substitute retrieval, chunk
summaries or a truncated source. The evidence partition covers all source
bytes. It allows at most 31 windows plus the whole-source head. A source that
fits a single window reuses the whole-source head rather than scoring it twice.

`max_evidence_spans` is positive and cannot exceed `max_evidence_windows`.
Impossible or excessive partitions are rejected before opening model metadata.
The native finalizer retains uncertain and contrary windows, verifies reported
quote coordinates against the original source and can explicitly abstain. A
native failure is not the model relation `unsupported`. Source membership proves
that a quotation occurs; correlated model reads are not a factuality certificate.
All score distributions and judgments remain uncalibrated.

## Complete-source INT8 faithfulness

Use `candidate faithfulness-map` when the complete source exceeds one native
context or one evidence partition:

```sh
fnlp candidate faithfulness-map request.json \
  --model ./model.fnlpq --memory-mib 16384 \
  --max-chunks 64 --max-chunk-bytes 4096
```

The input is the same explicit `{mode: "faithfulness", source, claim, policy}`
object shown above. Only the source is partitioned. Every chunk receives the
unchanged claim, a global chunk judgment and all of its local evidence windows.
The planner calculates source capacity after accounting for the exact claim,
trusted scaffold and candidate continuation. It preserves all source bytes,
Unicode scalars and CRLF boundaries; it refuses an indivisible unit that cannot
fit instead of dropping or rewriting it. The ordinary one-context `judge`
command keeps its complete-source fit requirement.

The completed output retains each native chunk result and its original-source
byte/scalar span. Its aggregate lifts decisive supporting and contrary quotes
into original-source coordinates. Contrary evidence can force an explicit
`conflicting_evidence` abstention even when supporting evidence exists.
The separate `--minimum-classified-coverage-ppm` policy defaults to 1000000:
all original source scalars must belong to classified chunk judgments before
the document can meet that coverage floor. Insufficient coverage or unresolved
evidence produces an explicit abstention. The result declares
`single_context_equivalence: false`; separate contexts cannot establish an
inference that requires jointly reading different chunks.

Document limits are independent:

| Option | Default | Scope |
| --- | ---: | --- |
| `--max-input-bytes` | 1048576 | Complete JSON, including source and claim |
| `--context-tokens` | 2048 | Largest live native head |
| `--max-chunks` | 64 | Complete lossless source partition, maximum 256 |
| `--max-tokenizer-calls` | 8192 | All accepted and rejected partition probes |
| `--max-result-bytes` | 65536 | Complete native result for each chunk |
| `--max-retained-result-bytes` | 16777216 | Sum of retained complete chunk results |
| `--max-map-result-bytes` | 16777216 | Final document, chunks and evidence aggregate |
| `--preparation-mib` | 1024 | Modeled preparation reserve; additional ownership is priced separately |

All five native-work ceilings apply to the complete document across every
chunk and evidence head. Nothing renews per chunk. The forward-position default
is 262144; projected logits 10000000000; attention pairs 1000000000000; dot
products and multiply-accumulates each 10000000000000000. These are admission
ceilings, not measured work or performance estimates.

Compact source and claim encoding is explicit:

```sh
fnlp candidate faithfulness-map request.json \
  --model ./model.fnlpq --memory-mib 16384 \
  --input-encoding safe-bpe-v1
```

One bounded tokenizer ledger covers claim-capacity calculations, successful and
rejected source probes, and all final chunk/evidence-head plans in each complete
preparation. The four `--max-tokenizer-*` ceilings bound input bytes, output
tokens, pair checks and pair bytes. They do not reset between chunks or evidence
heads. Exact source/claim bytes remain untrusted data, and template-control token
IDs cannot enter their segments. The CLI preflight and later owned hosted
preparation use the same exact encoding policy, each with its own complete
preparation allowance. Output records the selected policy and actual hosted
tokenizer work. Compact prompts do not claim numerical or output equivalence.

The process quote admits source and claim ownership, every retained plan,
tokenizer workspace, complete chunk results, aggregate evidence, one physical
native cache and final output staging before input/model IO. Preflight completes
before weight loading. The owned host reconstructs the exact request under its
charge, checks every chunk before the first forward, and verifies sealed
completion before returning. Private request/result commitments bind the original
source, claim, model, policies, declared ceilings and payload; they are omitted
from JSON and diagnostic formatting. A late failure never returns a successful
document prefix. Output ownership survives checked serialization, write and flush.

Library callers can use `Int8FaithfulnessMapHostConfig::requirements` or
`requirements_with_encoding`, then `NlpEngine::map_int8_faithfulness` or
`map_int8_faithfulness_with_encoding`. The supplied base identity must already
match the resident model and exact pinned judge tokenizer/template. The host
does not repair mismatched identities or trust caller-supplied chunk scores.

This route supports one complete request. It does not expose corpus execution,
retained jobs, prompt-prefill scheduling or compute-team controls. The one-context
`judge` route and the INT4 faithfulness population routes retain their own options.

## Admission, delivery and limits

Shared finite-scoring options match `candidate classify` and `sentiment`:
2048 context tokens, 16 maximum candidate tokens including EOS, 65536 input
bytes and a 1 MiB complete native-result cap by default. Input is capped at
1 MiB; candidate provenance adds at most 4096 bytes to completed output.

All five computational-work limits cover the ENTIRE request across its heads:
`--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs`,
`--max-dot-products` and `--max-multiply-accumulates`. They retain the shared
scored-command defaults. Context is the largest live head, not the sum of all
forward work. Repeated prompt preparation remains covered by the existing
modeled preparation floor; the ledger is not an operating-system RSS limit.

Duplicate JSON keys, escaped duplicate aliases, unknown fields, invalid modes,
missing policies and invalid task data are refused before model metadata access.
Request JSON cannot supply a TaskBudget, tokens, templates, model identities,
work receipts, score-space overrides or tools. Source/criterion strings never
enter the trusted template renderer. The CLI supplies the finite task budget.

The same candidate Session reserves preparation before input or tokenizer
allocation. The pinned planner seals the complete request and checks all work
axes before weight loading. Loading compares the actual retained model facts;
execution uses the existing process host, resident model and native judge.
One failed comparison order, criterion or evidence head fails the whole request.

One cooperative elapsed budget follows input, preparation, loading and native
execution. Bounded compiler operations and blocking IO are not preempted midway.
Native and loader checkpoint allowances remain separately declared. Complete
JSON is staged before publication, and hosted output/preparation ownership
survives writing and flushing. A failed sink is not successful task delivery.
No provisional heads, private prompt fingerprints or nested parser diagnostics
are printed. Output retains `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`.

This remains a single-request command. For ordered corpora use
`candidate score-batch --task judge`; for explicit retained work use
`candidate score-job start/resume --task judge`. See
[candidate-judge-corpus.md](candidate-judge-corpus.md) for defaults, per-record
arguments and authenticated resume. Existing source/extraction and
classification/sentiment routes are unchanged. Without `asupersync-runtime`,
execution refuses before input or model IO.

## Compact input on the native INT4 routes

`fnlp candidate source-int4 judge` accepts `--input-encoding safe-bpe-v1`
for all three request modes. Its model input is an explicit pinned source or an
independently pinned INT4 snapshot. The one-context strict-INT8 `judge` and
complete-source `faithfulness-map` commands support compact input under their
separately described preparation and admission contracts above.

Compact input preserves exact untrusted bytes and excludes template-control
tokens. Evidence windows are independently encoded from original UTF-8 spans,
so a byte offset is never used to slice compact token IDs. All actual encoding
work in one judgment shares a single ledger, with four optional ceilings:

- `--max-tokenizer-input-bytes` (default 1048576).
- `--max-tokenizer-output-tokens` (default 1048576).
- `--max-tokenizer-pair-checks` (default 4194304).
- `--max-tokenizer-pair-bytes` (default 67108864).

These are cumulative preparation limits. Request size, per-head native context,
model work and result size remain separate constraints. Invalid or unused
encoding flags refuse before private input IO. The CLI admits one transient
encoder workspace; the host receives the already-prepared judgment and does
not tokenize again. Snapshot saving follows that same preparation.

`source-int4 score-batch --task judge` accepts the same encoding and all four
ceiling flags for every record, including explicit head cohorts and persistent
CPU teams. Its input-byte and output-token defaults remain the record-byte cap
and per-head prompt capacity. Larger explicit tokenizer ceilings may therefore
be needed for repeated preparation. Every frame reports the fixed policy.
Retained scoring policy and resume requirements are documented separately in
[candidate-scored-jobs.md](candidate-scored-jobs.md).

Compact encoding selects different prompt tokens. These candidate routes do
not claim identical model output, qualified task accuracy or measured speed.

## Validation scope

Sixteen added regression definitions cover strict parsing, policy units,
original text, rubric/evidence constraints, feature-disabled no-IO behavior,
all three real pinned planners, complete head counts, identity binding, context
reuse, every computational-work axis and cancellation. They are test sources,
not executed tests. Rust compilation/tests, full-model inference, numerical
parity, task quality, benchmarks and controller DSR were not run here.

The complete-source INT8 faithfulness addition includes 39 further regression
definitions across native planning/evidence reduction, hosted ownership/admission,
and public command/runtime boundaries. They cover lossless long sources, unchanged
claims, original evidence, conflict/coverage abstention, all five work axes,
compact tokenizer limits, private completion binding, strict request parsing and
all eleven typed cancellation causes. These definitions were independently
source-reviewed but remain unrun; DSR production compilation and model qualification
are still required.

The compact one-context INT8 additions include further regression definitions
for all three actual pinned planners, cumulative tokenizer work and exhaustion,
policy-bound identities, all scheduling combinations, optional output fields,
exact memory admission and original cancellation causes. These are source
definitions only; they have not been compiled or executed in this session.

## Persistent CPU teams for one complete judgment

`fnlp candidate judge` accepts `--compute-threads N` (1 through 64,
including the coordinator) and can combine it with `--prefill-rows ROWS`
(1 through 64). Defaults preserve serial execution. One fixed team spans every
pairwise order, rubric criterion or faithfulness/evidence head in the request;
it never publishes a subset after a later head fails.

```sh
fnlp candidate judge request.json --model ./model.fnlpq --memory-mib 8192 \
  --compute-threads 4 --prefill-rows 4
```

The entire threaded memory peak is checked before input or model IO, including
ordinary native storage, caller-owned preparation, coordinator prompt workspace,
fixed child capacity, output and cleanup. Runtime threads are additional.
The native route keeps full-vocabulary candidate denominators, EOS scores,
the complete work ledger and the existing task finalizer. Neither width nor
prompt scheduling changes the sealed task identity or its decision policy.

The library entrypoints are `execute_int8_judge_with_threads` and
`execute_int8_judge_layer_major_with_threads`. They return a guarded
`HostedInt8TeamResult<Int8JudgeRun>` after physical worker join, including
cancel/panic drain. The CLI adds completed physical work as a separate
`compute` object only for threaded execution; ordinary task output remains
unchanged. Batch, complete-source maps and retained jobs keep their own
explicit scheduling interfaces.

Regression definitions cover parsing, rejection before IO, all three modes'
unchanged identities and complete evidence windows, alongside the hosted
artifact-dependent differential gate. Compilation and executable tests remain
**unrun** pending DSR; no performance or model-quality claim follows from this
source integration.
