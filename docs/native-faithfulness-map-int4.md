# Native INT4 faithfulness over complete sources

`fnlp candidate source-int4 faithfulness-map` judges one exact claim against
every chunk of a complete source. It returns the original native judgments
and a separately named reduction with evidence in original-source coordinates.
It extends the bounded `judge` faithfulness route to sources that exceed one
model context, following [plan §§7.6 and 7.10](../COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md).

## Input and invocation

Use the same explicit faithfulness JSON shape as `judge`:

```json
{
  "mode": "faithfulness",
  "source": "The complete original source, including every section...",
  "claim": "The exact claim to assess against that source.",
  "policy": {
    "minimum_candidate_weight_ppm": 600000,
    "minimum_margin_milli": 100,
    "evidence_window_bytes": 256,
    "max_evidence_windows": 31,
    "max_evidence_spans": 31
  }
}
```

These thresholds are explicit, uncalibrated decision settings. They do not
assert that the selected relation has a particular probability of being true.
`minimum_margin_milli` is a log-score margin in thousandths, not a probability.
The evidence window and span limits apply independently to each complete
chunk judgment. The original long source need not fit one such partition.

For an independently pinned local candidate snapshot:

```sh
fnlp candidate source-int4 faithfulness-map request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib "$MEMORY_MIB" \
  --group-width 128 --context-tokens 2048 \
  --max-chunks 64 \
  --minimum-classified-coverage-ppm 1000000
```

Use `-` or omit the input path to read stdin. The input is one JSON object,
not NDJSON. The complete JSON request must fit `--max-input-bytes`; the hosted
library also bounds the combined source and claim bytes by the corresponding
mapping input cap. Both source and claim retain their exact UTF-8 bytes,
including whitespace and literal control-marker text under the existing
untrusted-source encoder.

The explicit `--source-dir` and `--source-manifest` pair can replace the
snapshot arguments. The command never downloads a model, selects an inferred
model identity, or falls back to another backend. Pairwise and rubric input,
unknown or duplicate keys, empty source or claim, invalid evidence policies,
and caller-supplied execution authority are refused. JSON cannot provide
tokens, prompts, model identities, budgets, score-space controls or scores.

In byte mode, an optional `--save-snapshot NEW_FILE` first compiles the complete document
against the loaded model and admits every future cohort window. The snapshot
uses the existing create-new policy; it can remain if subsequent inference or
publication fails. This explicit candidate snapshot does not activate a model
or authenticate a publisher.

## Compact source and claim encoding

Add `--input-encoding safe-bpe-v1` to select compact untrusted input explicitly.
Source, claim and evidence quotes preserve their original UTF-8 bytes, and
template-control IDs remain excluded. Chunk sizing uses actual compact token
counts. Every evidence window is independently encoded from its original text;
source byte offsets never index compact token vectors.

The four tokenizer ceilings are cumulative planning-work limits:

- `--max-tokenizer-input-bytes` and `--max-tokenizer-output-tokens` default to 1048576.
- `--max-tokenizer-pair-checks` defaults to 4194304.
- `--max-tokenizer-pair-bytes` defaults to 67108864.

One ledger includes claim-capacity preparation, every source sizing probe
(including rejected probes), final chunk judgments and their evidence windows.
A failure does not refund consumed work. Context, original input bytes, native
work, retained output and tokenizer calls keep their separate ceilings.
The compact completion's `input_encoding` record contains the exact policy
and work for that planning pass, and completion verification binds both.

Serial and native chunk-cohort execution admit one transient encoder workspace
beside retained plans. Evidence byte limits can constrain a chunk even when its
compact tokens fit; Unicode scalars and CRLF pairs remain indivisible, and a
span that cannot fit is refused without truncation or fabricated token counts.
Pinned source input and pinned snapshot reload are supported. Compact
`--save-snapshot` refuses before IO because its preliminary planning pass would
otherwise duplicate preparation. Byte fallback preserves the existing default
identity and output shape.

Compact input changes prompt tokens. It carries no model-output parity,
task-quality or performance claim.

## Source partition and native judgment

The complete original source is partitioned into disjoint UTF-8 chunks with
byte and Unicode scalar coordinates. There is no normalization, overlap,
retrieval filter or tail truncation. The unchanged claim and trusted prompt
scaffold consume capacity in every native head. A claim that leaves no room
for source bytes is refused. `--max-chunk-bytes` can impose a smaller bound.

Each chunk uses the existing three-way faithfulness language: `entailed`,
`contradicted` and `unsupported`. It scores the whole chunk plus its complete
bounded evidence-window partition; a one-window chunk reuses its whole-chunk
head. All candidate sequence scores retain full-vocabulary denominators and
scored EOS. A chunk's source is its own local context, so cross-chunk facts
are not combined into a new model judgment.

Every chunk plan is complete before the first native forward. The checked
work sum covers forward positions, projected logits, attention pairs, dot
products and multiply-accumulates across the entire source and every evidence
head. No allowance renews per chunk. An invalid later chunk or exhausted
document allowance fails the whole operation without a partial map.

The original chunk results preserve all heads and all uncertain, unsupported
and contrary windows. Their evidence spans stay chunk-local; the aggregate
separately lifts decisive witnesses into original-source coordinates and
checks their quoted bytes. Evidence quotes demonstrate source membership;
they do not prove entailment or correctness. If a chunk exceeds its explicit
evidence-span or result cap, it fails instead of silently pruning witnesses.

## Document evidence and decisions

The reduction profile is `faithfulness-complete-chunk-evidence-v1`. It uses
explicit witness, coverage and conflict rules. It does not average candidate
weights into a document correctness probability.

| Output under `output` | Meaning |
|---|---|
| `chunks` | Every complete native chunk judgment with its original byte/scalar range. |
| `aggregate.coverage` | Original scalar counts assigned to entailed, contradicted, unsupported and abstained chunk decisions. |
| `aggregate.classified_scalars`, `classified_coverage` | Source coverage belonging to chunks with a non-abstaining local decision. |
| `aggregate.evidence` | Every decisive entailed or contradicted evidence window, with chunk/head identifiers and original-source coordinates. |
| `aggregate.relation`, `abstention` | The explicit document decision or the reason it abstains. |
| `single_context_equivalence` | Always `false`; independent chunk context is not a whole-context model pass. |

Conflicting entailment and contradiction evidence blocks acceptance and
remains visible even when the result abstains. A local conflict also blocks
acceptance. Otherwise the configured classified-coverage requirement must
pass. An accepted entailed or contradicted chunk supplies the corresponding
document witness; decisive evidence without a compatible accepted chunk
produces an unresolved-evidence abstention. With no such evidence, accepted
unsupported chunks describe lack of support in the inspected source.

`--minimum-classified-coverage-ppm` defaults to `1000000`, requiring all
original Unicode scalars, including whitespace, to belong to classified
chunks. Its range is 0 through 1000000. A lower value relaxes coverage but
never suppresses contrary evidence; at least one chunk must still classify.
Multibyte characters do not receive extra coverage weight from their larger
UTF-8 encoding. Changing chunk geometry can change the result because it
changes the local context read by the model.

## Serial and native cohort execution

Serial execution is the default and reuses one native cache. Add
`--cohort-chunks 4`, for example, to share native projections across windows
of four complete chunks. Every head retains independent KV and the original
judgment finalizer. The complete semantic result remains under `output`;
cohort responses add a sibling `cohort` record with physical grouping counts.

Faithfulness chunks have variable head counts: one for a one-window chunk,
or a whole-chunk head plus every evidence-window head. Every actual future
cohort window must fit at most 64 simultaneous heads. Oversized windows fail
before execution, and before optional snapshot creation, without shrinking
the requested schedule or switching to serial execution.

Before the source is known, the memory quote prices the maximum admissible
window: the smaller of 32 times the requested chunk width and 64 active
heads. This is an allocation ceiling, not a change to scheduling. The sealed
plan checks the actual ragged geometry. Logical per-chunk task authority is
identical in serial and cohort modes. The cohort record is not a measured
speedup, fidelity result or task-quality claim.

## Memory, deadlines and publication

The command admits its modeled peak memory before input or model IO. It
includes input parsing, moved source and claim ownership, retained plans and
chunk results, native KV and scratch, aggregate witnesses, complete JSON
publication, model loading/saving and runtime cleanup. The hosted route uses
the same pure quote and also charges excess capacity retained by caller-owned
strings. The process ledger tracks allocation ownership; it is not an
operating-system RSS cap.

The main controls are `--max-input-bytes`, `--context-tokens`, `--max-chunks`,
`--max-chunk-bytes`, `--max-tokenizer-calls`, the five native work ceilings,
`--max-result-bytes` per complete chunk, `--max-retained-result-bytes` for all
retained chunk records, and `--max-map-result-bytes` for the complete semantic
document. Candidate framing has a separate fixed allowance. The finite
cohort wrapper and simultaneous chunk results also have checked caps; a
declared window is never silently reduced to fit them.

`--timeout-seconds` covers input, model loading, optional snapshot preflight
and saving, inference, aggregation and publication. Native checkpoints retain
typed cancellation through physical drain. Blocking IO observes the deadline
between operations. Serialization must finish before the external write
begins; the guarded result and staging charge remain owned through physical
write, flush and the final deadline check.

Planning, execution and aggregation failures publish no partial inference
result. A write or deadline failure during physical publication can leave
complete or truncated bytes and still returns failure. Consumers must require
exit 0 as well as a complete JSON object.

## Validation scope

Execution requires `asupersync-runtime`; help, parsing and pure admission are
available without it. The implementation includes model-free regressions for
strict input, source/snapshot selection, memory ownership, cancellation,
complete evidence reduction and publication. Repository validation remains
DSR-only. This route preserves candidate provenance and makes no model-parity,
quality, performance or release-qualification claim.
