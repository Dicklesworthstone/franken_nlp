# Candidate long-document map/merge

`fnlp candidate map` processes one original document that does not fit a single
native context. It uses the existing lossless source partitioner, strict-INT8
source mapper, independent result validators and process-hosted model. It does
not concatenate a truncated prefix or require callers to manufacture chunks.

```sh
fnlp candidate map report.txt --task ner \
  --model ./model.fnlpq --memory-mib 8192

fnlp candidate map report.txt --task keyphrases \
  --model ./model.fnlpq --memory-mib 8192

fnlp candidate map report.txt --task summarize \
  --model ./model.fnlpq --memory-mib 8192
```

The runtime-enabled binary (`asupersync-runtime`) and compatible explicit local
candidate artifact are required. No model download, catalog activation, tool
execution, publisher authentication, quality award or release promotion occurs.
Without the runtime feature, refusal precedes input, options and model IO.

## What the output means

Output is one completed candidate JSON response. Its native result contains
ordered independent chunk results, chunk boundaries, source-local evidence,
original-document byte/scalar coordinates, actual and planned work, and the
existing map/reduce lineage. All output retains `evidence=non_authoritative`.
No partial document success is published after cancellation or a failed chunk.

NER does not claim a global entity census or cross-chunk entity resolution.
Keyphrases retain per-chunk rankings, not a global reranking. Summarization
retains independently cited chunk summaries, not a new global synthesis pass.
Non-overlapping chunk boundaries can split entities or semantic context; exact
source preservation does not establish recall or single-context equivalence.
Source membership is not semantic entailment. QA, arbitrary extraction schemas,
classification and generation are deliberately not accepted by this command.

## Exact capacity instead of a guessed scaffold reserve

The library method `SourceTaskPlanner::int8_map_capacity_with_control` renders
and tokenizes the same segmented, code-owned schema/template fragments as the
actual source task. It prices the selected typed options, reserves all possible
output tokens, and intersects the prompt and context ceilings. No placeholder
source, model call or assumed general-purpose characters/token ratio is used.
Its `Int8SourceMapCapacity::constrain_chunks` method only tightens caller limits.
All actual source chunks still pass the real source encoder and task compiler.

`--max-chunk-bytes` optionally adds a byte ceiling; omitted, it is chosen from
that task's available capacity. The existing partitioner verifies actual token
counts and can shrink chunks geometrically. It preserves every source byte,
Unicode scalar boundaries and CRLF pairs. Boundaries are not promised to be
linguistic sentences or the longest possible fitting chunks.

Before weights are materialized, the CLI makes a bounded partition/counting
pass over the real document, computes all five native work counters, and checks
the complete mask allowance. That temporary metadata plan is dropped. The host
then independently rebuilds the deterministic partition and compiles all actual
source-bound grammars before the first native forward. Full source-grammar
compilation is therefore pre-inference, not pre-weight-loading. Completed work,
chunk counts and whole-source coordinates must agree with the CLI preflight.
No second model, alternate tokenizer or inference retry is introduced.

## Limits and ownership

`--max-input-bytes` bounds the whole UTF-8 document (65536 by default, at most
1 MiB, with a minimum configured ceiling of four bytes). Source bytes are never
trimmed or normalized. Input may be a local file or omitted/`-` for stdin.
`--options FILE` supplies the selected task's complete existing options object,
capped at 16 KiB; duplicate keys, unknown fields and partial shapes are refused.

Context and per-chunk limits reuse the source command options: 2048 context
tokens, 512 maximum output tokens including EOS, and 1 MiB per native result.
The default maximum is 64 chunks, hard maximum 256; exceeding it fails rather
than losing the document tail. Larger chunk counts can require a larger explicit
`--preparation-mib`. The modeled preparation floor includes aggregate retained
prompt/grammar storage and complete-result staging, not only one chunk.

The default whole-map result cap is 16 MiB (`--max-map-result-bytes`), the live
serialized-value cap is 32 MiB (`--max-live-value-bytes`), and cumulative accepted
values are capped at 64 MiB (`--max-total-value-bytes`). These three caps have a
64 MiB hard maximum. Candidate provenance adds at most 4096 bytes. The native
host separately reserves transient result/reduction storage and explicit
`--reduction-reserve-mib` headroom, default 64 MiB. These are modeled process
ledger commitments, not measured or operating-system-enforced RSS ceilings.

The five work options are whole-document totals: `--max-forward-positions`,
`--max-projected-logits`, `--max-attention-pairs`, `--max-dot-products` and
`--max-multiply-accumulates`. Defaults are 262144, 10000000000, 1000000000000,
1000000000000 and 10000000000000000 respectively. Raising `--max-chunks` never
multiplies these allowances. `--max-mask-node-visits` applies per chunk, while
`--max-total-mask-node-visits` caps the document, default 64000000000. The entire
partition reserves those per-chunk bounds; early completion does not refund them.

One end-to-end cooperative deadline follows input, sizing, loading and native
execution. Tokenizer calls and chunk counts also have finite preparation bounds;
`--max-checkpoints` is the existing native-run allowance, not a claim of a shared
cross-stage checkpoint counter. Blocking IO and in-progress bounded tokenizer
operations are not forcibly preempted. One model load and one hosted map/merge
call serve the document. Native input/workspace drains before delivery; output
and preparation guards survive complete serialization, writing and flushing.

## Validation scope

Added test sources compare capacity and work preflight against real pinned
planners, including exact and one-over boundaries, UTF-8/CRLF preservation,
changed schemas, cancellations, independent work axes, cardinality refusal,
strict options and feature-disabled no-IO behavior. They do not simulate a
successful full-model inference. Rust compilation, Rust tests, model execution,
performance, task quality and controller DSR were not run in this environment.
