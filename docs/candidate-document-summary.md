# Complete cited document summaries on the native INT8 candidate

`fnlp candidate map --task summarize --reduce-summary` turns one long original
UTF-8 document into a single cited summary. It runs real native INT8 summary maps,
unions their exact bullet text and evidence, and ranks the complete bullet set
before applying the final cap. Ordinary `candidate map` remains unchanged and
returns independent ordered chunk results unless this flag is present.

```sh
fnlp candidate map report.txt --task summarize --reduce-summary \
  --summary-bullets 16 --model ./model.fnlpq --memory-mib 8192
```

This requires a build with `asupersync-runtime` and a compatible explicitly
selected local candidate artifact. Omit `report.txt` or use `-` for stdin. Input
is exact plain UTF-8, not a JSON task envelope or NDJSON. Empty input is refused.
The mode is only valid with `--task summarize`; NER and keyphrase maps cannot
silently enter a summary reducer. No model discovery or download is performed.

## Native maps, then evidence-preserving reduction

The pinned task scaffold and source encoder determine the actual chunk capacity.
The existing partitioner covers every original byte exactly once, preserving
Unicode scalar and CRLF boundaries. It does not promise sentence boundaries,
overlapping windows or recognition of context that crosses chunk boundaries.
The CLI checks the complete partition and all five native work axes before
weight loading. The host compiles and admits every actual chunk plan before
its first forward, using one process-owned model, engine and invocation.

Each chunk produces the existing source-cited summary result. Reduction
independently rescans every quoted citation against the unchanged chunk. All
byte/scalar coordinates and anchored/ambiguous occurrence lists must agree.
All verified occurrences are lifted to original-document coordinates. A broken
last citation fails the entire result, including when that bullet would lose
final selection. Duplicate proposals are verified before deduplication.

Only EXACT bullet text is deduplicated; different wording and contradictions
remain distinct. Duplicate occurrences of a bullet in the same chunk count as
one supporting chunk, retaining the earliest local rank and the union of exact
quotes. Every candidate and its evidence survives intermediate reductions, or
the operation fails its configured limits. Top-k is applied only once, after
all chunks are processed. For the same successful map results, final selection
is independent of the reduction tree's fan-in.

Ranking uses the number of distinct supporting chunks (descending), sum of
local ranks (ascending), earliest original evidence offset, then exact text.
These are deterministic heuristics, not calibrated importance or confidence.
The output records omitted bullet counts and full mapped source coverage,
including chunks whose bullets are not selected. Empty bullet maps still count
as processed chunks and still contribute actual native work.

**This is not a second neural synthesis pass.** It does not rewrite bullets,
resolve semantic contradictions or establish that quotes entail their bullets.
Structural source membership is verified; semantic support remains
`not_assessed`. Single-context equivalence and summary quality remain unproven.

## Separate local and whole-document limits

The existing `--options FILE` contains complete native `SummaryOptions` applied
to EACH chunk: `max_bullets`, `max_bullet_scalars`, `max_citations_per_bullet`,
and `max_quote_scalars`. The new `--summary-bullets` sets the final document-wide
cap, default 16. It does not reduce the per-chunk native output-token reservation
or allow an oversized intermediate union to be truncated into apparent success.

The opt-in reduction flags are:

| Flag | Default | Scope |
| --- | ---: | --- |
| `--summary-bullets` | 16 | Final selected bullets, maximum 1024 |
| `--max-unique-summary-bullets` | 4096 | Entire pre-selection exact-text union |
| `--max-summary-citations` | 16384 | Retained chunk/bullet citations across the complete union |
| `--max-summary-evidence-spans` | 65536 | Original-source occurrence spans across the complete union |
| `--max-summary-scan-steps` | 536870912 | Independent citation-verification work shared across all maps |

These flags require `--reduce-summary`; they are not silently ignored. Byte
bounds remain the existing map controls: `--max-map-result-bytes` bounds each
reduction value, the unselected root envelope and the final complete native
summary (16 MiB default). `--max-live-value-bytes` bounds simultaneously retained
serialized values (32 MiB default); `--max-total-value-bytes` bounds cumulative
accepted values across the tree (64 MiB default). A small final result cannot
waive these complete-union limits. `--max-result-bytes` still bounds one native
chunk result. Candidate provenance adds the existing 4096-byte envelope allowance.

Native forward positions, projected logits, attention pairs, dot products and
multiply-accumulates retain the existing WHOLE-document ceilings. Mask work
remains bounded per chunk and for the whole invocation. Selecting one final
bullet instead of sixteen does not grant more work or fewer required maps.
Reduction itself makes no model calls and cannot renew the native allowance.

## Process ownership and embedding

`NlpEngine::summarize_int8_source` accepts the resident `ResidentInt8`, an owned
source `String`, pinned planner/vocabulary Arcs, the existing `SourceMapConfig`,
`Int8SummaryLimits` and one cancellation token. Only a summary task is accepted.
The same process host checks model identity, runtime domain and full resident
KV capacity. It charges actual source capacity and explicit preparation costs,
then reserves the complete native/evidence frontier and final output separately.
The reservation is not sized only to the final top-k. Native allocations drain
before their memory guards; the result's output guard survives external delivery.
These are modeled ledger commitments, not measured or OS-enforced RSS limits.

The lower-level consumed `PreparedInt8SourceMap::execute_summary_with_control`
uses the same real INT8 driver and receipt checker as ordinary source maps.
The shared reducer has separate BF16 and INT8 validation paths; no eager result
is relabeled. One cooperative controller spans maps, coordinator checkpoints,
reduction and final output checks. Blocking I/O and individual bounded scans,
serializations or tokenizer operations are not preempted mid-operation.

No source digest, prompt fingerprint or generated-token transcript is exported
by the complete-summary envelope. Bullet text and quotes are sensitive untrusted
model/source data, not telemetry. CLI output retains candidate provenance and
`evidence=non_authoritative`. No numerical, artifact-activation or quality gate
is promoted by this new composition. Failed native work, evidence checks or
budgets return no successful partial summary. Transport errors can still leave
partial bytes on the external stream and are reported as failures, without retry.

## Validation scope

The core adds 12 pinned-planning/private scripted regression definitions. The
host and CLI add 14 more definitions, including a feature-disabled test. These
cover exact Unicode citations, late support, tree-independent selection, full
union limits, corrupt evidence/receipts, source coverage, typed cancellation,
mode isolation, real pinned planning and resource arithmetic. Scripted results
are private corruption fixtures, not neural-success evidence.

The required `rch` test invocation could not start in this editing environment
because `rch` is absent. Source hashes, targeted diffs, whitespace and lexical
delimiter checks were performed; Rust compilation/tests, full-model inference,
summary-quality evaluation, performance measurements and controller DSR were
not executed here. No passing build or neural-quality claim is made.
