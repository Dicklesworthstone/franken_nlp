# Candidate hierarchical summary synthesis

`candidate map --task summarize --synthesize-summary --hierarchical-summary`
adds explicit, bounded, **lossy quotation selection across multiple contexts**.
It removes the single-final-context bottleneck of the one-pass synthesis mode.
It is not an automatic fallback, an exact summary union, or full-document recall.
The existing independent map, `--reduce-summary`, and single-pass
`--synthesize-summary` contracts remain separate and unchanged.

This is a code-first candidate implementation. Regression definitions are not
executed receipts. Compilation, native model-present correctness, performance,
semantic support and summary quality remain subject to their controller-owned
validation and evaluation gates. Output remains `non_authoritative`.

## Invocation

An `asupersync-runtime`-enabled binary and an explicit local candidate INT8
artifact are required. Example command shape:

```sh
fnlp candidate map report.txt \
  --task summarize --synthesize-summary --hierarchical-summary \
  --options summary-options.json --synthesis-bullets 4 \
  --summary-max-levels 8 --summary-max-passes 16 \
  --max-synthesis-evidence-bytes 131072 \
  --model /path/to/local.fnlpq \
  --memory-mib 16384 --preparation-mib 1024 \
  --context-tokens 8192 --max-new-tokens 512 \
  --max-input-bytes 1048576 --max-chunks 32 \
  --max-forward-positions 524288
```

Example complete `summary-options.json`:

```json
{"max_bullets":2,"max_bullet_scalars":256,"max_citations_per_bullet":1,"max_quote_scalars":64}
```

These are admission ceilings, not benchmarks or a guarantee this configuration
fits a particular input. All five model-work axes, masks, memory, tokenization,
verification and result-size ceilings still apply. `--synthesis-bullets` applies
to every intermediate and final synthesis call; other options are inherited.

## Reduction policy

The existing lossless partitioner and pinned task compiler prepare every source
map before native execution. Before weights load, the CLI also reserves a
maximum-context native work slot for **every allowed reduction pass**, not merely
for passes that happen to execute. The host repeats admission and retains one
resident model, engine, cancellation controller and process-resource lease.
Each actual dynamic evidence prompt is separately compiled and admitted before
its own forward pass.

Verified map quotations form an ordered evidence frontier. At each level, a
source-order greedy grouping counts the **actual joined source tokens** using
the same pinned source encoder and exact synthesis scaffold. Each quote remains
atomic. The first oversized extension ends a group; no token-count monotonicity
assumption or characters-per-token estimate is used. All groups are determined
before the level's first native call, and every frontier segment enters exactly
one group. A single group is the final synthesis pass.

For a multi-group level, every group's native summary is independently checked
and its citations lifted to the original document. Only those verified **quote
strings and reachable original occurrences** become the next frontier. Generated
bullet assertions never become source facts. Equal quotes from separate groups
retain their origins; duplicate proposals within a group are checked before
being deduplicated. Source occurrences elsewhere are not newly invented as
support. A final citation crossing a synthetic join is refused even when equal
text exists elsewhere in the source.

This selection is deliberately lossy: evidence not cited by an intermediate
summary may be absent from later levels. Every intermediate native result and
its original-coordinate bullets remain available for inspection. A nonempty
next frontier must be strictly smaller in bytes, including separators, or the
whole run fails rather than looping or silently truncating it.

## Output and limits

The native hierarchy result retains `discovery`, all `passes`, and `levels` with
contiguous input-segment and pass ranges. Each pass contains its unchanged native
receipt (group-local coordinates) and independently lifted `bullets` (original
UTF-8 byte and Unicode-scalar coordinates). `final_pass` indexes the one final
single-group pass; `passes[final_pass].bullets` is the final summary. The Rust
`Int8SummaryHierarchyRun::bullets()` accessor avoids copying that payload.

`no_evidence_collected` means discovery yielded no quotations. `no_bullets_produced`
means the final pass emitted no bullets or all intermediate groups selected no
remaining evidence. It does not mean the original document has no useful content.
`synthesized` requires a completed final single-group pass. Completed results
retain actual/planned/reserved model work, masks and verification/tokenizer usage.

The defaults are eight levels, sixteen total reduction calls, 8192 additional
grouping tokenizations and 67108864 cumulative bytes submitted to those grouping
counts. The options are `--summary-max-levels`, `--summary-max-passes`,
`--summary-max-tokenizer-calls`, and `--summary-max-tokenizer-bytes`; all require
`--hierarchical-summary`, which itself requires `--synthesize-summary`.

Existing synthesis evidence caps bound the complete initial and each subsequent
frontier. Existing verification fields, matches and scan work form **one shared
additional ledger** across collection, all citation lifts and inter-level
transport. Native per-pass validation retains its separate existing bounds.
Grouping counters measure additional sizing work, not task-compilation work.
The host prices all retained native results and transcripts, both evidence
frontiers and shared occurrence fanout; modeled memory is not an OS RSS cap.

An atomic quote that cannot fit one context, a nonshrinking level, unavailable
remaining pass/depth budget, or any shared budget exhaustion refuses the entire
run. No partial document success, alternate model, unconstrained repair or
single-pass retry is published. Citation membership remains structural only;
semantic entailment, factual completeness and single-context equivalence are
not established.
