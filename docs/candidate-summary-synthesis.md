# Candidate document-summary synthesis

`candidate map --task summarize --synthesize-summary` implements an explicit
additional native INT8 summary pass over the verified verbatim quotations from
all chunk summaries. The default independent map and `--reduce-summary` exact
bullet/evidence union remain unchanged. The two reduction flags conflict.

This is a code-first, non-certified candidate capability. Its regression tests
are definitions, not retained execution results. Compilation, model-present
correctness, summary quality and production qualification still require the
controller-owned validation checkpoint and their separate evidence gates.

## Invocation

With an `asupersync-runtime`-enabled binary and an explicit local current-candidate
INT8 artifact, the command shape is:

```sh
fnlp candidate map report.txt \
  --task summarize --synthesize-summary \
  --options summary-options.json --synthesis-bullets 6 \
  --model /path/to/local.fnlpq \
  --memory-mib 16384 --preparation-mib 1024 \
  --context-tokens 8192 --max-new-tokens 512 \
  --max-input-bytes 1048576 --max-chunks 32 \
  --max-synthesis-evidence-bytes 8192
```

The numbers are example admission ceilings, not benchmark results or a promise
that a particular document fits. Memory is modeled by the process ledger, not
an operating-system RSS cap. Actual source-token, scaffold, grammar, whole-run
compute, mask and result limits also apply. An example complete options file is:

```json
{"max_bullets":2,"max_bullet_scalars":256,"max_citations_per_bullet":1,"max_quote_scalars":64}
```

These options govern each map; the final pass inherits them except for the
optional `--synthesis-bullets` override. All final options bind its actual schema.
Input text and quotes remain untrusted data, never trusted template instructions.

## Execution and output

Every original chunk is planned before weights load. The host reserves the
complete map work plus a maximum-context final pass, rechecks every admitted
identity and complete resident KV capacity, and runs both stages on one engine
and cancellation controller. The actual evidence prompt is separately compiled
and admitted before the final native forward.

Only exact source quotes enter the evidence document. Generated map assertions
never become source facts. Duplicate proposals are verified before deduplication;
equal quotes from different chunks retain their distinct origins. Final citations
are independently rescanned and lifted only through collected source evidence.
A citation crossing a synthetic join is refused even when identical text occurs
elsewhere in the original. Overlapping origin fanout is charged before duplicate
coordinates are removed.

The candidate response's `output` retains `discovery` (all original chunk results),
`synthesis_native` (the unchanged native result with evidence-document-local
coordinates), and `bullets` (final text with original-document byte/scalar
coordinates). Its `status` distinguishes `no_evidence_collected`,
`no_bullets_produced`, and `synthesized`. Native and reserved work, mask charges,
evidence sizes and independent verification usage remain visible.

## Limits and interpretation

`--max-synthesis-evidence-segments` bounds distinct chunk/quote segments.
`--max-synthesis-evidence-bytes` includes the two-newline separators.
`--max-synthesis-fields`, `--max-synthesis-matches`, and
`--max-synthesis-scan-steps` bound one shared additional independent verification
ledger across quote collection and final citation lifting. Scan matches and
original-coordinate fanout both consume the match allowance. Native per-pass
validation retains its existing separate caps. None of these allowances renews
with another chunk or the synthesis pass.

This implementation performs **one bounded final synthesis pass**, not recursive
compression. Every collected quote must fit; oversized evidence or an actual
prompt/context overflow fails without ranking away evidence, truncation, repair
retry or partial document success. An empty evidence set does not claim that the
original contains no useful information. Citation existence establishes structural
source membership, not semantic entailment; output remains `non_authoritative`
with semantic support unassessed. Full-document recall and single-context
 equivalence are not established.

## Explicit multi-context alternative

Adding `--hierarchical-summary` opts into a separate bounded, lossy multi-level
quote-selection policy. It does not change this single-pass contract or silently
retry failures. See [candidate-summary-hierarchy.md](candidate-summary-hierarchy.md)
for its pass/depth reservations, exact-token grouping, retained lineage and
original-document citation guarantees.
