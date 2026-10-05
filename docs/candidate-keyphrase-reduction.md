# Candidate document-wide keyphrase ranking

`candidate map --task keyphrases --reduce-keyphrases` runs the native INT8
keyphrase task on every source chunk, verifies all proposed occurrence evidence,
and publishes one document-wide ranking. Default maps remain independent chunk
results. This reduction is deterministic exact-text aggregation, not a second
model pass, neural reranking, stemming, or approximate candidate merging.

## Command

With the `asupersync-runtime` feature and an explicit local candidate artifact:

```sh
fnlp candidate map report.txt \
  --task keyphrases --reduce-keyphrases --document-keyphrases 12 \
  --model /path/to/local.fnlpq \
  --memory-mib 16384 --preparation-mib 1024 \
  --context-tokens 8192 --max-new-tokens 256 \
  --max-input-bytes 1048576 --max-chunks 32
```

These are example admission ceilings, not measurements or a promise that every
input fits. `--options FILE` still supplies each native chunk's complete options,
for example `{"max_phrases":8,"max_phrase_scalars":128}`. The separate
`--document-keyphrases` cap controls only final publication.

## Ranking and evidence

All native candidates survive every intermediate merge, or the complete run
fails its bounds. A candidate ranked second in every chunk can therefore win the
final top-one selection over different chunk-local winners. Ranking is by
supporting-chunk count descending, local-rank sum ascending, first original byte
offset ascending, then exact text bytes. Repeated occurrences within one chunk
add evidence, not extra votes. Case and Unicode normalization remain unchanged.

Each retained phrase records source chunk IDs, local ranks, and every verified
original-document byte/scalar occurrence reported within those chunks. There is
no invented span across chunk boundaries and no assertion that an unselected
occurrence in some other chunk was recognized by the model.

The result includes the full mapped source extent, complete mapped-chunk count,
omitted-candidate count, reduction statistics, numerics and ranking identities,
all five planned/actual model-work axes, mask charges and independent scan work.
Empty model selections remain completed maps; an empty original input is refused.

## Admission and failure

`--max-unique-keyphrases` bounds the entire pre-selection union (default 4096).
`--max-keyphrase-evidence-spans` bounds complete original-coordinate evidence
(default 65536). `--max-keyphrase-scan-work` bounds one independent verification
allowance over all chunks (default 536870912). These do not renew per chunk.
The existing map-value, live-value, cumulative-value and final-result byte limits
also apply; a tiny final top-k never hides an oversized candidate union.

The hosted `NlpEngine::keyphrases_int8_source` path retains process memory claims
for preparation and the complete reduction frontier, checks every actual native
identity and full resident KV capacity before inference, and uses one resident
engine and cancellation controller. Output ownership persists through CLI write
and flush. No retry, fallback profile, partial-document success, or second runtime
is introduced. The eager corpus constructor remains eager-only.

## Qualification boundary

This is a code-first candidate feature. Added regression tests are definitions,
not retained execution results. Compilation, model-present behavior and relevance
quality still require their controller-owned validation checkpoints. Output stays
`non_authoritative`. Source membership does not establish relevance; chunk
boundaries may split phrases, model selection has no recall guarantee, and
single-context equivalence is not established.
