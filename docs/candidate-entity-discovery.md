# Automatic candidate entity discovery and resolution

`fnlp candidate resolve --discover-entities` accepts raw original documents,
extracts source-constrained named entities with the real strict INT8 model,
independently checks all occurrences, then resolves the complete snapshot with
the existing two-order scorer and conservative complete-link clustering.
The same model and native engine serve both stages. This is an explicit mode,
not a change to ordinary `candidate resolve` input or a rules-only substitute.

```sh
fnlp candidate resolve snapshot.json --discover-entities \
  --model ./model.fnlpq --memory-mib 8192
```

The build requires `asupersync-runtime` and a compatible explicit local candidate
artifact. There is no download, implicit model selection, release activation,
tool execution or numerical/recognition quality certification. Without the
feature, refusal occurs before source or model I/O.

## Raw snapshot input

One JSON object is read from a file or stdin (`-` or omitted). This is not NDJSON.

```json
{
  "documents": [
    {"id": "article-a", "text": "Alice leads Acme."},
    {"id": "article-b", "text": "Alice founded Beta."}
  ],
  "options": {
    "blocking": "ascii_word_overlap",
    "context_scalars": 32,
    "minimum_margin_milli": 1000
  },
  "ner": {
    "types": ["person", "organization", "location"],
    "max_entities": 64,
    "max_mention_scalars": 256
  }
}
```

`options` is required and selects the existing lexical/context/uncalibrated
margin policy. `ner` is optional; omission uses the shown existing defaults.
When supplied it must be a complete valid `NerOptions` object. Unknown fields,
duplicate JSON keys, duplicate document IDs, repeated NER types and input
scores, identities, mentions or per-document task overrides are refused.
The same spelling in the example does not establish that the people are equal.

Without `--discover-entities`, the existing explicit-mention contract remains
unchanged: each document supplies exact mention surfaces and byte/scalar spans.
The two modes cannot silently reinterpret one another's document records.

## One complete native operation

The CLI checks raw input and policy before model metadata access. It then builds
both actual pinned planners and preflights every exact NER request before loading
weights. The consumed prepared snapshot owns canonical-order original documents
and private complete-plan witnesses; source text cannot change during handoff.
Only one source grammar/index is rebuilt at a time, checked against its witness,
and executed on the existing native engine.

Every NER result is independently rescanned against its unchanged original.
All byte/scalar occurrence spans and anchored/ambiguous evidence must agree.
Repeated names become separate contextual mentions, not one guessed first match.
Duplicate proposals are checked before exact-span deduplication; occurrence
verification limits are shared across all documents. Documents with zero NER
proposals remain in final per-document receipts.

The candidate graph is not knowable before NER, so complete pair planning and
admission occur after extraction but before the first pair forward. Every pair
gets both presentation orders. All assessments complete before clustering, and
every cross-component comparison must explicitly agree before a merge. Missing,
conflicting and uncertain comparisons remain barriers. A failure at any stage,
a malformed receipt, cancellation or exceeded bound returns no successful corpus;
no intermediate mentions or provisional clusters are published.

Only an empty raw `documents` list can finalize without loading weights or
allocating native KV. Compatible model metadata is still required. Nonempty
raw inputs, including empty/name-free text, must run NER. After real NER, a graph
with no candidate pairs needs no pair inference; its resolution explicitly
reports `model_evaluated=false` even though the outer receipt records NER work.

## Whole-snapshot limits

`--max-ner-tokens` defaults to 256 tokens including EOS PER document. It is
independent of `--max-candidate-tokens`, which bounds pair-label scoring depth.
Every document and pair order must fit the admitted live context (2048 by
default); summed work across reset contexts is not a larger live context.

The five shared limits (`--max-forward-positions`, `--max-projected-logits`,
`--max-attention-pairs`, `--max-dot-products`, `--max-multiply-accumulates`)
cover NER AND resolution together. All conservative NER work is reserved before
inference. Pair scoring gets only the nonrenewable remainder, also intersected
with its stage cap; early EOS cannot give its unused NER reservation to pairs.
Raising document or pair counts does not multiply compute authority. In
particular, default 256-token NER reservations already charge 42,532,864 head
logits per document, so the default 100,000,000-logit ceiling can reject three
raw documents even before pair scoring. Explicitly budget the complete job.

`--max-ner-mask-node-visits` defaults to 1,000,000,000 per document;
`--max-snapshot-mask-node-visits` defaults to 1,000,000,000,000 across the snapshot.
Each mask is separately capped at 2,000,000 trie visits. NER count/mention-length
caps come from `ner`; expanded occurrence counts use `--max-mentions` (4096).
Independent occurrence recovery shares at most 1,000,000 fields, the configured
mention match ceiling and `--max-scan-steps` across the snapshot. Graph source
validation has its existing separate scan allowance.

`--max-input-bytes` remains the bounded wire-JSON ceiling (65536 by default).
`--max-expanded-bytes` defaults to 16 MiB and bounds original text/IDs plus all
expanded mention surfaces/types. It never permits a larger wire read. Graph
memory admission includes additional headroom for this expansion; raising the
expanded cap can require raising `--graph-reserve-mib` (64 by default).
Shared preparation defaults to 512 MiB; hosted execution separately retains
its transferred-input preparation commitment and graph/intermediate commitment.
Actual raw String/Vec capacities are charged, not just logical lengths. These
are modeled ledger commitments, not measured or OS-enforced RSS limits.

One cooperative elapsed budget follows input, preparation, loading, native
execution and delivery. One native checkpoint control spans all NER passes,
occurrence recovery and pair work; it is not restarted per document or stage.
Blocking I/O and individual bounded tokenizer operations are not preempted.

## Completed output and embedding

One candidate-provenance JSON envelope contains the complete `Int8EntityRun`:
per-document NER counts, all reserved/actual native work, mask charges,
verification work, and the full pair judgments/clusters. `--max-result-bytes`
bounds this entire native envelope (1 MiB by default), not only mention text.
Preparation and guarded-output memory ownership survive serialization, writing
and flushing. A transport failure can still truncate bytes and is an error.

Embeddings use `corpus::entities_int8::prepare_int8_entities`, then transfer the
consumed plan and pinned vocabulary to `NlpEngine::execute_int8_entities` with
explicit native, preparation and graph reservations. The existing process host
checks the resident domain, both model identities and full admitted KV capacity;
it creates one blocking invocation and does not introduce a second runtime.

Mention surfaces are sensitive output, not redaction or anonymization. Private
source/prompt witnesses and generated NER token transcripts are not exported.
Source membership proves location, not entity recognition recall/correctness.
Pair scores remain uncalibrated; cluster IDs remain snapshot-local. All output
retains `evidence=non_authoritative`.

## Validation status

The implementation adds 14 core regression definitions and 12 host/CLI
regression definitions, with feature-dependent subsets. They exercise real
pinned preparation, genuine empty finalization, private corruption fixtures,
Unicode occurrences, typed cancellation, all five work ceilings, strict raw
input, independent NER/pair limits and memory arithmetic. These are not
full-model success fixtures. Rust compilation/tests, native full-model runs,
recognition/resolution quality, benchmarks and controller DSR were not run.
