# Candidate cross-document entity resolution

`fnlp candidate resolve` scores source-anchored entity mentions across an entire
bounded document snapshot using the existing strict-INT8 two-order scorer and
conservative complete-link clustering. It does not mistake lexical similarity
for identity or apply a transitive merge through uncertain/missing comparisons.
The binary requires `asupersync-runtime`, a compatible explicit local candidate
model and a process memory ceiling.

```sh
fnlp candidate resolve snapshot.json --model ./model.fnlpq --memory-mib 8192
```

This page describes the default explicit-mention mode. For raw documents, use
`--discover-entities`; see [automatic entity discovery](candidate-entity-discovery.md)
for its NER input, whole-operation limits and no-model behavior.

The input is one JSON object (not NDJSON). Omit the path or use `-` for stdin.
An explicit options object selects the blocking and uncalibrated margin policy:

```json
{
  "documents": [
    {
      "id": "article-a", "text": "Alice leads Acme.",
      "mentions": [{
        "entity_type": "person", "surface": "Alice",
        "span": {"byte_start": 0, "byte_end": 5, "scalar_start": 0, "scalar_end": 5}
      }]
    },
    {
      "id": "article-b", "text": "Alice founded Beta.",
      "mentions": [{
        "entity_type": "person", "surface": "Alice",
        "span": {"byte_start": 0, "byte_end": 5, "scalar_start": 0, "scalar_end": 5}
      }]
    }
  ],
  "options": {"blocking": "ascii_word_overlap", "context_scalars": 32, "minimum_margin_milli": 1000}
}
```

The example describes a possible comparison, not an assertion that the two
people are the same. `minimum_margin_milli` is a positive threshold in
thousandths of natural-log score difference; it is not confidence in identity.
Valid thresholds range from 1 through 1000000. `context_scalars` ranges from
0 through 2048. This is a caller-selected policy, not a calibrated recommendation.

## Exact sources and bounded comparisons

Every mention supplies an exact surface, type, and half-open UTF-8-byte and
Unicode-scalar coordinates. The resolver checks all four coordinates and the
source slice; it does not normalize or fuzzy-relocate an invalid anchor.
Document IDs must be unique. Duplicate mention records, unknown JSON fields,
duplicate keys (including escaped aliases), invalid policies and oversized
graphs are rejected. Input cannot provide model identities, scores or budgets.

Without `--discover-entities`, mentions are explicit inputs; this mode does not
discover omitted mentions or silently run NER. Existing NER results can supply their exact occurrence
spans: expand every occurrence into a separate mention and retain the unchanged
original document. Source membership proves location, not correct recognition.

`blocking` is `exact_surface` or `ascii_word_overlap`. Both preserve exact type
separation. The latter uses the existing ASCII-word/initialism heuristic, not
Unicode normalization or an alias-completeness guarantee. Every candidate pair
gets both presentation orders and the complete same/different/uncertain language,
including EOS and full-vocabulary denominators. A matching name alone is never
a merge instruction. Both orders must pass the explicit margin policy.

All pair assessments complete before clustering. A merge requires EVERY pair
across the two components to be explicitly same; missing, different or uncertain
comparisons block it. Output retains mentions, complete pair judgments, clusters,
blocked merges, limits-related accounting and the existing warnings. Cluster IDs
are deterministic only within this exact snapshot, not global persistent IDs.
Output contains sensitive mention surfaces and is not redacted or anonymized.

## Preflight and no-model results

All source validation and lexical candidate enumeration precede model metadata
access. The entire pair set and both exact prompts per pair are then compiled
and admitted before weight loading. The same owned documents and pinned planner
enter one existing hosted invocation, which independently revalidates them
before its first forward. There is one model load/engine, not one per pair.
Any failed head, invalid result or cancellation rejects the whole snapshot.

When there are no candidate pairs, the genuine source-validated graph finalizes
without any weight load, KV allocation or inference. The output explicitly sets
`model_evaluated=false`, zero heads and zero native work; mentions remain
singletons. A nonempty comparison plan cannot use this shortcut. Compatible
model metadata is still required to bind the candidate request. This is not a
lexical merge or a simulated model result.

## Resource and delivery contract

The shared scored options default to a 2048-token context, 65536 input bytes
and a 1 MiB complete result. The input-byte ceiling includes JSON syntax and is
at most 1 MiB. Whole-snapshot native limits retain the shared scored defaults:
`--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs`,
`--max-dot-products` and `--max-multiply-accumulates`. Context is the largest
live head, not the sum of forwards across reset contexts.

Graph defaults are 256 documents, 4096 mentions, 256 candidate pairs, 1000000
pair-enumeration visits (including duplicate lexical-block visits), 10000000
clustering checks and 268435456 source-validation scan units. Flags are
`--max-documents`, `--max-mentions`, `--max-pairs`, `--max-pair-visits`,
`--max-cluster-checks` and `--max-scan-steps`. Raising pair/count bounds never
multiplies compute authority; the default compute ceiling may refuse a graph
below its cardinality cap. A required comparison beyond a cap fails rather
than disappearing. `--max-pairs 0` permits only graphs needing no comparisons.

Source surfaces are capped at 1024 bytes, lexical keys at 32 per mention and
block membership at 512. The explicit graph reservation (`--graph-reserve-mib`,
64 default) has a checked floor for snapshot/graph/ticket/result storage, in
addition to shared preparation. Preflight releases its graph before the host
constructs its own; no second whole graph is silently free. These are modeled
ledger commitments, not measured or OS-enforced RSS limits.

One cooperative elapsed budget spans reads, preflight, loading and execution.
Native cancellation/work limits cannot refresh at a pair or order boundary.
Blocking I/O and bounded tokenizer calls are not preempted mid-operation.
Completed output is staged before writing; memory authority survives write and
flush, including the no-model path's preparation-owned result. Failures use
closed diagnostics without source excerpts. Every result retains candidate
provenance and `evidence=non_authoritative`; nothing activates a catalog or
promotes numerical fidelity, recognition quality or identity confidence.

## Validation scope

Added test sources exercise exact Unicode coordinates, invalid anchors,
duplicate IDs and JSON keys, explicit policy, all work axes, graph refusal,
real pinned two-order planning, cancellation and genuine no-comparison
finalization. They do not simulate neural success. Rust compilation/tests,
full-model runs, resolution accuracy, benchmarks and controller DSR were not
executed for this implementation.
