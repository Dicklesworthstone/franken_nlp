# ADR 0001: OQ-31 FNLPQ envelope review

Status: ratified design record; writer and reader implementation remains
separate and batch-verification is pending.

## Decision

FNLPQ v1 uses a fixed 80-byte prelude, canonical hash-checked header JSON, a
fixed-size authenticated binary section table, and ordered binary payloads.
The binary prelude and table own all byte ranges. The header owns identities and
cross-references but never an alternate range or raw binary representation.
The complete frozen contract is
[`docs/fnlpq-envelope-v1.md`](../fnlpq-envelope-v1.md); its successor fixtures
are `tests/fixtures/fnlpq/field_inventory.json` and
`tests/fixtures/fnlpq/hostile_cases.json`.

This review considered the envelope as hostile input, independently of an
implementation. It preserves the three-family release dependency policy: v1
adds no compression, base64, Unicode-normalization library, or format-specific
dependency. Authority-bearing names use a printable-ASCII grammar.

## Decisions and killed alternatives

| OQ-31 clause | Decision | Killed alternative and reason |
|---|---|---|
| Authority tradeoff | Prelude, header digest, and fixed binary section table are the range authority. JSON holds no raw binary or offsets. | JSON-only/base64 payload directory: duplicates range authority, costs memory, admits ambiguous decoding, and bypasses prelude-first bounds checks. |
| Digest taxonomy | Use six framed names: `source_root_sha256`, `logical_model_sha256`, `packing_set_sha256`, `license_bundle_sha256`, `fnlpq_file_sha256`, and `release_manifest_sha256`. `semantic_digest` is forbidden. | One catch-all semantic digest: cannot distinguish model semantics from source, legal notice, packing, or physical bytes, so it makes a notice-only correction appear semantically material. |
| Duplicate and numeric policy | Duplicate JSON keys, tensor names, section names, and required singleton section kinds reject. Header JSON is one canonical UTF-8 form; floats are fixed IEEE bits in binary scale sections and NaN/Inf reject. | Permissive parser/last-key-wins and JSON float scales: two decoders can authenticate different meanings, and host/serializer float spelling becomes an authority surface. |
| Packing multiplicity | Multiple representations are allowed only in a named PackingSet with checked explicit aggregate bytes; dispatch chooses only a matching installed representation. | One implicit representation or arch fallback: hides duplicate byte cost and can silently execute an unmeasured or wrong-ISA payload. |
| Malformed limits | Validate fixed prelude before allocation, cap header/table/tensor/rank/dimension/mapped-byte sizes, use checked `u64` ranges, require regular-file length equality, reject unknown required flags/enums, and permit only zero alignment gaps. | Parse-then-validate and best-effort newer-file handling: turns hostile lengths, overlap, and unknown semantics into allocations or partially trusted data. |

## Digest-scope consequences

Native repacking changes the packing-set, physical-file, and release-manifest
identities but cannot change `logical_model_sha256`. A correction confined to
the Apache-2.0/attribution/modification-notice bundle changes only the legal,
physical-file, and release-manifest identities. The legal bundle is a legal
identity, not model semantics. Whole-file and release-manifest digests are kept
outside their respective bytes to avoid self-reference.

## Evidence artifacts

The field inventory is the writer's golden-serializer contract. The hostile
checklist is the reader's fuzz-corpus contract. Their SHA-256 values are
recorded after the review artifacts are finalized:

| Artifact | SHA-256 |
|---|---|
| `docs/fnlpq-envelope-v1.md` | `fe580cb136d9a7fcda6311d9b1a6ef148f88e65c656be74b72649c71bf25022f` |
| `tests/fixtures/fnlpq/field_inventory.json` | `1766c97e559eb01db42386a04087edc0657f7da0a7728ba0f285c33615fef624` |
| `tests/fixtures/fnlpq/hostile_cases.json` | `8bd51d9b57b3ae272600fc85aec48817713ba3b307263456c14dc8d34aedfca0` |

## Successor obligations

- The writer must serialize every field in the field inventory, in the stated
  order and byte order, and include every hostile-case id in test coverage.
- The reader must validate prelude-before-allocation and emit the exact failing
  field/range in every rejection.
- The writer and reader must reference both fixture files by stable id; prose
  acknowledgement alone is not coverage.
- Changing this ADR or the frozen contract is a plan-revision event, not a
  local code tweak.

## OQ-31 completion checklist

- [x] Authority tradeoff resolved with a killed alternative.
- [x] Digest taxonomy resolved with a killed alternative.
- [x] Duplicate-key and floating-value policy resolved with a killed alternative.
- [x] Packing-set multiplicity and dispatch policy resolved with a killed alternative.
- [x] Malformed-file limits and fail-closed policy resolved with a killed alternative.
- [x] Golden field inventory and hostile-case fixture deliverables created.

The research-decision register can record OQ-31 as resolved when the central
batch verifies this bead; the bead remains `in_progress` until that evidence
exists.
