# ADR 0001: OQ-31 FNLPQ envelope review

> **STATUS OVERRIDE: AUTHORITY CONFLICT / REOPENED.** This ADR's 256-byte-per-entry binary
> table conflicts with ADR-0031's JSON-only ranges and ADR OQ-31's 80-byte
> directory. It is retained as a design candidate, not a ratified implementation
> authority. `franken_nlp-g6f` may compare candidates and regenerate evidence;
> only the owner ratifies the choice, after which the Bead records it and marks
> rejected records historical.

Every decision, digest, fixture reference, and killed-alternative statement
below is historical evidence for that candidate, not current v1 authority.

Historical status claim (challenged): ratified design record; writer and reader
implementation remained separate and batch verification was pending.

## Historical candidate decision

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

## Historical candidate decisions and killed alternatives

| OQ-31 clause | Decision | Killed alternative and reason |
|---|---|---|
| Authority tradeoff | Prelude, header digest, and fixed binary section table are the range authority. JSON holds no raw binary or offsets. | JSON-only/base64 payload directory: duplicates range authority, costs memory, admits ambiguous decoding, and bypasses prelude-first bounds checks. |
| Digest taxonomy | Use six framed names: `source_root_sha256`, `logical_model_sha256`, `packing_set_sha256`, `license_bundle_sha256`, `fnlpq_file_sha256`, and `release_manifest_sha256`. `semantic_digest` is forbidden. | One catch-all semantic digest: cannot distinguish model semantics from source, legal notice, packing, or physical bytes, so it makes a notice-only correction appear semantically material. |
| Duplicate and numeric policy | Duplicate JSON keys, tensor names, section names, and required singleton section kinds reject. Header JSON is one canonical UTF-8 form; floats are fixed IEEE bits in binary scale sections and NaN/Inf reject. | Permissive parser/last-key-wins and JSON float scales: two decoders can authenticate different meanings, and host/serializer float spelling becomes an authority surface. |
| Packing multiplicity | Multiple representations are allowed only in a named PackingSet with checked explicit aggregate bytes; dispatch chooses only a matching installed representation. | One implicit representation or arch fallback: hides duplicate byte cost and can silently execute an unmeasured or wrong-ISA payload. |
| Malformed limits | Validate fixed prelude before allocation, cap header/table/tensor/rank/dimension/mapped-byte sizes, use checked `u64` ranges, require regular-file length equality, reject unknown required flags/enums, and permit only zero alignment gaps. | Parse-then-validate and best-effort newer-file handling: turns hostile lengths, overlap, and unknown semantics into allocations or partially trusted data. |

## Historical candidate digest-scope consequences

Native repacking changes the packing-set, physical-file, and release-manifest
identities but cannot change `logical_model_sha256`. A correction confined to
the Apache-2.0/attribution/modification-notice bundle changes only the legal,
physical-file, and release-manifest identities. The legal bundle is a legal
identity, not model semantics. Whole-file and release-manifest digests are kept
outside their respective bytes to avoid self-reference.

## Historical candidate evidence artifacts

Mutable working-tree paths cannot identify this historical candidate. Its
recoverable snapshot is commit `c161e496010206643297eb2f6f21c991ebbb7b1c`:

| Artifact at `c161e496…` | Git blob | Recomputed raw SHA-256 |
|---|---|---|
| `docs/fnlpq-envelope-v1.md` | `6074cdbc8f99321ddf881aa6cdf1c8c73fb2692c` | `fe580cb136d9a7fcda6311d9b1a6ef148f88e65c656be74b72649c71bf25022f` |
| `tests/fixtures/fnlpq/field_inventory.json` | `75e643b2b88716141c989cbe289e2dad5d181f1c` | `098bb4421b4e0ec35e5c148e3ecacb0e3a36d2b0be7f3d4455d622fb379c0d58` |
| `tests/fixtures/fnlpq/hostile_cases.json` | `154788dea44ccbbbccfe5e90e5baf3fb0262e0b2` | `c62a38d93b7d896718c79599d41801043fd841fa3860ab36730494df97e2a81f` |

The earlier table's two fixture hashes were not reproducible from any committed
version in this repository and are withdrawn rather than silently refreshed.
These immutable blobs preserve the proposal only; they are not current v1
authority or successor-fixture inputs.

## Historical candidate successor obligations

- The writer must serialize every field in the field inventory, in the stated
  order and byte order, and include every hostile-case id in test coverage.
- The reader must validate prelude-before-allocation and emit the exact failing
  field/range in every rejection.
- The writer and reader must reference both fixture files by stable id; prose
  acknowledgement alone is not coverage.
- Changing this ADR or the frozen contract is a plan-revision event, not a
  local code tweak.

## Historical candidate checklist (not current closure)

- [x] Authority tradeoff resolved with a killed alternative.
- [x] Digest taxonomy resolved with a killed alternative.
- [x] Duplicate-key and floating-value policy resolved with a killed alternative.
- [x] Packing-set multiplicity and dispatch policy resolved with a killed alternative.
- [x] Malformed-file limits and fail-closed policy resolved with a killed alternative.
- [x] Golden field inventory and hostile-case fixture deliverables created.

This checklist once proposed that central batch verification could resolve
OQ-31. It cannot: `franken_nlp-g6f` is now **open** on the prior selection itself.
Only an owner-ratified choice followed by regenerated fixtures and reconciled
implementations can return the register to a resolved state.
