# FNLPQ envelope v1

Status: frozen OQ-31 input for the `franken_nlp-rsk` writer and
`franken_nlp-vk7` reader beads.  Any semantic change requires an owner-approved
plan-revision event; an implementation must not silently reinterpret this
document.

## Scope and authority

This format carries one Nanbeige4.2-3B artifact.  It is deliberately a small,
strict binary envelope: the prelude and binary section directory are the sole
authority for byte ranges; the canonical header supplies names, identities,
and logical relationships only.  Header JSON never carries raw binary or
base64.

The only supported v1 file is a regular file.  The reader must reject a
symlink, device, directory, or file whose observed length differs from the
prelude's `file_len`.

## Byte layout

All fixed-width integers are unsigned little-endian.  The file is exactly:

```
+----------------------+ 0
| 80-byte prelude      |
+----------------------+ 80
| canonical header JSON| header_len bytes
+----------------------+ 80 + header_len
| section directory    | section_count * 80 bytes
+----------------------+ directory_end
| zero alignment gaps  | only where required before a section
+----------------------+
| ordered section bytes|
+----------------------+ file_len
```

### Fixed 80-byte prelude

| Offset | Size | Field | Rule |
|---:|---:|---|---|
| 0 | 8 | `magic` | Exactly `FNLPQ\0\0\x01`. |
| 8 | 4 | `format_version` | Exactly `1`. |
| 12 | 4 | `required_flags` | Exactly `0` in v1; any set or unknown required bit rejects. |
| 16 | 8 | `header_len` | At most `1,048,576`; used only after checked bounds arithmetic. |
| 24 | 8 | `section_count` | `1..=4,096`. |
| 32 | 8 | `tensor_count` | `1..=4,096`; must equal the header tensor-array length. |
| 40 | 8 | `file_len` | Equal to the observed regular-file length and at most `68,719,476,736` (64 GiB). |
| 48 | 32 | `header_sha256` | `SHA-256` of the exact canonical-header bytes. |

`PRELUDE_BYTES` and `SECTION_DIRECTORY_ENTRY_BYTES` are both exactly `80`.

### Fixed 80-byte section-directory entry

The directory entry at index `i` describes the one header section whose
`ordinal` is `i`.

| Offset | Size | Field | Rule |
|---:|---:|---|---|
| 0 | 4 | `kind` | Known v1 enum only. |
| 4 | 4 | `flags` | Exactly `0` in v1; unknown required bits reject. |
| 8 | 8 | `name_index` | Matches the header section's ordinal and name. |
| 16 | 8 | `file_offset` | Starts at or after `directory_end`, is aligned, and is within `file_len`. |
| 24 | 8 | `stored_len` | At most the file cap and within `file_len`. |
| 32 | 8 | `logical_len` | Equal to `stored_len` in v1: compression is not supported. |
| 40 | 8 | `alignment` | Power of two in `1..=4,096`. |
| 48 | 32 | `stored_sha256` | Domain-framed digest of the exact stored section bytes. |

The valid section kinds are:

| Value | Name | Cardinality |
|---:|---|---|
| 1 | `GENERIC_TENSOR_PAYLOAD` | exactly one |
| 2 | `GENERIC_TENSOR_SCALES` | exactly one |
| 3 | `GENERIC_TENSOR_ROW_SUMS` | exactly one |
| 4 | `TOKENIZER_MODEL` | exactly one |
| 5 | `MODEL_CONFIG` | exactly one |
| 6 | `TOKENIZER_CONFIG` | exactly one |
| 7 | `CHAT_TEMPLATE` | exactly one |
| 8 | `LICENSE_BUNDLE` | exactly one |
| 9 | `NATIVE_PACKING_PAYLOAD` | zero or more, each explicitly named |

Every singleton kind above is required and may appear exactly once.  Repeated
`NATIVE_PACKING_PAYLOAD` entries are legal only when each appears exactly once
in a declared packing representation.  Unknown kinds, duplicate names,
duplicate singleton kinds, omitted singleton kinds, unreferenced directory
entries, or a header/directory disagreement reject.

Directory ranges are sorted by `file_offset`, non-overlapping, and may not
point into the prelude, header, or directory.  Every gap from `directory_end`
to the first section and between adjacent sections must be the minimum bytes
needed to satisfy the following section's declared alignment and must contain
only zero bytes.  There is no arbitrary prefix, payload, or trailing padding.

## Canonical header JSON

The header is UTF-8 without a BOM and is the exact byte sequence covered by
`header_sha256`.  It is canonical only when all of the following hold:

- It has no insignificant whitespace.
- Object members are ordered lexicographically by raw ASCII key bytes.
- Duplicate keys reject at every nesting level; parsers that keep the last
  key are not acceptable.
- Strings are printable ASCII where they carry an identifier or authority.
  Authority-bearing IDs use `[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}`.  `\"` and
  `\\` are the only permitted escape spellings; `\u`, `\/`, and control-byte
  escapes reject.  No Unicode normalization is performed or needed.
- All numbers are unsigned base-10 integers written as `0` or
  `[1-9][0-9]*`; signs, exponents, fractions, `NaN`, and `Infinity` reject.
- The schema is closed: unknown objects, members, enums, and required flags
  reject rather than receiving a best-effort interpretation.

Required top-level members are `header_schema`, `model`, `recipe_id`,
`source_root_sha256`, `logical_model_sha256`, `packing_set_sha256`,
`license_bundle_sha256`, `limits_profile`, `materialized_sources`, `sections`,
`tensors`, and `packing_sets`.  `header_schema` is exactly
`fnlpq-header-v1`; `limits_profile` is exactly `fnlpq-limits-v1`.

`model` contains `model_id` and the full 40-hex `revision`.  The four
`materialized_sources` entries (`model_config`, `tokenizer_model`,
`tokenizer_config`, and `chat_template`) each name a required section ordinal
and its section digest.
`sections` provides only `ordinal`, `name`, `kind`, and `required`; it never
duplicates offsets, lengths, or hashes from the binary directory.

Each `tensors` entry contains `name`, `canonical_dtype`, `shape`,
`canonical_logical_sha256`, and `generic`.  `shape` has rank `1..=8`; every
dimension is in `1..=4,294,967,295`, and checked dimension products must fit
`u64`.  Tensor names are sorted ASCII and unique.  `generic` contains the
fixed `quantization` recipe ID plus `data`, `scale`, and `row_sum` mappings;
each mapping names a section ordinal, byte offset, and byte length that are
checked against its corresponding singleton section.  Tensor declarations may
not overlap inside any one payload section unless the two declarations are
byte-for-byte identical aliases with the same tensor name, which v1 forbids;
therefore all such overlap rejects.

Each `packing_sets` entry contains `id`, `target`, and `representations`.
Representations name their directory section, exact stored byte cost, and
representation digest.  The mandatory `generic` set binds the three generic
tensor sections.  A native set may bind one or more
`NATIVE_PACKING_PAYLOAD` entries.  A packing set is capped at 16
representations, and the artifact is capped at 16 packing sets.

## Digest domains

All named identities use this domain framing:

```
D(tag, fields...) = SHA-256(tag_ascii || 0x00 || LE64(field_count) ||
                            concat(LE64(field_len) || field_bytes))
```

Digest strings in canonical JSON are lowercase 64-hex encodings.  A section
entry's `stored_sha256` is `D("fnlpq-section-v1", section_name, section_bytes)`.
The normative artifact identities are:

| Name | Domain and contents | Does not cover |
|---|---|---|
| `source_root_sha256` | `D("fnlpq-source-root-v1", exact canonical source-manifest bytes)` | packing and legal notice bytes |
| `logical_model_sha256` | `D("fnlpq-logical-model-v1", model id, revision, ordered `(name, canonical_dtype, shape, canonical logical bytes)` records, exact materialized config/tokenizer/template bytes)` | packing layout and license bundle |
| `packing_set_sha256` | `D("fnlpq-packing-set-v1", recipe id, ordered set/representation IDs, targets, byte costs, and section-byte digests)` | logical source tensors and legal bundle |
| `license_bundle_sha256` | `D("fnlpq-license-bundle-v1", exact Apache-2.0, attribution, and modification-notice bundle bytes)` | all model semantics |
| `fnlpq_file_sha256` | `D("fnlpq-file-v1", exact serialized file bytes)` | release metadata outside the file |
| `release_manifest_sha256` | `D("fnlpq-release-manifest-v1", exact canonical release-manifest bytes)` | an unlisted local file |

`semantic_digest` is retired and must not appear in v1.  Native repacking may
change only `packing_set_sha256`, `fnlpq_file_sha256`, and the release identity;
it must not change `logical_model_sha256`.  A notice-only correction may change
only `license_bundle_sha256`, `fnlpq_file_sha256`, and the release identity.

## Reader order and hard limits

Before any variable-sized allocation, the reader reads only the fixed prelude,
validates magic/version/flags, obtains the actual regular-file length, compares
`file_len`, checks every cap, and performs checked `u64` arithmetic for
`80 + header_len + section_count * 80`.  Only then may it allocate/read the
header or directory.  It verifies `header_sha256` before parsing JSON and
validates the directory before accessing any section payload.

The v1 hard limits are:

| Subject | Cap |
|---|---:|
| file length and aggregate mapped bytes | 64 GiB |
| header JSON | 1 MiB |
| directory entries | 4,096 |
| logical tensors | 4,096 |
| tensor rank | 8 |
| tensor dimension | 4,294,967,295 |
| packing sets / representations per set | 16 / 16 |
| tokenizer model section | 64 MiB |
| model config section | 16 MiB |
| tokenizer config section | 16 MiB |
| chat-template section | 8 MiB |
| license bundle section | 1 MiB |
| alignment | 1 through 4,096, power of two |

All additions, products, offsets, ends, element counts, byte counts, and
section-gap calculations use checked arithmetic.  The reader reports the exact
field, section name/ordinal, or byte range that failed; a generic
"invalid fnlpq" error is insufficient.

## Packing and dispatch rule

A `PackingSet` is an explicit inventory, not a promise that a host can consume
an arbitrary representation.  Every retained representation pays its own
stored byte cost in admission, receipts, and manifests.  Dispatch may select
only a representation physically present in the active artifact and declared
for the observed target.  An architecture mismatch is a hard error that names
the missing target derivation; it never falls back silently to a differently
qualified native packing.  The generic representation remains the universal
declared fallback only when it is installed and is an approved route for the
requested operation.

## Successor fixtures

`tests/fixtures/fnlpq/field_inventory.json` is the writer's golden field
inventory.  `tests/fixtures/fnlpq/hostile_cases.json` is the reader's required
malformed-input corpus.  Writer and reader test suites must reference every
stable fixture ID and preserve the required diagnostic location.
