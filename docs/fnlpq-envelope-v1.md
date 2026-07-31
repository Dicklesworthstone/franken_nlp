# FNLPQ Envelope v1

Status: frozen by OQ-31 for the writer and reader beads. A change to this
document is a plan-revision event and requires a renewed independent review.

This is the container for Nanbeige4.2-3B at source revision
`f56ec5a9650268aa098496734743c25ea778bd2d`. It describes 201 logical tensors,
the exact materialized tokenizer/config/template inputs, and the legal bundle.
It does not make a claim about parity or a converter result.

## Layout and authority

The file is, in order:

1. an 80-byte fixed binary prelude;
2. `header_len` bytes of canonical header JSON;
3. `section_count` fixed 256-byte binary section-table entries; and
4. ordered section payloads, separated only by declared, zero-filled alignment
   gaps.

The prelude, header digest, and binary section table are the sole authority for
file ranges. Header JSON identifies the logical model, materialized source
inputs, packing sets, and required section kinds; it never carries ranges,
offsets, payload bytes, base64, or a second binary directory. A reader must not
use a JSON value as an alternate range authority.

All multibyte binary integers are unsigned little-endian. `header_sha256` is
the SHA-256 of the exact canonical header bytes. The header carries
`section_table_sha256`, the SHA-256 of the exact contiguous table bytes. Each
section-table entry carries the SHA-256 of its payload bytes. The whole-file
digest is deliberately external to the file (in the converter receipt and
release manifest) because including a digest of the whole file in the file
would be self-referential.

## Fixed prelude

`PRELUDE_LEN` is exactly 80 bytes. Every reserved byte described below must be
zero; this rule permits a reader to reject a file that only resembles v1.

| Offset | Field | Encoding and v1 rule |
|---:|---|---|
| 0 | `magic` | 8 raw bytes, exactly `46 4e 4c 50 51 00 00 01` (`FNLPQ\\0\\0\\x01`) |
| 8 | `format_version` | `u32`, exactly `1` |
| 12 | `required_flags` | `u32`, exactly `0x0000_000f`; unknown set bits reject |
| 16 | `header_len` | `u64`, `1..=16,777,216` |
| 24 | `section_count` | `u64`, `1..=4,096` |
| 32 | `tensor_count` | `u64`, exactly `201` for this model and never greater than `512` |
| 40 | `file_len` | `u64`, exactly the length of the opened regular file and at most `34,359,738,368` |
| 48 | `header_sha256` | 32 raw bytes, SHA-256 of the header range |

The required-flag bits are `SECTION_TABLE_V1` (`0x1`),
`CANONICAL_HEADER_JSON_V1` (`0x2`), `BINARY_FLOATS_ONLY` (`0x4`), and
`PACKING_SET_MANIFEST_V1` (`0x8`). A v1 writer sets all four. A v1 reader
rejects a missing required bit, an unknown set bit, or a later
`format_version`; it never guesses a compatible parse.

Before any variable allocation, a reader obtains the opened handle's regular
file length, validates the entire prelude, and checked-adds
`PRELUDE_LEN + header_len + section_count * SECTION_ENTRY_LEN`. That result
must be within `file_len`. The parser has a hard total mapped-byte cap equal to
`MAX_FILE_LEN`; an implementation may map less, but may never map or allocate
from an unvalidated declared range.

## Binary section table

`SECTION_ENTRY_LEN` is exactly 256 bytes. The table follows the header without
padding and is itself authenticated by `section_table_sha256` in the header.
No field in this entry is a host-native layout.

| Offset | Field | Encoding and v1 rule |
|---:|---|---|
| 0 | `kind` | `u32` closed enum |
| 4 | `entry_flags` | `u32`, currently zero; unknown bits reject |
| 8 | `name_len` | `u16`, `1..=96` |
| 10 | `dtype` | `u16` closed enum |
| 12 | `rank` | `u16`, `0..=8` |
| 14 | `alignment_log2` | `u16`, `0..=16`; payload offset must be aligned to `1 << alignment_log2` |
| 16 | `payload_offset` | `u64` checked range start |
| 24 | `payload_len` | `u64`, `1..=17,179,869,184` |
| 32 | `logical_len` | `u64`; checked shape/dtype product for tensor data, otherwise the logical byte count |
| 40 | `dims` | eight `u64` dimensions; unused slots are zero |
| 104 | `name` | 96 bytes; first `name_len` bytes are printable ASCII authority-bearing name and remainder is zero |
| 200 | `payload_sha256` | 32 raw SHA-256 bytes |
| 232 | `reserved` | 24 zero bytes |

Section kinds are closed: `TENSOR_DATA=1`, `TENSOR_SCALES=2`,
`TENSOR_ROW_SUMS=3`, `TOKENIZER_MODEL=10`, `SOURCE_CONFIG=11`,
`SOURCE_TOKENIZER_CONFIG=12`, `SOURCE_CHAT_TEMPLATE=13`, `LICENSE_BUNDLE=14`,
and `PACKING_SET_MANIFEST=15`. The five non-tensor singleton kinds
`TOKENIZER_MODEL`, `SOURCE_CONFIG`, `SOURCE_TOKENIZER_CONFIG`,
`SOURCE_CHAT_TEMPLATE`, and `LICENSE_BUNDLE` are required exactly once.
`PACKING_SET_MANIFEST` is also required exactly once. A duplicate required
singleton kind, any duplicate section name, or an unknown kind rejects.

`dtype` is closed: `RAW=0`, `BF16=1`, `I8=2`, `I4_PACKED=3`, `U8=4`,
`I32=5`, and `F32_IEEE754_LE=6`. The byte representation of `BF16` and
`F32_IEEE754_LE` is fixed little-endian IEEE bits, never a host float. Scale
sections use `F32_IEEE754_LE` and reject every NaN or infinity bit pattern.
Floating values never occur as JSON numbers.

Table entries are sorted by `(payload_offset, name)`; payload ranges are
strictly non-overlapping. The only bytes between the end of one declared range
and the next declared range are alignment gaps, and every such byte must be
zero. Every range uses checked `u64` addition and must end at or before
`file_len`. Tensor dimensions are nonzero through `rank`, zero after `rank`,
each dimension is at most `4,294,967,295`, and their checked byte product is at
most `17,179,869,184`. This permits the 8,339,601,408-byte bf16 logical
payload while bounding hostile shape claims.

## Canonical header JSON

The header is UTF-8 and is accepted only when it is already its canonical
serialization. Its grammar permits objects, arrays, booleans, `null`, unsigned
base-10 integers, and strings. It forbids a leading zero on a nonzero integer,
negative integers, exponent notation, decimal numbers, `NaN`, and `Infinity`.
Object keys sort by their raw UTF-8 bytes and duplicate keys reject at parse
time. There is no insignificant whitespace.

Authority-bearing strings (model id, revision, field names, section names,
tensor names, architecture ids, and packing ids) are printable ASCII and use
only the grammar `[A-Za-z0-9._/-]+`; this avoids Unicode-normalization
authority. Other strings are not admitted in v1's header. The only permitted
JSON escapes are `\\\"` and `\\\\`; non-ASCII and control-character escapes are
forbidden. Source JSON, templates, and tokenizer bytes remain exact binary
sections rather than being normalized or re-emitted through this header.

The top-level object has exactly these fields:

| Field | Rule |
|---|---|
| `schema` | exactly `franken_nlp.fnlpq.header.v1` |
| `model` | object with `id`, full 40-hex `source_revision`, and `architecture_id` |
| `source_root_sha256` | lowercase 64-hex digest |
| `logical_model_sha256` | lowercase 64-hex digest |
| `packing_set_sha256` | lowercase 64-hex digest |
| `license_bundle_sha256` | lowercase 64-hex digest |
| `section_table_sha256` | lowercase 64-hex digest of the binary table |
| `materialized_sources` | exactly the tokenizer model, source config, tokenizer config, and chat-template identities and section names |
| `logical_tensors` | exactly 201 unique tensor records: `name`, `dtype`, `shape`, `logical_sha256`, and canonical section name |
| `packing_sets` | one or more unique packing-set records with architecture, representation ids, and explicit byte totals |
| `required_section_kinds` | sorted list of the six singleton kinds required by this format |

The header does not contain `fnlpq_file_sha256` or `release_manifest_sha256`.
Those identities are recorded by their enclosing receipt or release manifest.

## Digest domains

Every digest is SHA-256 of a domain tag terminated by `00`, followed by framed
records. Every variable-length field is framed by its unsigned little-endian
length before its bytes; names are ASCII byte-sorted. This framing is
normative, so concatenation cannot create an ambiguous identity.

| Name | Domain tag and identity |
|---|---|
| `source_root_sha256` | `franken_nlp.source-root.v1\\0`; sorted source-closure records `(path, byte_len, sha256)` |
| `logical_model_sha256` | `franken_nlp.logical-model.v1\\0`; ordered records `(name, dtype, rank, dims, canonical logical tensor bytes)`, followed by exact materialized config, tokenizer model, tokenizer config, and chat-template byte identities |
| `packing_set_sha256` | `franken_nlp.packing-set.v1\\0`; ordered packing-set `(id, architecture, representation, section-name, byte_len, payload_sha256)` records |
| `license_bundle_sha256` | `franken_nlp.license-bundle.v1\\0`; exact ordered `(name, bytes)` legal-bundle records |
| `fnlpq_file_sha256` | `franken_nlp.fnlpq-file.v1\\0` followed by the exact complete file bytes; stored externally to avoid a cycle |
| `release_manifest_sha256` | `franken_nlp.release-manifest.v1\\0` followed by exact canonical release-manifest bytes; stored by its receipt/attestation, not inside itself |

`semantic_digest` is retired and forbidden: it did not state whether it meant
source, logical model, packing, legal bundle, or physical file identity. A
native repack changes `packing_set_sha256`, `fnlpq_file_sha256`, and the
containing `release_manifest_sha256`, but never `logical_model_sha256`. A
notice-only correction changes `license_bundle_sha256`, `fnlpq_file_sha256`,
and `release_manifest_sha256`, but not source-root, logical-model, or packing
identities.

## Packing sets and dispatch

A `PackingSet` can contain multiple representations only when its
`explicit_byte_total` equals the checked sum of its referenced table payload
lengths. The byte cost of a duplicate representation is therefore visible in
the canonical header and receipt. Dispatch chooses only a representation that
is installed and whose architecture id matches the detected, measured profile.
An architecture mismatch is a hard `FNLPQ_ARCH_MISMATCH` error that names the
missing required derivation; it never silently uses a generic or wrong-ISA
fallback.

## Required validation order

1. Open a non-symlink regular file and obtain its actual length.
2. Validate all fixed prelude fields and fixed caps before allocation.
3. Checked-add the header and fixed-size table ranges; reject overflow or a
   range beyond `file_len`.
4. Read at most `MAX_HEADER_LEN`, verify `header_sha256`, parse canonical JSON,
   and enforce closed enums, duplicate policy, and header caps.
5. Read the fixed table extent, verify `section_table_sha256`, validate every
   entry, then validate sorted ranges, zero gaps, names, required singleton
   kinds, tensor mapping, and packing totals.
6. Hash each needed payload before use; validate every scale's IEEE bits and
   every tensor's shape/dtype/logical length.
7. Only after all applicable validation succeeds may the loader map, pack,
   dispatch, or activate data.

Every rejection must name the exact failing field or byte range, for example
`prelude.header_len`, `section[17].payload_offset`, or
`header.logical_tensors[94].name`. The successor fuzz corpus and serializer
tests must consume the stable ids in `tests/fixtures/fnlpq/`.
