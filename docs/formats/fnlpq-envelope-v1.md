# FNLPQ envelope v1

Status: frozen by ADR-0031 for writer/reader implementation. Any semantic
change requires a new format-version decision, updated golden fixtures, and a
plan-revision event; it is not a compatibility shim.

## 1. Physical file layout

An FNLPQ v1 file is exactly:

```text
prelude[80] || canonical_header[header_len] || zero_padding || payload_sections
```

There is no trailing data. `file_len` in the prelude equals the actual length
of the opened regular file and the final byte of the final payload section.

All fixed-width integer fields are unsigned little-endian. The fixed prelude
is parsed field-by-field before the reader allocates any variable-sized object.

| Offset | Size | Field | Encoding / required value | Cap or validation |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | `magic` | bytes `46 4e 4c 50 51 00 00 01` (`FNLPQ` + v1 marker) | exact match |
| 8 | 4 | `format_version` | `u32`, little-endian | exactly `1` |
| 12 | 4 | `required_flags` | `u32`, little-endian | exactly zero in v1; any unknown set bit rejects |
| 16 | 8 | `header_len` | `u64`, little-endian | `1..=4,194,304` |
| 24 | 8 | `section_count` | `u64`, little-endian | `1..=4,096` |
| 32 | 8 | `tensor_count` | `u64`, little-endian | `1..=512`; exactly 201 for the pinned Nanbeige artifact |
| 40 | 8 | `file_len` | `u64`, little-endian | `80 + header_len <= file_len <= 17,592,186,044,416` |
| 48 | 32 | `header_sha256` | bytes | `SHA-256("franken_nlp/fnlpq/header/v1\\0" || canonical_header)` |

`header_len` is checked with `80 + header_len` in `u64` before a header buffer
is allocated. The implementation rejects a non-regular file, a physical length
different from `file_len`, or a `file_len` over the v1 total-file cap before
parsing the header.

## 2. Canonical header profile

The header is UTF-8 but constrained to printable ASCII bytes `0x20..=0x7e`.
It has no BOM and no whitespace outside strings. It is an object encoded with:

- object keys in ascending unsigned-byte lexicographic order;
- no duplicate object key at any depth;
- arrays retained in their specified order;
- unsigned integer literals only, with `0` or a nonzero first digit and no
  leading plus, minus, decimal point, exponent, `NaN`, or infinity;
- strings limited to printable ASCII names and values needed by this schema;
  a quote is encoded only as `\\"` and a reverse solidus only as `\\\\`;
  `\\u`, slash, and control-character escape forms are forbidden;
- ASCII identifier values matching `[A-Za-z0-9][A-Za-z0-9._/-]{0,255}`;
- SHA-256 values as exactly 64 lower-case hexadecimal ASCII characters.

The reader must run a duplicate-preserving parse, validate this profile, then
canonicalize and require byte equality with the stored header before accepting
any range. Parsing JSON into a map before duplicate detection is noncompliant.

The top-level object has exactly these lexicographically ordered keys:

```json
{
  "artifact": { ... },
  "digests": { ... },
  "format": { ... },
  "packing_sets": [ ... ],
  "sections": [ ... ],
  "tensors": [ ... ]
}
```

| Header path | Type / byte size | Rule / cap |
| --- | --- | --- |
| `artifact.model_id` | ASCII identifier, 1–256 bytes | must be `Nanbeige4.2-3B` for this v1 product profile |
| `artifact.model_revision` | 40 ASCII hex bytes | exact pinned source revision |
| `artifact.profile` | ASCII identifier, 1–256 bytes | named conversion recipe/profile |
| `digests.*` | 64 ASCII hex bytes each | exactly the six digest names in §4, no extras |
| `format.header_encoding` | ASCII identifier | exactly `fnlpq-cjson-ascii-v1` |
| `format.payload_alignment` | JSON unsigned integer | exactly `64` |
| `packing_sets` | array | 1–64 entries, each unique `id` |
| `sections` | array | length equals prelude `section_count`; unique `name` |
| `tensors` | array | length equals prelude `tensor_count`; unique `name`; sorted by name |

Each tensor descriptor has exactly `dtype`, `logical_bytes`, `name`, and
`shape`. `dtype` is exactly `bf16` in the Nanbeige4.2-3B v1 logical source
model; quantized dtypes describe physical `tensor_representation` payloads,
not this source identity. `shape` is an array of `1..=8` positive `u64`
dimensions, each at most `4,294,967,296`. The checked product of dimensions is
capped at `2^48` elements and `logical_bytes` must equal twice that product.
The header's tensor descriptors identify the source logical tensor set; they do
not embed those tensor bytes.

## 3. Section table and canonical ranges

`sections` is the only section table. A section object has exactly these
lexicographically ordered keys:

```json
{
  "bytes_sha256": "<64 lower-case hex>",
  "kind": "<kind>",
  "length": 0,
  "name": "<unique ascii identifier>",
  "offset": 0,
  "packing_set": "<packing set id or none>",
  "representation": "<representation id or none>",
  "tensor": "<logical tensor name or none>"
}
```

`offset` and `length` are `u64`; `length` is `1..=1,099,511,627,776` bytes and
every checked range end must be at most `file_len`. `bytes_sha256` is
`SHA-256("franken_nlp/fnlpq/section/v1\\0" || kind || 0x00 || name || 0x00 || payload_bytes)`.

The singleton required kinds are exactly one each of `attribution`,
`chat_template`, `license_text`, `model_config`, `modification_notice`,
`tokenizer_config`, and `tokenizer_model`. `tensor_representation` may repeat,
but only with a unique section name and an existing unique `(tensor,
representation, packing_set)` tuple. No unknown v1 kind is accepted.

The first section must begin at `align_up(80 + header_len, 64)`. Each following
section must begin at `align_up(previous.offset + previous.length, 64)`. Every
intervening byte must be zero; no arbitrary gaps, overlaps, header-contained
ranges, omitted bytes, or trailing bytes are accepted. `align_up`, range end,
and total-byte accumulation are checked `u64` operations. The sum of payload
lengths is capped by `file_len` and the file cap from §1.

`model_config`, `tokenizer_config`, and `chat_template` carry exact source
bytes; `tokenizer_model` carries exact tokenizer bytes. `license_text`,
`attribution`, and `modification_notice` carry the legal bundle's exact bytes.
No section is compressed or base64-encoded in v1.

## 4. Digest domains

All records use unsigned lengths encoded as `u64` little-endian and byte
strings prefixed by their length. The ASCII domain marker ends in one `0x00`.
The notation `record(name, bytes)` means `u64(name.len) || name ||
u64(bytes.len) || bytes`; tensor shapes encode a `u32` rank then `u64` dims.
These rules permit streaming and prohibit concatenation ambiguity.

| Field | Exact domain |
| --- | --- |
| `source_root_sha256` | `SHA-256("franken_nlp/source-root/v1\\0" || canonical_source_manifest_bytes)` |
| `logical_model_sha256` | `SHA-256("franken_nlp/logical-model/v1\\0" || ordered logical tensor records || ordered materialized identity records)`; tensors are sorted by ASCII name and each record includes name, dtype tag, rank, dims, canonical logical-byte length, and canonical logical bytes. Materialized identities are exact `model_config`, `tokenizer_config`, `tokenizer_model`, and `chat_template` bytes sorted by kind. |
| `packing_set_sha256` | `SHA-256("franken_nlp/packing-set/v1\\0" || packing-set id || target || recipe id || ordered representation records)`; each record includes logical tensor name, representation id, section byte length, and packed section bytes. |
| `license_bundle_sha256` | `SHA-256("franken_nlp/license-bundle/v1\\0" || ordered records for license_text, attribution, modification_notice)`; it must not enter `logical_model_sha256`. |
| `fnlpq_file_sha256` | `SHA-256("franken_nlp/fnlpq-file/v1\\0" || exact_file_bytes)`; it is published outside the file to avoid self-reference. |
| `release_manifest_sha256` | `SHA-256("franken_nlp/release-manifest/v1\\0" || canonical_release_manifest_bytes)` |

The header's `digests` object contains exactly these six names. A native
repacking must preserve `logical_model_sha256`; a legal-notice-only correction
must preserve both `logical_model_sha256` and every packing-set digest.

## 5. Packing-set contract

Each packing set has exactly `id`, `recipe`, `representations`, `stored_bytes`,
and `target`. `id`, `recipe`, and `target` use the ASCII identifier grammar.
`representations` is a sorted array of nonempty references to matching
`tensor_representation` sections; `stored_bytes` is the checked sum of their
section lengths. A packing set may carry multiple representations only by
charging every member's bytes in that total.

The loader selects only a physically present representation declared compatible
with the active target. If one does not exist, it returns
`RequiredDerivation { tensor, requested_target, available_targets }`; a generic
or other-architecture representation is never an implicit fallback.

## 6. Reader rejection order and diagnostics

The reader must reject in this order where applicable:

1. opened object is not a regular file, fixed prelude is truncated, magic,
   version, or required flags are unsupported, or physical length differs;
2. `header_len`, counts, or `file_len` violates a fixed cap or a checked fixed
   arithmetic invariant;
3. header bytes fail the header digest, UTF-8/ASCII profile, duplicate-key
   check, canonical-byte equality, or schema/count consistency;
4. a name, singleton required kind, tensor descriptor, packing set, or digest
   field violates its uniqueness, enum, relation, or cap;
5. a range overflows, overlaps, falls before payload start, violates canonical
   alignment/zero padding, or fails a section digest;
6. target dispatch lacks its declared installed representation.

Every failure reports the exact prelude field, header JSON path, section name,
or byte range that failed. It never reports a successful parse with ignored
unknown required flags/enums or unverified data.
