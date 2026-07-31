# ADR-0031: FNLPQ v1 envelope authority and hostile-input review

- Status: ratified design contract; code-first, batch-test pending
- Decision id: OQ-31
- Date: 2026-07-31
- Scope: the `.fnlpq` container for Nanbeige4.2-3B at
  `f56ec5a9650268aa098496734743c25ea778bd2d`
- Implements: [FNLPQ envelope v1](../formats/fnlpq-envelope-v1.md)

## Review method and evidence

This is an independent format review. It derives its decisions from the OQ-31
bead's stated container contract and hostile-input requirements, not from a
writer, reader, packer, or pull implementation. The review deliberately
assumes an attacker can control every byte of a purported artifact.

The successor fixtures are part of this decision. Their committed SHA-256
values are recorded below after their exact bytes are finalized:

| Artifact | SHA-256 | Purpose |
| --- | --- | --- |
| `tests/fixtures/fnlpq/field_inventory.json` | `85b6d69f0c6cd546d12fec51090a77781c9d1b02787281db9bd17b4ba42a5e69` | Golden prelude/header inventory for the writer and reader. |
| `tests/fixtures/fnlpq/hostile_cases.json` | `8f3d695c40bee52f5bdb8ed17ce021f55f3ccb4178626258b496660fd4696520` | Stable hostile-input corpus checklist for the reader. |

The batch verifier must recompute these values from the committed fixtures
before it cites this ADR as green evidence. A mismatch is a format-fixture
change and requires review rather than a silent digest edit.

## Decision 1: canonical header is the sole range directory

The 80-byte prelude and the canonical, ASCII JSON header are the sole
authority for the file's declared structure and section ranges. The header's
`sections` array is the section table. It carries metadata and ranges only;
no section's raw bytes, base64 encoding, or JSON number representing a
floating-point value is permitted in the header.

The prelude is parsed before the header is allocated. Its domain-separated
header digest authenticates the exact canonical header bytes. The reader then
requires a byte-for-byte canonical reserialization match before it trusts the
table. Every payload section is separately range-checked and digest-checked.

Killed alternative: a second binary directory that independently names ranges.
It was rejected because two directories create disagreement and parser-order
authority hazards. A binary directory does not buy enough at 201 logical
tensors to justify a second authority surface.

Killed alternative: JSON sections containing base64 payloads. It was rejected
because it duplicates bytes, blurs size accounting, accepts many spellings of
the same data, and weakens pre-allocation limits.

## Decision 2: six non-interchangeable identities

The following names are normative and all use the exact domain-framed
algorithms in the envelope specification. `semantic_digest` is retired and
must not appear in a writer, reader, receipt, manifest, or public claim.

| Name | What it identifies | Must change when |
| --- | --- | --- |
| `source_root_sha256` | The pinned source-closure manifest and its source bytes. | Any source input or source manifest record changes. |
| `logical_model_sha256` | Ordered logical tensors plus exact materialized config, tokenizer, and template identities. | A logical tensor or those materialized inputs change. |
| `packing_set_sha256` | The installed physical representations selected from the logical model. | A representation, target, packing recipe, or packed byte changes. |
| `license_bundle_sha256` | Apache-2.0 text, factual attribution, and modification notice bytes. It is legal identity, never model semantics. | Any legal-bundle byte changes. |
| `fnlpq_file_sha256` | Exact bytes of one serialized `.fnlpq` file. | Any serialized byte changes. |
| `release_manifest_sha256` | Exact canonical release manifest bytes. | Any release manifest byte changes. |

Consequences are mandatory: native repacking changes `packing_set_sha256` and
`fnlpq_file_sha256`, but not `logical_model_sha256`. A notice-only correction
changes `license_bundle_sha256`, `fnlpq_file_sha256`, and
`release_manifest_sha256`, but never `logical_model_sha256` or a packing-set
digest. These mutation cases are named fixtures in the successor suites.

Killed alternative: one overloaded content or semantic digest. It was rejected
because source provenance, logical semantics, host-specific representation,
legal material, a container byte stream, and a release manifest have different
mutation and verification scopes.

## Decision 3: canonical JSON and duplicate rejection are parse-time security

Header JSON is restricted to the pinned canonical ASCII profile. Object keys
are unique and byte-sorted; integers are finite, unsigned, base-10 values;
strings use the pinned printable-ASCII grammar; and the permitted escape
spellings are fixed. The reader detects duplicate keys before constructing a
map, rejects noncanonical input even if a permissive JSON parser would accept
it, and never accepts a float, `NaN`, or infinity.

Section names, logical tensor names, representation ids, and each required
singleton section kind are unique. Tensor representations are repeatable only
because each has a distinct section name and an explicit logical-tensor and
representation-id pairing. Floating scales are payload bytes encoded at their
declared fixed IEEE bit width; they are never JSON numbers.

Killed alternative: ordinary `serde_json::Value` parsing followed by map lookup.
It was rejected because duplicate object keys can be lost before validation.

Killed alternative: accepting equivalent JSON spellings and normalizing them.
It was rejected because header hash/replay identity would become parser-defined
rather than byte-defined.

## Decision 4: packing sets make multiplicity and architecture explicit

A packing set can contain multiple physical representations only when every
representation is listed with its target, recipe id, section name, and exact
stored-byte cost. The declared `stored_bytes` total is checked against the sum
of member payload lengths; duplicate storage is therefore visible to admission
and receipts. Dispatch may select only a representation physically installed
in the active packing set and whose target is an exact compatible match.

An unavailable compatible target is a hard `RequiredDerivation` error naming
the logical tensor(s), requested target, available targets, and the derivation
required. It never silently falls back to another ISA or an unpacked source
format.

Killed alternative: one opaque "best packing" field. It was rejected because
it hides duplicate byte cost and turns target selection into an unreviewable
fallback policy.

Killed alternative: arch-mismatch fallback to any available representation.
It was rejected because it makes performance, memory, and numerics claims
depend on undeclared host behavior.

## Decision 5: pre-allocation bounds and canonical physical layout

The reader validates magic, version, required flags, exact regular-file length,
and all fixed prelude fields before allocating a header, a section list, a
tensor list, a map, or an mmap. It enforces compile-time caps for header,
counts, ranks, dimensions, component sections, individual sections, and total
file bytes. Every addition, multiplication, alignment operation, and range end
is checked in `u64` before conversion to any platform index type.

After canonical-header verification, ranges are sorted by offset and must form
the one canonical layout: zero-filled padding from header end to the first
64-byte-aligned payload, then each section followed only by zero-filled padding
to the next 64-byte boundary. Ranges cannot overlap, point into prelude/header
bytes, omit a declared payload byte, or leave nonzero arbitrary gaps. Unknown
required flags, enums, kinds, and future required features reject rather than
best-effort parse.

Killed alternative: allocate header/table structures before checking lengths.
It was rejected as an attacker-controlled allocation and integer-overflow
surface.

Killed alternative: permit arbitrary ignored gaps and trailing bytes. It was
rejected because an unsigned opaque region makes file identity and forensic
inspection ambiguous.

## Resolution checklist

| OQ-31 clause | Resolution | Named killed alternative |
| --- | --- | --- |
| Authority tradeoff | Canonical header JSON owns the one section table; prelude bounds it. | Second binary directory; base64 payloads. |
| Digest taxonomy | Six domain-framed identities replace `semantic_digest`. | One overloaded semantic/content digest. |
| Duplicate-key policy | Detect duplicates before map construction; require one byte spelling. | Permissive JSON normalization. |
| Packing multiplicity | Explicit targets, representations, and duplicate byte costs; hard mismatch error. | Opaque best packing; silent fallback. |
| Malformed limits | Prelude-first checked caps and canonical zero-padded layout. | Pre-allocation parsing; ignored gaps/trailing bytes. |

## Registry disposition

`OQ-31` changes from `OPEN` to `RESOLVED` for the frozen v1 envelope contract
defined here. It is not an implementation or release-certification result:
writer/reader/puller successors remain blocked on their own named fixtures,
hostile cases, and batch verification evidence.
