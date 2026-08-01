# ADR OQ-31: FNLPQ envelope review before writer/reader freeze

Status: **AUTHORITY CONFLICT / REOPENED.** This 80-byte-per-entry-directory
candidate conflicts with the 256-byte-per-entry-table and JSON-only records that
also claimed frozen authority. It is retained as a design candidate, not
implementation authority. `franken_nlp-g6f` may compare candidates and
regenerate evidence; only the owner ratifies the choice, after which the Bead
records it and marks rejected records historical. Writer, reader, converter,
packager, receipt, and pull acceptance remain blocked.

Historical status claim (challenged): resolved for the code-first implementation
wave with batch verification pending.

All decisions and evidence identifiers below are historical candidate evidence;
none may be used as the active writer/reader or acceptance contract.

## Historical candidate decision

Freeze the v1 contract in
[`docs/specs/fnlpq-envelope-v1.md`](../specs/fnlpq-envelope-v1.md).  It uses an
80-byte validated prelude, a canonical UTF-8 JSON header for identities and
relationships, and an 80-byte-per-entry binary section directory as the sole
authority for byte ranges.  The file is a regular file only; all variable
work begins only after the prelude has passed checked arithmetic, actual-length,
version, flag, and hard-cap validation.

The following table binds this historical proposal to immutable commit
`c2af23c87366298e6065cdf995a39e1d7b37ab34`; the current mutable paths are not
candidate identity and these blobs are not active successor inputs:

| Evidence artifact at `c2af23c…` | Git blob | Recomputed raw SHA-256 |
|---|---|---|
| `docs/specs/fnlpq-envelope-v1.md` | `3bc851f695e70aa9544512037e559a33d4d5e463` | `0d4c3c672790c3025adddc03d08d4ffc5df7cdcebd118428b026cd0fe7806686` |
| `tests/fixtures/fnlpq/field_inventory.json` | `75e643b2b88716141c989cbe289e2dad5d181f1c` | `098bb4421b4e0ec35e5c148e3ecacb0e3a36d2b0be7f3d4455d622fb379c0d58` |
| `tests/fixtures/fnlpq/hostile_cases.json` | `154788dea44ccbbbccfe5e90e5baf3fb0262e0b2` | `c62a38d93b7d896718c79599d41801043fd841fa3860ab36730494df97e2a81f` |

Those hashes are evidence identifiers for this review record.  If an artifact
changes, recalculate the identifier and review the semantic change; do not
edit the table as bookkeeping alone.

## Historical candidate clause 1: header versus binary-directory authority

**Historically proposed, not settled:** the prelude and binary section directory own all stored byte
ranges, alignment, stored lengths, and stored-byte digests.  The canonical
header owns names, logical tensor metadata, digest identities, and the
relationship between a name and a directory ordinal.  Header JSON never holds
raw payload bytes or base64.

This lets the reader prove bounds and non-overlap before parsing an unbounded
metadata graph, while retaining a deterministic, inspectable identity catalog.
The directory and header cross-check every section ordinal/name/kind; neither
may silently override the other.

Killed alternatives:

- **JSON ranges as authority:** rejected because duplicate-key/parser behavior,
  numeric rendering, and header parsing would decide an allocation/range
  boundary before the fixed-size directory could reject it.
- **Binary directory only:** rejected because it would force names, digest
  taxonomy, tensor mappings, and packing intent into ad-hoc binary extensions
  and weaken reviewability without improving the range proof.
- **Base64 payload in JSON:** rejected because it duplicates bytes, obscures
  exact payload lengths, changes digest domains, and invites accidental large
  allocations.

## Historical candidate clause 2: digest-domain taxonomy

**Historically proposed, not settled:** the six candidate names are `source_root_sha256`,
`logical_model_sha256`, `packing_set_sha256`, `license_bundle_sha256`,
`fnlpq_file_sha256`, and `release_manifest_sha256`.  Each is explicitly
domain-framed in the frozen specification.  `logical_model_sha256` covers the
ordered logical tensor records and exact materialized config/tokenizer/template
bytes; it does not cover packing or legal bytes.  `license_bundle_sha256` is
legal identity only, never model semantics.

The resulting mutation rules are mechanical:

- native repacking changes packing/file/release identities but not logical
  model identity;
- a notice-only correction changes license/file/release identities but not
  source, logical-model, or packing identities.

Killed alternatives:

- **One ambiguous `semantic_digest`:** rejected because a reviewer cannot tell
  whether it changes for source, packing, or notice changes.
- **A file digest used as model identity:** rejected because a byte-identical
  file identity changes on a native repack even when model semantics do not.
- **License bytes folded into logical model identity:** rejected because a
  notice correction would falsely look like a model change.

## Historical candidate clause 3: duplicate keys, names, and non-finite values

**Historically proposed, not settled:** duplicate JSON keys, tensor names, section names, singleton
required section kinds, packing-set IDs, and representation IDs all reject.
The header has one ASCII-oriented canonical UTF-8 spelling and a closed schema;
unknown members, enum values, required flags, Unicode escape forms, noninteger
numbers, NaN, and Infinity reject.  Floating scales reside in a binary scale
section with fixed IEEE bits and are rejected when non-finite.

Killed alternatives:

- **Last key wins:** rejected because parsers disagree and malicious headers
  can show one range in review while loading another.
- **Permissive Unicode normalization:** rejected because it requires a new
  authority-bearing normalization surface and can make distinct names collide.
- **Header JSON floating scales:** rejected because JSON's number grammar does
  not preserve the fixed IEEE representation or a portable NaN policy.

## Historical candidate clause 4: packing-set multiplicity and dispatch

**Historically proposed, not settled:** a `PackingSet` may contain several named representations only
when each representation's section digest and full stored byte cost are listed
in the set identity.  Dispatch selects only a representation physically
present and declared for the observed target.  An architecture mismatch names
the required derivation and fails; it never silently treats a different native
packing as equivalent.  The approved generic representation is an explicit
route, not an implicit fallback.

Killed alternatives:

- **One mutable "best" native payload:** rejected because it makes byte cost,
  receipts, and reproducibility host-dependent and hides a repack under the
  same identity.
- **Free duplicate representations:** rejected because admission and release
  accounting would understate the actual artifact footprint.
- **Silent architecture fallback:** rejected because it converts an explicit
  derivation/qualification boundary into an unreported numerics/performance
  change.

## Historical candidate clause 5: malformed-file limits and parser order

**Historically proposed, not settled:** prelude validation precedes every variable allocation.  The
reader checks actual regular-file length, fixed caps, checked directory-size
arithmetic, canonical-header digest, canonical JSON, directory range/order,
zero-only necessary alignment gaps, required singleton inventory, and only
then accesses payload bytes.  Every failure reports the exact field, section
name/ordinal, or range.  The frozen caps include a 1 MiB header, 4,096 sections
and tensors, rank 8, bounded per-kind material sections, 64 GiB total file and
mapped-byte limits, and checked u64 arithmetic throughout.

Killed alternatives:

- **Allocate from `header_len` before validating it:** rejected as an
  attacker-controlled allocation path.
- **Trust EOF rather than declared/observed equality:** rejected because an
  appended or truncated file could be mistaken for a valid artifact.
- **Allow arbitrary padding or overlapping ranges:** rejected because hidden
  bytes and aliases defeat reproducible hashing and bounds review.

## Historical successor obligations

The writer must serialize the prelude/directory/header exactly as the field
inventory requires, with a golden fixture for every stable field ID.  The
reader must exercise every hostile-case ID in its fuzz/regression corpus,
reject before the case's forbidden access, and assert the required diagnostic
location.  Both must expose the exact failing field/range rather than collapse
errors to a generic parse failure.

This record previously purported to resolve OQ-31's five design clauses for a
code-first wave. That disposition is withdrawn: `franken_nlp-g6f` is **open**
on the three-way authority conflict, stale fixture digests, and implementation
reconciliation. Central execution of this candidate cannot resolve the conflict
without an owner-ratified selection first.
