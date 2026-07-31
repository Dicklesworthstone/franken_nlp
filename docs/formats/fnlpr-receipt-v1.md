# FNLPR receipt v1

Status: implementation contract for `franken_nlp-2xo`. This document defines
a typed evidence record; it does not grant acceptance authority to any run,
artifact, or claim.

## Scope

An `.fnlpr` file is a canonical JSON receipt for one conversion, pull,
inference, evaluation, scheduler failure, or other named operation. It records
what was attempted, the exact semantic/provenance identities, what checks were
run, and the receipt's replayability grade. It is not a boolean certificate.

The v1 implementation has one encoder and one duplicate-key-rejecting decoder.
Unknown required fields, unknown enums, duplicate keys, and noncanonical bytes
are refusals. There is no pre-1.0 compatibility parser or best-effort import.

## Canonical JSON

Receipt bytes use the repository canonical JSON profile:

- UTF-8, no BOM or insignificant whitespace;
- object keys sorted by unsigned ASCII byte order;
- no duplicate keys at any depth;
- arrays retain their declared order;
- integers are base-10 unsigned values; floats, exponents, and nonfinite
  numbers are not part of v1;
- digest values are exactly 64 lower-case hexadecimal bytes;
- identifiers use `[A-Za-z0-9][A-Za-z0-9._/-]{0,255}`.

The exact byte stream is the receipt artifact. A future semantic change needs a
new `receipt_schema`, frozen fixtures, and an explicit format decision.

## Top-level record

The top-level object has exactly these fields:

| Field | Type | Rule |
| --- | --- | --- |
| `artifact` | object | Complete public artifact/provenance binding, §4. |
| `checks` | array | Ordered typed check records, §5. |
| `code` | object | Converter and binary provenance, §4. |
| `completeness_grade` | enum | Exactly one grade from §3. |
| `content` | object | Private-content commitments or public identities, §6. |
| `evidence` | array | Ordered immutable evidence references, §5. |
| `identity` | object | Projection of the one `ExecutionIdentity`, §2. |
| `receipt_schema` | string | Exactly `fnlpr-v1`. |
| `replay` | object | Grade-constrained replay metadata, §3. |
| `run` | object | Operation name, result disposition, and stable run metadata. |

No field named `verified`, `is_verified`, or an ungraded synonym is permitted.
Checks have their own scoped dispositions; the receipt has one completeness
grade.

## Identity binding

Receipt code receives an `ExecutionIdentity`; it never constructs an ad-hoc
parallel tuple. The `identity` object has exactly:

| Field | Rule |
| --- | --- |
| `execution_identity_schema` | The input identity's schema version. |
| `disclosure` | `public` or `committed`. |
| `receipt_identity` | Lower-case `ExecutionIdentity::receipt_identity()` only for public disclosure; otherwise `null`. |
| `value` | A public canonical identity byte commitment or an HMAC commitment, as below. |

For `disclosure=public`, `value` is the lower-case SHA-256 of the complete
canonical `ExecutionIdentity` bytes. This spelling is permitted only when all
fields represented by that identity are authorized for public receipt export.

For `disclosure=committed`, `value` is an HMAC commitment of the complete
canonical `ExecutionIdentity` bytes under the `execution-identity` content
domain. This is mandatory when an identity contains private prompt or input
material. It preserves one-source identity construction without exporting an
unkeyed, dictionary-testable prompt/content hash. In this mode
`receipt_identity` is `null`: the internal unkeyed projection may still drive
owner-only bookkeeping, but it is not an exported receipt field.

## Completeness grades and replay fields

`completeness_grade` is exactly one of:

| Grade | Required replay meaning |
| --- | --- |
| `replayable` | Authorized inputs, artifacts, and any required commitment secret are resolvable through the named retention policy. |
| `structural_replay` | Structure, cost, command, code, and provenance are replayable; private content is absent. |
| `verifiable_if_artifacts_supplied` | Identities and required artifact names are retained, but caller-supplied bytes and/or a commitment secret are needed. |
| `audit_only` | Historical facts are retained with no replay assertion. |

`replay` has exactly `command`, `missing_requirements`, and `retention_policy`.
`command` is required for the first three grades and empty for `audit_only`.
`missing_requirements` is empty for `replayable`, names every required absent
artifact/secret for `verifiable_if_artifacts_supplied`, and records omitted
private content for `structural_replay`. It is an explanatory inventory, not a
substitute for a grade.

A scheduler/concurrency-failure receipt is required to use
`structural_replay` and adds a `failure_replay` object to `run` with exactly
`crashpack_id`, `lab_seed`, `replay_command`, and `trace_fingerprint`.

## Public artifact and code binding

`artifact` has these six optional-but-explicit lower-case SHA-256 fields:
`fnlpq_file_sha256`, `license_bundle_sha256`, `logical_model_sha256`,
`packing_set_sha256`, `release_manifest_sha256`, and `source_root_sha256`.
An operation without an applicable artifact writes JSON `null`, never silently
omits the field. `artifact.provenance_receipt` is the canonical
`ProvenanceIdentity::receipt_digest()` when all required provenance fields are
present, otherwise `null`.

`code` has exactly `binary_commit` and `converter_commit`. Both are either a
lower-case 40-hex Git revision or `null` when inapplicable; a timestamp,
branch name, or moving checkout label is not provenance.

The combination of `identity`, `artifact`, and `code` binds the semantic
request, model recipe/profile, full artifact taxonomy, and code without
treating legal-notice changes as an execution-semantic change. In committed
mode the HMAC, rather than an unkeyed semantic digest, is the exported binding.

## Checks and evidence

Every `checks` item has exactly `id`, `scope`, and `verdict`:

- `id` is a stable identifier such as `l2_44exec` or `convert-reload`;
- `scope` names the evaluated artifact/profile/dataset boundary;
- `verdict` is one of `pass`, `fail`, `skipped`, or `not_run`.

Every `evidence` item has exactly `id`, `kind`, `sha256`, and `uri`.
`sha256` is allowed only for public retained evidence bytes. Evidence ordering
is semantic and must remain the producer's stated order.

`pass` is a result for that named scope only. A receipt implementation must
not infer a product acceptance state from a collection of check records.

## Content privacy and HMAC commitments

`content` has exactly `commitment_key_id`, `inputs`, and `outputs`.
`commitment_key_id` is a nonsecret ASCII rotation identifier or `null`; it
must be non-null when an input or output uses a commitment.

Each input/output item has exactly `kind`, `label`, and `value`:

| `kind` | `value` |
| --- | --- |
| `absent` | `null`; no content identity was exported. |
| `public_sha256` | 64 lower-case hex; allowed only for public, non-low-entropy bytes. |
| `hmac_sha256` | 64 lower-case hex HMAC commitment. |

Raw SHA-256 of private, low-entropy, user-supplied, or result bytes is
forbidden anywhere in the exported receipt, including errors, evidence labels,
and `identity`. The commitment key and private bytes are owner-only retention
state and never appear in `.fnlpr`, telemetry, filenames, command lines, or
diagnostics.

The HMAC is RFC 2104 HMAC-SHA-256 over the approved `sha2` primitive. The
implementation uses a key of at least 32 bytes and the following unambiguous
message framing:

```text
LE64(len(domain)) || domain || LE64(len(namespace)) || namespace ||
LE64(len(entity_kind)) || entity_kind || LE64(len(bytes)) || bytes
```

The v1 domain strings are exactly:

| Content | Domain |
| --- | --- |
| canonical execution identity | `fnlp-receipt-execution-identity-v1` |
| input bytes | `fnlp-receipt-input-v1` |
| output bytes | `fnlp-receipt-output-v1` |
| configuration/private auxiliary bytes | `fnlp-receipt-config-v1` |

Using an input commitment in an output field, a duplicate `(kind,label)`
record, a short key, or a missing key id is a typed receipt-construction
refusal. The code must carry RFC 4231 vectors and a cross-domain inequality
fixture; a cryptographic digest is not a claim that the committed bytes will
remain available.

## Construction order

1. Validate and derive the one `ExecutionIdentity` and its receipt projection.
2. Classify every content identity as public, committed, or absent before
   serializing any field.
3. Validate grade/replay requirements and typed checks/evidence.
4. Serialize the fixed record through canonical JSON and retain the exact
   bytes as the receipt artifact.

No stage may emit an ungraded receipt, a receipt with a raw private-content
digest, or a receipt that claims a stronger replay grade than its retention
inventory supports.
