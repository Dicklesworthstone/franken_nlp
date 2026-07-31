<!-- fnlp-ledger-schema: negative-evidence/v1 -->

# Negative evidence

This ledger retains every rejected optimization so that a failed idea is not
mistaken for unexplored work later. A rejected candidate leaves no production
source landed: the entry preserves its hypothesis, proof and measurement
record, and the explicit condition that would justify another attempt.

## Entry schema

Every `NE-...` entry contains the shared artifact-graph fields below, followed
by the rejection-specific fields. `Disposition` is one of `won`, `rejected`,
`prior`, or `deferred`; this ledger normally records `rejected`, while a
reserved seed remains `deferred` until its triggering evidence exists.

- `Claim ID`
- `Evidence`
- `Fixture hashes`
- `CPU feature string`
- `Command + environment`
- `Disposition`
- `Hypothesis`
- `Five-pass loop`
- `Loss basis`
- `Revert proof`
- `Re-evaluation conditions`

`Fixture hashes` is a comma-separated `sha256:<64-lowercase-hex>` list for an
observed result. A `deferred` reservation may instead say `pending:` and name
the missing fixture; that text is not evidence and cannot be promoted without
replacing it with immutable fixture hashes.

## Reserved seed entries

`NE-AVX2-RAW-VPMADDUBSW-001` is pre-registered as rejected by construction:
the raw `vpmaddubsw` path may saturate before widening, so no performance win
can make it an exact mode. Its eventual entry must retain the full-domain
scalar/i64 differential fixture and a no-source-landed revert proof.

`NE-CONVERTER-BYTE-SHORTCUT-001` is reserved for any Phase-2 converter shortcut
that changes canonical output bytes. It remains deferred until a digest-bound
fixture captures the mismatch, after which the entry must name the rejected
shortcut and the restored canonical conversion path.

## Current entries

No measured rejection has been entered at scaffold time.
