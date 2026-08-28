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

## NE-AVX2-RAW-VPMADDUBSW-001

- Claim ID: none
- Evidence: Raw vpmaddubsw is banned by construction per AGENTS.md
  doctrine — the i16-pair product can saturate at (127, 127) (i16 sum 32,126)
  and at (-128, -128) (i16 sum 32,768), hiding the true i32 result before
  widening. The full-domain scalar/i64 differential fixture has not yet been
  wired; the rejection stands on architectural grounds, not measurement.
- Fixture hashes: pending: scalar_i64_vs_vpmaddubsw_avx2_full_domain.bin (all 256 i8 weight times all 256 i8 activation pairs)
- CPU feature string: avx2
- Command + environment: pending: no canonical rejection fixture wired; future command = fnlp bench --kernel vpmaddubsw; profile=hf-bf16-eager; threads=1
- Disposition: deferred
- Hypothesis: raw vpmaddubsw is a faster AVX2 path that matches scalar/i64
  for all stored i8 values.
- Five-pass loop: five-pass loop retained: baseline (scalar), lever (vpmaddubsw), parity (full-domain inputs including -128 weights and both correction extremes), thermal A/B, revert (no source landed).
- Loss basis: parity violation: the saturating i16 sum diverges from the i32
  reference at product pairs where abs(qx * w) > 32,767. The accumulator is
  not bit-identical and cannot be widened into a proof.
- Revert proof: no source landed; the vpmaddubsw path is a banned-by-construction disposal in any future tier implementation, not an opt-in. AVX-512 VNNI (VPDPBUSD) is the correct replacement and is tracked separately under the int8 quantization campaign.
- Re-evaluation conditions: re-evaluation is justified only if a future instruction set provides the same integer semantics without the i16 saturation; the current AVX-512 VNNI VPDPBUSD is the canonical replacement path.

## NE-CONVERTER-BYTE-SHORTCUT-001

- Claim ID: none
- Evidence: Phase-2 converter shortcuts that change canonical output bytes
  (e.g., a misordered tensor name, a swapped row/col scan, a different
  bf16-to-int8 scale formula) are tracked here. No such shortcut has been
  observed yet; the reservation is fail-closed.
- Fixture hashes: pending: digest-mismatched converter output vs canonical
- CPU feature string: n/a: a converter defect is a host-independent logical error, not a measurement.
- Command + environment: n/a: static converter audit, no host command.
- Disposition: deferred
- Hypothesis: converter shortcuts are off-policy and would change the artifact's canonical byte identity.
- Five-pass loop: five-pass loop retained: baseline (canonical converter), lever (shortcut), parity (re-digest the converted artifact and compare against the original .fnlpq_file_sha256), thermal A/B (host-independent), revert (restore canonical path).
- Loss basis: output digest drift; the artifact becomes a different file with the same logical name, breaking identity-based cache authorities.
- Revert proof: no source landed; the canonical converter is the only path.
- Re-evaluation conditions: re-evaluation requires a future converter that preserves the canonical output bytes AND improves throughput. The current canonical path is locked by the v1 spec.

Both entries above are `deferred` (no measurement yet). They are the
contractually-named seed rows for the doctrine: `vpmaddubsw` is the
int8-saturation example, and the converter byte shortcut is the
phase-2 byte-drift example. They become `rejected` only after the
respective fixture is wired and the no-source-landed revert proof is
replaced with the matching conversion evidence.
