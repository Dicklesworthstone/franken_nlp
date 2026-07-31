<!-- fnlp-ledger-schema: discrepancies/v1 -->

# Numeric discrepancies

This ledger records accepted, measured numeric divergence from a named
reference behavior. It is not for intentional product behavior: those entries
belong in [BEHAVIOR_NOTES.md](BEHAVIOR_NOTES.md). Every entry must name an
actual rollback authority; an environment variable is never a valid rollback
for a weight precision that the active artifact does not contain.

## Entry schema

Each `DISC-...` entry contains these exact fields:

- `Reference behavior` and `Our behavior`.
- `Profile` (`hf-bf16-eager`, `diagnostic-f32`, `strict-quantized-vN`, or
  `fast-vN`) and `Affected operators/surfaces`.
- `Measured impact` with a named fixture and digest, plus `Review date`.
- `Rollback mechanism` beginning with exactly one real authority:
  `kernel-selector:`, `cli-or-builder-option:`, or
  `prior-immutable-artifact:`.

An entry that scopes a registered public claim also carries `Claim ID`. The
cross-registry validator will resolve that id once `docs/CLAIMS.json` lands.

## Current entries

No accepted numeric divergence is recorded at scaffold time. A proposed or
unmeasured difference must not be added here.
