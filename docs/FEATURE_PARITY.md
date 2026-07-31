<!-- fnlp-ledger-schema: feature-parity/v1 -->

# Feature parity

This matrix records the product surface against the named reference stack. It
does not round a partial implementation up to present: every `partial` state
names the missing behavior and the bead or phase that gates it.

## Entry schema

Every `PARITY-...` row has the common artifact-graph fields plus these fields:

- `Claim ID`
- `Evidence`
- `Fixture hashes`
- `CPU feature string`
- `Command + environment`
- `Disposition` — one of `won`, `rejected`, `prior`, or `deferred`.
- `Surface`
- `Reference counterpart`
- `State` — exactly `present`, `partial`, `missing`, or `n/a`.
- `Missing behavior`
- `Gate`

Interface-only rows use `n/a: no CPU kernel selected` for `CPU feature string`
and `n/a: static surface audit` for `Command + environment`; those values make
the non-measurement scope explicit. `present` and `n/a` require `Missing
behavior: none` and `Gate: none`. `partial` requires a concrete missing
behavior and non-`none` gate.

`Fixture hashes` uses immutable `sha256:<64-lowercase-hex>` values. A deferred
row may say `pending:` while it has no observed fixture.

## Current entries

The parity population is intentionally empty until the pinned reference
surface inventory is promoted into the truth pack. Adding a row without that
reference counterpart would make the matrix look more authoritative than its
evidence.
