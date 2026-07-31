# Public claim annotations

`docs/CLAIMS.json` is the single registry for public benchmark, quality,
portability, security, determinism, and artifact claims. It is canonical JSON;
changing its schema or any entry is a deliberate, reviewable migration.

Public surfaces carry a machine annotation before their first claim-bearing line:

```text
<!-- fnlp-claim: claim-id; wording=targeted -->
```

Rust help/schema templates use the equivalent ordinary comment:

```rust
// fnlp-claim: claim-id; wording=targeted
```

The annotation remains active until another annotation in the same file. Its
`claim-id` must exist in [CLAIMS.json](CLAIMS.json), the file's surface must be
listed in that entry's `public_surfaces`, and `wording` must not outrun the
registry's state. The ordered wording tiers are `targeted`, `observed`, and
`evidenced`; a `withdrawn` claim may only be labeled `withdrawn`.

This is intentionally annotation-based: the checker does not claim it can infer
natural-language equivalence. Instead it rejects every numeric or superlative
public line without an active annotation and compares the explicit wording tier
against the registered state. An annotation labeled `targeted` marks a
target-state specification; it does not manufacture evidence.

## R4 long-context practicality claims

A public line that makes a context claim above the default 8,192-token cap
needs the ordinary `fnlp-claim` annotation at `wording=evidenced` **and** this
single-use companion annotation immediately before the line:

```text
<!-- fnlp-r4-context: ledger=PERF-EXACT-R4-ROW -->
```

The named ledger row must be an `R4-long-context` row with `Disposition: won`
and immutable fixture digests. This binds a positive >8K practicality claim to
both an evidenced public claim id and the measured R4 receipt. An explicit
`observed model limit` is not a practicality claim: it may report an upstream
position limit, but must never be worded as a usability, throughput, or
admission promise without the R4 pair.

Use `UPDATE_GOLDENS=1` nowhere here: claim registry updates are ordinary
reviewed JSON diffs. Run `scripts/check_claims.sh` to validate the registry,
its negative fixtures, and all currently present public surfaces.
