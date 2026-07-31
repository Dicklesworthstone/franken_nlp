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

Use `UPDATE_GOLDENS=1` nowhere here: claim registry updates are ordinary
reviewed JSON diffs. Run `scripts/check_claims.sh` to validate the registry,
its negative fixtures, and all currently present public surfaces.
