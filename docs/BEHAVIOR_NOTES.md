<!-- fnlp-ledger-schema: behavior-notes/v1 -->

# Behavior notes

This ledger records intentional departures from the pinned reference behavior
and discovered upstream defects. It is not a numeric-divergence ledger: an
accepted measured numeric difference belongs in
[DISCREPANCIES.md](DISCREPANCIES.md). A behavior note never silently changes
the reference contract; it records the source, product decision, compatibility
effect, and the condition that requires reconsideration.

## Entry schema

Each `BN-...` entry contains these exact fields:

- `Source pin` — immutable file, revision, line span, and SHA-256.
- `Minimized fixture` — a named fixture or an explicit pending fixture gate.
- `Decision` and `Rationale` — the deliberate product behavior and why.
- `Compatibility impact` — what callers must not mistake for parity.
- `Revisit condition` — a concrete evidence or compatibility trigger.

Where an entry later scopes a public claim, it adds a `Claim ID` that resolves
through `docs/CLAIMS.json`. Until that registry lands, no behavior note may
invent a claim id merely to appear cross-linked.

## BN-GEN-DEFAULT-001

- Source pin: `generation_config.json@f56ec5a9650268aa098496734743c25ea778bd2d:1-6; sha256:68c690ce23efb6caae30c006ff3c1efd826297ff1df4338c04f7ac6f685d8746`
- Minimized fixture: `PENDING_PHASE_MINUS_1_GENERATION_DEFAULT_FIXTURE`; the
  fixture must compare the pinned configuration with the effective fnlp
  request and receipt.
- Decision: fnlp defaults to greedy generation. Upstream defaults to sampling with `temperature=0.6`, `top_k=20`, and `top_p=0.95`; callers opt into that recipe only with `--sample --preset nanbeige`.
- Rationale: greedy output is the reproducible product default and the only
  baseline that can support deterministic greedy parity claims before the
  separately specified sampling/replay contract is proved.
- Compatibility impact: this is an intentional product default, not HF
  sampling parity. Help text, receipts, and fixtures must report the effective
  sampling mode so a caller cannot mistake greedy output for upstream default
  sampling behavior.
- Revisit condition: revisit only when a pinned upstream configuration changes
  or a separately versioned sampling/replay contract proves and releases a new
  default through its own fixture and scorecard gates.
