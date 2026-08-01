# FrankenNLP campaign wiring evidence

Historical bootstrap bead: `franken_nlp-vsx`
Historical record time: 2026-07-31T09:58:51Z
Current authority bead: `franken_nlp-dsr-authority-docs-rul7`

## Current build and release authority

Ordinary swarm panes run no Cargo, RCH, DSR, or GitHub Actions command. Direct
RCH and GitHub Actions runs are non-authoritative even when green. After a
code-first wave has quiesced, the controller may select one clean immutable
commit and request an occasional DSR checkpoint. The DSR recipe must run
`scripts/check.sh` against the explicitly named Cargo feature graph
`production`; a default-empty graph and `--all-features` are not substitutes.

The required retained DSR terminal receipt schema is:

```text
DSR_CHECKPOINT source_sha=<40-lower-hex> source_tree=clean production_graph=production entrypoint=scripts/check.sh dsr_run_id=<stable-id> result=PASS|FAIL
```

At the current implementation snapshot, that receipt cannot yet be minted:

- `Cargo.toml` does not define the named `production` feature;
- `scripts/check.sh` invokes Cargo on the default feature graph rather than the
  named release graph;
- the release-graph dependency-policy script has not landed, so the check
  entrypoint cannot yet reject Rayon or multiple `asupersync` sources in the
  selected product closure; and
- `.github/workflows/ci.yml` still auto-triggers the non-authoritative check on
  pushes and pull requests, contrary to the DSR-only operating rule. The live
  repository API also reported Actions `enabled=true` with
  `allowed_actions=all`; recent pushes created cancelled CI runs. The workflow
  must be made inert and repository Actions disabled through separately
  assigned implementation and external-setting authority. This documentation
  record does not itself authorize either mutation or deletion.

These are setup blockers, not reasons to run the default graph and annotate the
result. No DSR execution should be requested until one immutable commit contains
the exact graph and fail-closed checks described above.

The receipt also retains the expanded recipe/command, DSR version, dated Rust
toolchain identity, host and target triples, start/end times, exit code, the
complete `scripts/check.sh` transcript, and its final `CHECK RESULT=PASS|FAIL`
line. `PASS` is invalid if the source SHA is not the clean commit selected by
the controller, the production aggregate is absent or not selected, the
release graph contains Rayon, or a required policy leg was skipped. A failed or
incomplete setup is `BLOCKED`/`FAIL`, never `PASS-WITH-NOTE`.

Build compatibility is only one evidence class. Model-present L-gates and
artifact/inference smoke, target-host performance and platform-native behavior,
and human review/authorization remain separate. The DSR code receipt cannot
promote any of them by implication.

A release job has additional fail-closed prerequisites: DSR repository/target
registration; healthy selected build hosts; the pinned signing-key fingerprint
and protected signing capability; SBOM and SLSA generators; exact explicit
asset inventory; and online plus network-denied verification of the exported
signature/provenance bundle. Missing tooling or key material blocks release; it
does not weaken the recipe or convert publisher authentication into checksums.

GitHub release immutability is another prerequisite, not a substitute for that
DSR provenance bundle. The repository API reported `enabled=false` and
`enforced_by_owner=false` for immutable releases during the 2026-07-31 audit,
so the first publication remains blocked until the owner enables the setting
and the release recipe retains a fresh state query. The recipe creates a draft,
uploads only the explicit receipted inventory, verifies the remote inventory,
and publishes only after every other gate passes. Publication must make the tag
and assets immutable and create GitHub's release attestation; the retained
online checks are `gh release verify <tag>` plus one
`gh release verify-asset <tag> <local-path>` invocation for every asset. This
is GitHub's immutable-release attestation, not a GitHub Actions build
attestation, and it does not authorize an Actions workflow.

The canonical model part size, 1,957,046,720 bytes, is 190,436,928 bytes below
GitHub's currently documented strict 2-GiB per-asset ceiling. The release job
still rechecks the live limit and rejects every non-tail part whose length is
not exact; a changed hosting policy blocks publication rather than silently
changing the artifact recipe.

The controller also retains a fresh coordination snapshot from the same
checkpoint window:

1. `br ready --json` with its actual schema and ready ids;
2. `br sync --flush-only` followed by `bv --robot-insights` over that exact
   JSONL, with zero dependency cycles and no unresolved dead-closed blockers;
3. a live MCP Agent Mail register → exact-path reserve → release round trip.

If Mail is degraded, bead-assignee locking may keep non-overlapping code-first
work moving, but it does not satisfy this live wiring receipt. Proof-sensitive
closures remain blocked until the live round trip is restored or the owner
ratifies a different authority contract.

## Historical, non-authoritative bootstrap transcripts

Everything below is retained as campaign history. It proves only what was
observed on 2026-07-31; it does not authorize a current build, close, or
release.

### Historical RCH diagnostic

```text
timestamp: 2026-07-31T09:45Z
command: rch doctor
exit_code: 0
key_output: ✓ 29 passed
key_output: ✓ All checks passed!
BOOTSTRAP rch-doctor RESULT=PASS detail=29 diagnostic checks passed
```

### Historical RCH observed capacity

```text
timestamp: 2026-07-31T09:58:51Z
command: rch status
exit_code: 0
key_output: Posture : degraded (Some workers unhealthy, partial remote capability)
key_output: Workers : 11/12 healthy, 86/94 slots available
key_output: Worker 'yto' is offline: Health probe failed: Command timed out after 10s (yto)
key_output: Worker 'hz1' in critical pressure state (disk_critical_without_fresh_telemetry)
key_output: Worker 'hz2' in critical pressure state (disk_critical_without_fresh_telemetry)
key_output: Worker 'vmi1293453' is unreachable
BOOTSTRAP rch-status RESULT=PASS-WITH-NOTE detail=owner-approved degraded fallback: 11/12 workers healthy and 86/94 slots available
```

### Historical degraded-capacity fallback policy

At the time, the owner-approved fallback treated an RCH posture of `degraded`
as workable for that historical campaign when the scheduler reported 11 of 12
workers healthy. That ruling no longer grants build or closure authority. The
host/slot observations remain useful diagnostic history only; any current build
or release must enter through the DSR contract above.

### Historical Beads graph and `bv` input

```text
timestamp: 2026-07-31T09:45Z
command: br ready --json
exit_code: 0
key_output: {"ready_count":1,"ready_ids":["franken_nlp-ilz"]}
BOOTSTRAP br-ready RESULT=PASS detail=ready graph returned one bead at observation time

timestamp: 2026-07-31T09:45:47Z
command: bv --robot-insights
exit_code: 0
key_output: {"cycles":null,"cycle_count":0,"generated_at":"2026-07-31T09:45:47Z"}
key_output: advanced_insights.cycle_break.cycle_count = 0
BOOTSTRAP bv-cycles RESULT=PASS detail=no dependency cycles detected
```

`bv` reads `.beads/issues.jsonl`; that graph input is produced from the Beads
database by `br sync --flush-only`. No agent stages broad working-tree changes:
the intended Beads export is staged explicitly by the coordinator.

### Historical MCP Agent Mail round trip

```text
timestamp: 2026-07-31T09:43:41Z
command: macro_start_session(project=/Users/jemanuel/projects/franken_nlp)
exit_code: 0
key_output: registered agent=PlumGoose

timestamp: 2026-07-31T09:46:11Z
command: file_reservation_paths(path=.agent-mail-vsx-roundtrip-probe, reason=franken_nlp-vsx)
exit_code: 0
key_output: granted id=13013 exclusive=true
command: release_file_reservations(file_reservation_ids=[13013])
exit_code: 0
key_output: released=1
BOOTSTRAP agent-mail-roundtrip RESULT=PASS detail=register reserve and release completed
```

Every pane still registers before work, reserves exact paths using the bead id
as the reason, and announces under that bead's Agent Mail thread. If mail is
degraded after the permitted attempts, the
`br update <id> --assignee <agent> --status in_progress` assignee is the
non-overlap lock so implementation does not stall. The old round trip above is
not a substitute for the fresh checkpoint-window wiring requirement.

## Historical verdict (retained, non-authoritative)

```text
BOOTSTRAP campaign-wiring RESULT=PASS-WITH-NOTE detail=owner-approved 11/12-worker degraded fallback; central verifier rechecks capacity before each wave
```

## Current verdict

```text
DSR_AUTHORITY RESULT=BLOCKED detail=exact clean-SHA DSR PASS|FAIL receipt for scripts/check.sh on production plus live br/bv/Mail wiring not yet retained
```
