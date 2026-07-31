# FrankenNLP campaign wiring evidence

Bead: `franken_nlp-vsx`
Recorded: 2026-07-31T09:58:51Z

This is the bootstrap record for the Code-First / Batch-Verify campaign. It is
an evidence record, not authority to run an individual build: ordinary panes do
not run Cargo commands or `rch exec`. The sole batch verifier uses its own
orchestrator-owned `CARGO_TARGET_DIR`; it is the only exempt target directory.

The flip from a code-first wave to central batch verification is driven by the
campaign's own ready-pool depth, in-flight-bead count, commit flow, and build
queue. The historical 20–40 commits per ten minutes and 20 → 12 → 5 saturation
shape are context only, not a FrankenNLP target. A central verifier closes a
bead only with the combined green suite and that bead's named contract evidence;
code-first commits remain `in_progress` until then.

## Required transcripts

### RCH diagnostic

```text
timestamp: 2026-07-31T09:45Z
command: rch doctor
exit_code: 0
key_output: ✓ 29 passed
key_output: ✓ All checks passed!
BOOTSTRAP rch-doctor RESULT=PASS detail=29 diagnostic checks passed
```

### RCH observed capacity

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

### Degraded-capacity fallback policy

The owner-approved fallback treats an RCH posture of `degraded` as workable for
this campaign when the scheduler still reports 11 of 12 workers healthy. It
authorizes the code-first wave and the orchestrator-owned central batch verifier
only; ordinary panes remain prohibited from running Cargo or `rch exec`. The
orchestrator must price scheduling from the observed healthy-worker and slot
counts, avoid dispatch that depends on `yto`, `hz1`, `hz2`, or `vmi1293453`, and
record a fresh `rch status` before every central verification wave. Worker
remediation remains operational debt, not a fabricated healthy-capacity claim.

### Beads graph and `bv` input

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

### MCP Agent Mail round trip

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

Every pane registers before work, reserves exact paths using the bead id as the
reason, and announces under that bead's Agent Mail thread. If mail is degraded
after two attempts, the `br update <id> --assignee <agent> --status in_progress`
assignee is the lock so implementation does not stall.

## Verdict

```text
BOOTSTRAP campaign-wiring RESULT=PASS-WITH-NOTE detail=owner-approved 11/12-worker degraded fallback; central verifier rechecks capacity before each wave
```
