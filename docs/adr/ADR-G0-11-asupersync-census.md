# G0-11 asupersync leverage census

```adr-metadata
{
  "adr_id": "G0-11",
  "blocked_surface": ["scheduler", "daemon", "jobs", "pull proofs"],
  "decision": "At the SUITE.lock-selected asupersync pin, the feature-gated G0-11 probes establish only the listed capability, combinator, GenServer, Lab, crashpack, preset, and cancellation observations; scheduler, daemon, jobs, and pull proofs remain blocked until every OQ-35 row has a retained transcript and a project-level adoption decision.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0_asupersync_census --features asupersync-census -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "asupersync leverage census", "probe": 11},
  "host_pin": {"applicability": "pin-specific semantic census; no host-sensitive performance conclusion"},
  "killed_alternatives": [{"name": "ambient capability re-expansion", "reason": "Cx::current() is runtime-masked under cap::None"}, {"name": "short-circuit first_ok mirror", "reason": "the pin drives all ExecPlan::first_ok children before input-order selection"}, {"name": "cast acknowledgement as commit", "reason": "cast().await and try_cast acknowledge mailbox admission only"}],
  "source_pin": {"asupersync": "362dc5b174427f66cfa76ab2bdd68cce1a95c6cc"},
  "status": "BLOCKED",
  "x_executable_verdicts": [{"commit": "8dce5ac369ec8cdecd68ac1e17ccd2a17e89509a", "items": ["cancelkind-eleven", "first-ok-sequential", "budget-typed", "preset-values", "current-no-regain", "execplan-first-ok"], "path": "tests/g0/asupersync_census/runtime_semantics.rs", "scope": "pin-scoped executable observations"}, {"commit": "2f23773b05c3108b7cbf2168d74a4e7014fb8d8f", "items": ["cast-async-ack", "try-cast-policies"], "path": "tests/g0/asupersync_census/gen_server_semantics.rs", "scope": "Lab mailbox acknowledgement and declared overflow observations"}, {"commit": "f7f2076f1c2b12a25417afeba9163f52a48b9d97", "items": ["lab-determinism", "obligation-leak-policy", "lab-crashpack"], "path": "tests/g0/asupersync_census/lab_determinism.rs", "scope": "Lab replay, leak-policy, and crashpack material observations"}, {"commit": "41d0aa36b1cc3e3bef4928c909623f1b10feb141", "items": ["compile-fail-suite"], "path": "tests/g0/asupersync_census/compile_fail.rs", "scope": "static narrowing proof; current-context runtime masking is separately covered above"}],
  "x_unratified_rows": ["retained raw G0_CENSUS transcript and evidence digest", "project capability aliases and budget-unit conversion", "project two-phase reservation and production leak-escalation policy", "Lab oracle/chaos/VirtualTcp coverage", "bracket normal-versus-drop evidence", "bounded exploration/TLC retained artifacts", "supervision/registry/substrate adoption decisions"]
}
```

## Pin-scoped executable observations

The selected suite pin is
`asupersync@362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`. `SUITE.lock` records
that it is the selected revision for the optional `asupersync-runtime` feature
and retains the audited-plan revision separately. This ADR adopts neither a
new dependency nor a new feature: it documents the evidence boundary of the
already locked suite selection.

The feature-gated `g0_asupersync_census` target contains executable
`G0_CENSUS ... RESULT=RATIFIED` lines for the following pin observations:

| Area | Observed contract | Executable probe |
| --- | --- | --- |
| Authority | `restrict` cannot widen; an ambient `Cx<cap::All>` remains runtime-masked under `cap::None`. | `runtime_semantics.rs`, `compile_fail.rs` |
| Outcomes | `first_ok_outcomes` classifies completed outcomes in input order; `ExecPlan::first_ok` drives all children. | `runtime_semantics.rs` |
| Messaging | `cast().await` and accepted `try_cast` acknowledge enqueue only; `Reject` reports full and `DropOldest` is explicitly lossy. | `gen_server_semantics.rs` |
| Cancellation/budgets/presets | Eleven `CancelKind` values, typed budget shapes, and concrete runtime preset values are observable. | `runtime_semantics.rs` |
| Lab | Same-seed replay, explicit leak policy, and failed-run crashpack replay material are observable. | `lab_determinism.rs` |

These are narrower than product readiness. In particular, `budget-typed` does
not freeze a FrankenNLP project-unit conversion, and `obligation-leak-policy`
does not adopt a production reservation/escalation protocol. The static
compile-fail fixture and the runtime `Cx::current()` probe are deliberately
separate proof tracks.

## Evidence and remaining gate

The executable probes are committed and name their intended replay command in
metadata, but this tree has no retained raw `G0_CENSUS` transcript under
`docs/adr/evidence/G0-11/`. Therefore the ADR stays `BLOCKED`: a source line
or a passing batch summary is not a substitute for the hash-addressed evidence
artifact required by the ADR schema. The retained transcript must be added
before changing either this ADR or the matching registry row to `RATIFIED`.

The remaining blocked rows are listed in `x_unratified_rows` above. Their
fallback remains the explicit project implementations and tests required by
the OQ-35 bead; no scheduler, daemon, job, or pull surface may inherit a
stronger claim from this partial census.
