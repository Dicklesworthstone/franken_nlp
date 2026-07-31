# G0-11 asupersync leverage census

```adr-metadata
{
  "adr_id": "G0-11",
  "blocked_surface": ["scheduler", "daemon", "jobs", "pull proofs"],
  "decision": "At the SUITE.lock-selected asupersync pin, the feature-gated G0-11 probes establish only the listed capability, combinator, GenServer, Lab, crashpack, preset, and cancellation observations. Ambient Cx::current() is not a least-authority boundary at this pin; scheduler, daemon, jobs, and pull proofs remain blocked until every OQ-35 row has retained evidence and a project-level adoption decision.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0_asupersync_census --features asupersync-census -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "asupersync leverage census", "probe": 11},
  "host_pin": {"applicability": "pin-specific semantic census; no host-sensitive performance conclusion"},
  "killed_alternatives": [{"name": "ambient Cx::current() as a least-authority boundary", "reason": "Cx::current() returns Cx<cap::All>; the capability metadata snapshot does not control the underlying now/random/spawn handles"}, {"name": "short-circuit first_ok mirror", "reason": "the pin drives all ExecPlan::first_ok children before input-order selection"}, {"name": "cast acknowledgement as commit", "reason": "cast().await and try_cast acknowledge mailbox admission only"}],
  "source_pin": {"asupersync": "362dc5b174427f66cfa76ab2bdd68cce1a95c6cc"},
  "status": "BLOCKED",
  "x_executable_verdicts": [{"commit": "8dce5ac369ec8cdecd68ac1e17ccd2a17e89509a", "items": ["cancelkind-eleven", "first-ok-sequential", "budget-typed", "preset-values", "execplan-first-ok"], "path": "tests/g0/asupersync_census/runtime_semantics.rs", "scope": "historical pin-scoped observations only; its current-no-regain assertion is superseded and must not be replayed as ambient-authority evidence"}, {"commit": "f234750aacf27ffe0c5ea53a43d0f6cf5f167fe9", "items": ["ambient-current-authority"], "path": "tests/g0/asupersync_census/runtime_semantics.rs", "scope": "corrected ABSENT_WITH_FALLBACK observation: Cx::current() retains static all capability and the restricted metadata snapshot is not effect enforcement"}, {"commit": "2f23773b05c3108b7cbf2168d74a4e7014fb8d8f", "items": ["cast-async-ack", "try-cast-policies"], "path": "tests/g0/asupersync_census/gen_server_semantics.rs", "scope": "Lab mailbox acknowledgement and declared overflow observations"}, {"commit": "f7f2076f1c2b12a25417afeba9163f52a48b9d97", "items": ["lab-determinism", "obligation-leak-policy", "lab-crashpack"], "path": "tests/g0/asupersync_census/lab_determinism.rs", "scope": "Lab replay, leak-policy, and crashpack material observations"}, {"commit": "41d0aa36b1cc3e3bef4928c909623f1b10feb141", "items": ["compile-fail-suite"], "path": "tests/g0/asupersync_census/compile_fail.rs", "scope": "static narrowing and absent generic current API only; no ambient authority proof"}, {"commit": "37e4dbaab1aa7912929e13a8f50c40ba5128bff7", "items": ["dpor-explorer", "tla-export"], "path": "tests/g0/asupersync_census/explorer_tla.rs", "scope": "bounded DPOR-style coverage and TLA export only; no exhaustive or TLC claim"}],
  "x_unratified_rows": ["ambient Cx::current() least-authority enforcement (ABSENT; explicit narrowed Cx parameter and no ambient leaf lookup)", "retained raw G0_CENSUS transcript and evidence digest", "project capability aliases and budget-unit conversion", "project two-phase reservation and production leak-escalation policy", "Lab oracle/chaos/VirtualTcp coverage", "bracket normal-versus-drop evidence", "bounded exploration/TLC retained artifacts", "supervision/registry/substrate adoption decisions"]
}
```

## Pin-scoped executable observations

The selected suite pin is
`asupersync@362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`. `SUITE.lock` records
that it is the selected revision for the optional `asupersync-runtime` feature
and retains the audited-plan revision separately. This ADR adopts neither a
new dependency nor a new feature: it documents the evidence boundary of the
already locked suite selection.

The feature-gated `g0_asupersync_census` target contains executable pin
observations and absence records for the following areas:

| Area | Observed contract | Executable probe |
| --- | --- | --- |
| Authority | `restrict` cannot widen statically. Ambient `Cx::current()` is `Cx<cap::All>` and its capability snapshot is metadata, not effect enforcement; the required fallback is an explicit narrowed `Cx` parameter with no ambient leaf lookup. | `runtime_semantics.rs`, `compile_fail.rs` |
| Outcomes | `first_ok_outcomes` classifies completed outcomes in input order; `ExecPlan::first_ok` drives all children. | `runtime_semantics.rs` |
| Messaging | `cast().await` and accepted `try_cast` acknowledge enqueue only; `Reject` reports full and `DropOldest` is explicitly lossy. | `gen_server_semantics.rs` |
| Cancellation/budgets/presets | Eleven `CancelKind` values, typed budget shapes, and concrete runtime preset values are observable. | `runtime_semantics.rs` |
| Lab | Same-seed replay, explicit leak policy, and failed-run crashpack replay material are observable. | `lab_determinism.rs` |
| Bounded exploration/model export | DPOR-style exploration records finite run/class/race/backtrack/saturation coverage; TLA+ behavior and skeleton export are inputs only. | `explorer_tla.rs` |

## Complete verdict ledger — source inventory, not acceptance evidence

This table is the complete OQ-35 item inventory as of the recorded source
state. `RATIFIED (source-only)` means only that the hash-addressed committed
probe expresses the stated pin observation; it is **not** a target execution,
retained raw transcript, DSR result, closure manifest, independent review, or
product-adoption decision. `ABSENT_WITH_FALLBACK` means no such surface exists
in this tree and names the required project-level replacement. Nothing in this
table authorizes a bead closure or a dependent scheduler, daemon, job, pull,
oracle-fixture, or parity claim.

Source-digest key:

| Key | SHA-256 | Source |
| --- | --- | --- |
| R | `08a1021d83e728115e151a00d84061a87907654f9451b1a96a87f3a5a8c15568` | `tests/g0/asupersync_census/runtime_semantics.rs` |
| C | `abae26b3c891eec6d1286320fd04914b9d981966cbada64c51ec16f078f9bd50` | `tests/g0/asupersync_census/compile_fail.rs` |
| W | `a4fe807d006e05bf26c48c222829698fc0654c4af0fc514b0270d1aa00d89f0a` | `tests/g0/compile_fail/capability_widening.rs` |
| N | `a60d3a1da485b6cfc7313def50a146052d6ec0ec15bb14cc5be8d76584b9c724` | `tests/g0/compile_fail/cx_current_regain.rs` |
| G | `85b88cdca830fed52ddc6cd46080128d8dc31109753b04dab278de05cf652567` | `tests/g0/asupersync_census/gen_server_semantics.rs` |
| L | `a985054524f5c0faa410a62dbdf8ecc502220954e023149f149c3f016603334d` | `tests/g0/asupersync_census/lab_determinism.rs` |
| E | `06f4a3d0323650704cf91d21a375cf2227e644b22fbe395c875e6e987cfef490` | `tests/g0/asupersync_census/explorer_tla.rs` |

`no artifact` is itself a negative inventory result, not a digest omission: no
committed source or retained receipt exists to bind the corresponding claim.

| Contract item | Verdict | Source digest or gap | Required fallback / limit |
| --- | --- | --- | --- |
| Static `Cx::restrict` narrowing | RATIFIED (source-only) | C + W | Pass an explicit narrowed `Cx`; no product alias is adopted. |
| Ambient `Cx::current()` least authority | ABSENT_WITH_FALLBACK | R | Explicit narrowed `Cx` parameter; prohibit ambient leaf lookup. |
| Project capability aliases (pull/inference/greedy/leaf) | ABSENT_WITH_FALLBACK | no artifact | Define and compile-check project aliases before use. |
| Typed `Budget` and `CapabilityBudget` shapes | RATIFIED (source-only) | R | Pin type presence only; no complete project budget claim. |
| Project unit conversion and typed-budget meet mapping | ABSENT_WITH_FALLBACK | no artifact | Checked project-unit conversion plus cleanup-reserve cases. |
| Same-seed Lab replay shape | RATIFIED (source-only) | L | Async-Lab observation only; no native-team inference. |
| Lab oracle suite, seed-bound chaos, loser drain, cancellation protocol, and `VirtualTcp` | ABSENT_WITH_FALLBACK | no artifact; L does not cover these rows | Add project Lab fixtures and retained replay receipts. |
| Pin leak-policy values | RATIFIED (source-only) | L | Builder observation only. |
| Project two-phase reservation/leak-escalation policy | ABSENT_WITH_FALLBACK | no artifact | Owned reservation ledger; panic in lab/CI and logged production escalation. |
| Runtime preset values | RATIFIED (source-only) | R | Concrete values only; no safe CPU-team sizing decision. |
| `first_ok_outcomes` classification semantics | RATIFIED (source-only) | R | Classification is not live sequential execution. |
| `ExecPlan::first_ok` all-child behavior | RATIFIED (source-only) | R | Not a first-success primitive; no loser cancellation claim. |
| Pull sequential mirror fallback | ABSENT_WITH_FALLBACK | no artifact | Explicit ordered `for`/`await` with per-attempt budgets. |
| `bracket` normal/drop durability split | ABSENT_WITH_FALLBACK | no artifact | RAII plus explicit await, journal/recovery, and atomic activation. |
| `GenServer::cast` enqueue acknowledgement | RATIFIED (source-only) | G | Enqueue is not processing or commit acknowledgement. |
| `GenServer::try_cast` reject/drop-oldest policy | RATIFIED (source-only) | G | Use only declared-lossy streams with counted drops. |
| Eleven `CancelKind` variants | RATIFIED (source-only) | R | Preserve variants until the product policy boundary. |
| Bounded DPOR-style exploration | RATIFIED (source-only) | E | Finite guided coverage only; never exhaustive. |
| TLA+ export | RATIFIED (source-only) | E | Export is input, not a model-check result. |
| TLC run, version/config/command/result/property/counterexample | ABSENT_WITH_FALLBACK | no artifact | Retain the complete TLC result bundle before a bounded-model-check claim. |
| Batch-daemon supervision, restart policy, and registry leases | ABSENT_WITH_FALLBACK | no artifact | Explicit region tree and hand-rolled restart-on-error until separately adopted. |
| Workspace substrates (`franken_kernel`, `franken_evidence`, `franken_decision`, `frankenlab`) | ABSENT_WITH_FALLBACK | no artifact | No silent import; per-substrate audit, ADR, and `SUITE.lock` decision. |
| Raw `G0_CENSUS` transcript and evidence digest | ABSENT_WITH_FALLBACK | no artifact | Retained hash-addressed transcript plus independent review; remains a closure prerequisite. |

These are narrower than product readiness. In particular, `budget-typed` does
not freeze a FrankenNLP project-unit conversion, and `obligation-leak-policy`
does not adopt a production reservation/escalation protocol. The static
compile-fail fixture and the runtime `Cx::current()` observation are
deliberately separate. The latter is an `ABSENT_WITH_FALLBACK` authority row,
not a ratified runtime least-authority mechanism.

## Supplemental adoption boundaries

The following pin-scoped ADRs consume the individual census verdicts without
changing this aggregate record's status. Together they keep the distinction
between an observed suite facility and a FrankenNLP product adoption explicit:

| Scope | Supplemental record | Adoption boundary |
| --- | --- | --- |
| Capability narrowing and budgets | `ADR-OQ35-authority-and-budget.md` | No project capability aliases or project-unit conversion is adopted. |
| Execution, cancellation, and durability | `ADR-OQ35-execution-and-durability.md` | The ordered `for`/`await` pull fallback and acknowledged durable transitions remain required. |
| Lab and bounded-model evidence | `ADR-OQ35-lab-and-model-evidence.md` | Lab observations do not prove native-team interleavings or exhaustive exploration. |
| Supervision and workspace substrates | `ADR-OQ35-supervision-and-substrates.md` | No supervisor, registry, or substrate crate is adopted; the generated `SUITE.lock` intentionally gains no substrate row. |

In particular, the last record is an adoption note, not a lock-file mutation:
`SUITE.lock` continues to record only the already selected optional
`asupersync-runtime` dependency for this census. Any future substrate adoption
requires its own product use, compatibility review, generated-lock update, and
replayed receipt.

## Evidence and remaining gate

The executable probes are committed and name their intended replay command in
metadata, but this tree has no retained raw `G0_CENSUS` transcript under
`docs/adr/evidence/G0-11/`. Therefore the ADR stays `BLOCKED`: a source line
or a passing batch summary is not a substitute for the hash-addressed evidence
artifact required by the ADR schema. In particular, the bounded DPOR source
test is not a retained coverage receipt, and the TLA+ export source test is
not a TLC run. The retained transcript must be added before changing either
this ADR or the matching registry row to `RATIFIED`.

The remaining blocked rows are listed in `x_unratified_rows` above. Their
fallback remains the explicit project implementations and tests required by
the OQ-35 bead; no scheduler, daemon, job, or pull surface may inherit a
stronger claim from this partial census. In particular, leaf authority is
passed explicitly and ambient `Cx::current()` is not an admissible substitute.
