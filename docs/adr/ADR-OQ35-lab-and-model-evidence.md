# OQ-35 laboratory and bounded-model census record

Status: **partial laboratory observation; no exhaustive or native-thread claim**

Related G0 record: `ADR-G0-11-asupersync-census.md` remains `BLOCKED`. The
selected asupersync revision is
`362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`.

## Decision

The current probe suite demonstrates individual LabRuntime facilities only.
It does not turn a same-seed workload comparison into a retained replay
receipt, a bounded explorer into exhaustive coverage, or a crashpack object
into a FrankenNLP production crashpack format. Lab evidence is restricted to
async lifecycle state; native `scoped_cpu` interleavings still need their own
bounded team-state model and hostile native stress.

## Census verdict index

| Census row | Existing executable probe | Exact emitted verdict | Decision scope |
| --- | --- | --- | --- |
| Same-seed run shape | `tests/g0/asupersync_census/lab_determinism.rs` | `lab-determinism RESULT=RATIFIED` | The probe compares two freshly constructed cooperative workloads with the same seed. |
| Leak policy values | `tests/g0/asupersync_census/lab_determinism.rs` | `obligation-leak-policy RESULT=RATIFIED` | The examined Lab and runtime builder configurations expose panic/log responses. |
| Crashpack material | `tests/g0/asupersync_census/lab_determinism.rs` | `lab-crashpack RESULT=RATIFIED` | An intentional obligation leak yields a failing report, divergent prefix, and replay command metadata. |
| DPOR-style explorer | `tests/g0/asupersync_census/explorer_tla.rs` | `dpor-explorer RESULT=RATIFIED` | The finite two-task workload records run, class, race, backtrack, sleep-pruning, and saturation fields under a fixed seed/run budget; its scope is explicitly bounded guided coverage, never exhaustiveness. |
| TLA+ export | `tests/g0/asupersync_census/explorer_tla.rs` | `tla-export RESULT=RATIFIED` | A non-empty Lab trace produces a behavior module and bounded skeleton. The emitted verdict says `tlc=NOT_RUN`; neither artifact is a TLC result. |

## Residuals and fallback

The aggregate census still needs a retained raw transcript/digest for each
emitted `G0_CENSUS` verdict, retained replay execution, named oracle-suite
results, seed-bound chaos replay, VirtualTcp behavior, and any TLC
version/config/command/result/property scope/counterexample before their
stronger claims may be used. The committed DPOR probe is a source-level
finite-coverage assertion, not its retained execution receipt; the committed
TLA+ probe is export-only, not a model-check. Until then, product work keeps
an owned bounded team-state model for native threads and records laboratory
observations as partial supporting evidence only.
