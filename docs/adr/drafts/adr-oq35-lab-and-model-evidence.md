# ADR draft: OQ-35 laboratory and bounded-model evidence

Status: **Proposed pre-write — decision not ratified**
Related census: `OQ35-01`, `OQ35-11`, `OQ35-12`
Suite pin: `8eb48575889c81b65f7556db4b26d47a8bc03197`

## Context

The pin exposes deterministic LabRuntime execution, trace/replay, seed-bound chaos, oracle suites, crashpack handling, VirtualTcp, and a bounded explorer described as DPOR-style. It also exposes TLA export code. Source presence does not establish FrankenNLP integration or exhaustive native-thread coverage.

## Proposed decision

If ratified, G0 uses the laboratory only for async state and lifecycle claims, with retained same-seed replay, oracle-suite, chaos, crashpack, and VirtualTcp receipts. Exploration is described as bounded guided coverage only. TLA export is an input producer only until an actual TLC run is retained.

Native `scoped_cpu` interleavings remain a separate bounded team-state model and hostile native-stress obligation. Laboratory results cannot claim exhaustive native OS-thread behavior.

## Required evidence before acceptance

| Claim | Required artifact |
| --- | --- |
| Deterministic replay | same seed, trace identity/replay result, and exact test revision |
| Lifecycle safety | quiescence, obligation leak, loser-drain, and cancellation-protocol output |
| Chaos/crash recovery | seed, injected action, crashpack, replay result, and recovery disposition |
| Virtual TCP | deterministic listener/stream scenario and ordering assertion |
| Bounded exploration | base seed, class count, run/step budget, coverage, saturation, violations, and saved traces |
| TLC bounded model | generated input, TLC version/config/command/result/property scope, and counterexample if found |

## Rejected interpretations

- A passing seed sweep is not exhaustive.
- DPOR-style source is not a completed DPOR proof.
- Generated TLA text is not a TLC result.
- Async lab replay is not proof of native `scoped_cpu` scheduling.

## Consequences

No Cargo adoption follows from this draft. The first executable work is limited to named G0 probes and the separate native-team model. A failed or absent surface records its scope and leaves the path closed or on an independently tested fallback.
