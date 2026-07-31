# ADR draft: OQ-35 execution, cancellation, and durability boundaries

Status: **Proposed pre-write — decision not ratified**
Related census: `OQ35-05` through `OQ35-10`
Suite pin: `8eb48575889c81b65f7556db4b26d47a8bc03197`

## Context

The pin offers presets, scoped CPU work, outcome combinators, bracket cleanup, GenServer delivery, and eleven cancellation kinds. Its contracts rule out shortcuts: `high_throughput` is not a CPU-team certificate; scoped spawn does not seal post-latch spawning; `first_ok` does not provide short-circuit mirror fallback; drop-time bracket cleanup is best effort; and `cast().await` is not a processing/commit acknowledgement.

## Proposed decision

If accepted, the engine forms and seals a fixed CPU team before any worker is released into fallible work. The process certificate accounts for runtime workers, maximum blocking coordinators, scoped children, and ratified helpers. `high_throughput` is never paired blindly with a physical-core-wide team.

Mirror fallback is explicit ordered `for`/`await`. No concurrent hedge is added without a first-success contract, duplicate-work/byte budget, and cancel-then-drain evidence for every loser.

Durable transitions use `GenServer::call` or reserve/permit plus processing/commit acknowledgement. Durable cleanup is RAII plus awaited cleanup, journal/recovery, and atomic activation. All eleven cancellation kinds remain typed until the product policy boundary.

## Required evidence before acceptance

| Boundary | Required proof |
| --- | --- |
| Team formation and drain | bounded team-state model, native stress, and latch/resource receipt |
| Preset envelope | exact preset behavior and process-thread inventory |
| Ordered fallback | result ordering and no unnecessary concurrent work |
| Any hedge | duplicate-work budget, loser cancellation, and loser drain |
| Durable transition | processing/commit acknowledgement and interrupted recovery |
| Queue overflow | declared policy and counted drop/reject receipt |
| Cancellation | exhaustive eleven-kind preservation fixture |

## Consequences

No scheduler topology or persistence protocol is adopted here. Until evidence accepts one, implementation stays on the stated fixed-team, ordered-fallback, and explicit-durability design.
