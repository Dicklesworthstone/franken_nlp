# ADR draft: OQ-35 authority narrowing and executable budgets

Status: **Proposed pre-write — decision not ratified**
Related census: `OQ35-02`, `OQ35-03`, `OQ35-04`
Suite pin: `8eb48575889c81b65f7556db4b26d47a8bc03197`

## Context

At the pin, capabilities are sealed and type-level (`SPAWN`, `TIME`, `RANDOM`, `IO`, `REMOTE`); `Cx::restrict` only narrows a context. The budget surfaces separate deadline/poll/cost from memory, abstract CPU units, I/O, cleanup, and artifact dimensions. Lab configuration can make obligation leaks panic or log. Product aliases, executable unit conversions, and compile-fail evidence do not yet exist in FrankenNLP.

## Proposed decision

If accepted, every product path gets a named narrowed capability alias and a meet-composed child budget.

| Region | Required authority boundary |
| --- | --- |
| Pull | only demonstrated spawn/time/IO and TLS entropy needs; never REMOTE |
| Ordinary inference | no network or remote authority |
| Greedy inference | no RANDOM |
| Kernel leaves | `cap::None` or the ratified checkpoint-only view; cannot spawn |

Every public/request limit maps to executable deadline, poll, cost, memory, CPU, I/O, cleanup, and artifact envelopes. Token, output, and page counters remain explicit. Cleanup receives reserved capacity.

## Required evidence before acceptance

- Compile-fail fixtures for widening and current-context recovery attempts.
- Nested restriction regression.
- Auditable mapping for every timed and resource envelope.
- Meet-composition cases proving no child relaxes a parent limit.
- Leaked-reservation evidence: panic in lab/CI and logged escalation in production.

## Consequences

No authority or budget claim follows from this draft. If the pin cannot express a required alias or checked mapping, FrankenNLP implements that boundary explicitly and records the suite surface as not adopted for it.
