# ADR draft: OQ-35 supervision, registry, and suite-substrate adoption

Status: **Proposed pre-write — decision not ratified**
Related census: `OQ35-13`, `OQ35-14`
Suite pin: `8eb48575889c81b65f7556db4b26d47a8bc03197`

## Context

The pin contains supervisor policies and registry/name-lease surfaces. The workspace also contains `franken-kernel`, `franken-evidence`, `franken-decision`, and `frankenlab`. Workspace presence does not amend the closed dependency universe.

## Proposed decision

No supervision, registry, or substrate crate is adopted by this draft. A later ADR may adopt one named, versioned surface after a narrow product use case, source review, compatibility review, `SUITE.lock` entry, and executable evidence. There is no blanket FrankenSuite adoption.

If adoption does not pass, the fallback is an owned supervised request-region tree with an identified restart owner and no implicit global registry behavior.

## Required evidence before acceptance

| Candidate | Acceptance requirement |
| --- | --- |
| Supervisor | selected restart policy, panic/cancel behavior, child ownership, and replayed restart test |
| Registry/name lease | lifetime, resolution/abort, collision, and shutdown test |
| `franken-kernel` | exact types used, lock entry, and no broader transitive authority |
| `franken-evidence` | receipt-schema compatibility and evidence-retention review |
| `franken-decision` | decision-contract mapping and versioning review |
| `frankenlab` | feature/dependency review and reproducible lab integration proof |

## Consequences

This document authorizes neither a Cargo dependency nor a registry runtime. Until a later ADR is accepted, product code remains on explicit owned regions and local evidence records.
