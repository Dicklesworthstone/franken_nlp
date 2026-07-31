# OQ-35 supervision and Suite-substrate non-adoption record

Status: **no adoption**

Related G0 record: `ADR-G0-11-asupersync-census.md` remains `BLOCKED`.

## Decision

No supervisor, registry/name-lease surface, `franken_kernel`,
`franken_evidence`, `franken_decision`, or `frankenlab` crate is adopted by
this census. Presence in the neighbouring FrankenSuite workspace is not a
FrankenNLP dependency decision. A later adoption requires a named product
use, source and compatibility review, an exact `SUITE.lock` entry, and a
replayed executable receipt.

## Exact-pin audit

This audit reads Git objects at the `SUITE.lock`-selected asupersync revision
`362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`, rather than inferring contracts
from a neighbouring checkout that may have advanced. All rows below are
therefore **present at the suite pin but absent from the FrankenNLP product**;
they are not implicit approvals to add a dependency or route a daemon through
the surface.

| Candidate | Exact source at the selected pin | Census decision and product fallback |
| --- | --- | --- |
| Supervision | `src/supervision.rs` exposes `SupervisionStrategy` (line 194) and `RestartConfig` (line 225). | No batch-daemon restart tree is designed, adopted, or replayed. Until one is, an owned request-region tree names its restart owner explicitly and preserves typed outcomes to the policy boundary. |
| Registry/name leases | `src/cx/registry.rs` exposes `RegistryCap` (line 52), `RegistryHandle` (line 56), `NameLease` (line 104), and `NameRegistry` (line 505). | No global registry or name lease is adopted. Product-local ownership remains explicit in the request region; any later registry use needs collision, lease-resolution, and shutdown evidence. |
| `franken-kernel` | `franken_kernel/Cargo.toml`: package `franken-kernel` 0.3.10, a type substrate for trace, decision, policy, and schema ids. | No typed-substrate gap has been assigned to this crate; existing project identities and receipts remain local until a narrow use and compatibility review justify an exact lock row. |
| `franken-evidence` | `franken_evidence/Cargo.toml`: package `franken-evidence` 0.3.10, canonical evidence-ledger schema. | FrankenNLP retains its ADR/receipt evidence boundary. No ledger-schema dependency is inferred from recording evidence in this repository. |
| `franken-decision` | `franken_decision/Cargo.toml`: package `franken-decision` 0.3.10; it depends on the two preceding substrate crates. | No task `DecisionPolicy` integration has a ratified product contract, so importing this crate would enlarge the adoption without a named use or replay evidence. |
| `frankenlab` | `frankenlab/Cargo.toml`: package/binary `frankenlab` 0.3.5, with an `asupersync` `test-internals` dependency. | The census uses the pinned asupersync laboratory feature directly for bounded tests; it does not adopt the standalone harness into the product or release graph. |

The source rows establish availability only. They do not ratify a daemon
supervision policy, a routable server, a global name authority, a
FrankenSuite evidence schema, or a decision runtime for FrankenNLP.

## SUITE.lock adoption note

`SUITE.lock` records the selected asupersync revision
`362dc5b174427f66cfa76ab2bdd68cce1a95c6cc` as an optional
`asupersync-runtime` dependency. It contains no dependency row for
`franken_kernel`, `franken_evidence`, `franken_decision`, or `frankenlab`.
This ADR intentionally makes **no generated-lock edit**: adding any one of
those rows would falsely imply an approved product adoption.

## Fallback

Until an adoption ADR is ratified, FrankenNLP uses an owned request-region
tree with an identified restart owner, explicit lifecycle acknowledgements,
and no implicit global registry. Evidence records remain local project data;
they do not acquire a suite-schema dependency by implication.
