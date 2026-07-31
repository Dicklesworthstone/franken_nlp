# OQ-35 authority narrowing and typed-budget census record

Status: **scoped pin observations; ambient-authority claim withdrawn; no product adoption**

Related G0 record: `ADR-G0-11-asupersync-census.md` remains the aggregate
decision and must remain `BLOCKED` until it has retained feature-gated run
evidence for every census row.

## Decision

At the selected asupersync revision
`362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`, a child `Cx` can only be
statically narrowed through `restrict`. Ambient `Cx::current()` nevertheless
returns `Cx<cap::All>`; its capability snapshot is metadata and does not
enforce reduced effects on the underlying now/random/spawn handles. The pin
also exposes typed `Budget` and `CapabilityBudget` values. FrankenNLP adopts
none of these as a product alias or unit conversion in this record: pull,
inference, greedy, and leaf capability aliases, plus the
project-unit-to-budget mapping, remain implementation obligations.

## Census verdict index

| Census row | Existing executable probe | Exact emitted verdict | Decision scope |
| --- | --- | --- | --- |
| Static widening | `tests/g0/asupersync_census/compile_fail.rs` | `compile-fail-suite RESULT=RATIFIED` | A `Cx<cap::None>` cannot be retyped to `Cx<cap::All>` by the tested fixture. |
| Ambient current lookup | `tests/g0/asupersync_census/runtime_semantics.rs` | `ambient-current-authority RESULT=ABSENT_WITH_FALLBACK` | The earlier `current-no-regain` result is withdrawn: it tested metadata booleans only. The fallback is an explicit narrowed `Cx` parameter and a ban on ambient leaf lookup. |
| Typed budget shapes | `tests/g0/asupersync_census/runtime_semantics.rs` | `budget-typed RESULT=RATIFIED` | The pin exposes the named budget types and their tested default fields. |

The `RATIFIED` words above are the individual static/budget probe outputs, not
a statement that FrankenNLP has adopted a complete authority or budget
contract. In particular, an ambient `Cx::current()` value has the static
`Cx<cap::All>` type and is not protected by an enforceable runtime authority
mask. Product code must pass explicit narrowed contexts and must not claim an
ambient authority boundary.

## Residuals and fallback

There is no executable project-unit conversion for deadline, poll, cost,
memory, CPU, IO, cleanup, or artifact units in the current tree. Until one
exists with meet-composition and reserved-cleanup cases, FrankenNLP retains
explicit request counters and treats suite budgets as an unadopted substrate.
No inference leaf receives ambient `Cx::current()` authority as a substitute
for an explicit narrowed view. This remains a BLOCKED foundation condition for
all dependent scheduler, daemon, job, and pull proof claims.
