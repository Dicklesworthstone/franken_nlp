# OQ-35 authority narrowing and typed-budget census record

Status: **scoped pin observations; no product adoption**

Related G0 record: `ADR-G0-11-asupersync-census.md` remains the aggregate
decision and must remain `BLOCKED` until it has retained feature-gated run
evidence for every census row.

## Decision

At the selected asupersync revision
`362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`, a child `Cx` can only be
statically narrowed through `restrict`, and an ambient `Cx::current()` lookup
under a `cap::None` scope has its runtime effects masked. The pin exposes
typed `Budget` and `CapabilityBudget` values. FrankenNLP adopts none of these
as a product alias or unit conversion in this record: pull, inference,
greedy, and leaf capability aliases, plus the project-unit-to-budget mapping,
remain implementation obligations.

## Census verdict index

| Census row | Existing executable probe | Exact emitted verdict | Decision scope |
| --- | --- | --- | --- |
| Static widening | `tests/g0/asupersync_census/compile_fail.rs` | `compile-fail-suite RESULT=RATIFIED` | A `Cx<cap::None>` cannot be retyped to `Cx<cap::All>` by the tested fixture. |
| Ambient current lookup | `tests/g0/asupersync_census/runtime_semantics.rs` | `current-no-regain RESULT=RATIFIED` | A `cap::None` restriction masks SPAWN, TIME, RANDOM, IO, and REMOTE at runtime, then restores the outer mask on scope exit. |
| Typed budget shapes | `tests/g0/asupersync_census/runtime_semantics.rs` | `budget-typed RESULT=RATIFIED` | The pin exposes the named budget types and their tested default fields. |

The `RATIFIED` words above are the individual probe outputs, not a statement
that FrankenNLP has adopted a complete authority or budget contract. In
particular, a `Cx::current()` value can still have the static `Cx<cap::All>`
type; the pin's protection for that path is its runtime mask. Product code
must preserve that distinction and must not claim a compile-time ambient
authority boundary.

## Residuals and fallback

There is no executable project-unit conversion for deadline, poll, cost,
memory, CPU, IO, cleanup, or artifact units in the current tree. Until one
exists with meet-composition and reserved-cleanup cases, FrankenNLP retains
explicit request counters and treats suite budgets as an unadopted substrate.
No inference leaf receives ambient `Cx::current()` authority as a substitute
for an explicit narrowed view.
