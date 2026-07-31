# OQ-35 execution, cancellation, and durability census record

Status: **scoped pin observations; no scheduler or durable-protocol adoption**

Related G0 record: `ADR-G0-11-asupersync-census.md` remains `BLOCKED`. This
record captures what the existing probes say about the selected asupersync
revision `362dc5b174427f66cfa76ab2bdd68cce1a95c6cc`; it does not certify an
engine team, a pull implementation, or a durable job protocol.

## Decision

The existing pin semantics require an explicit ordered mirror fallback and
acknowledged durable transitions. `first_ok_outcomes` classifies completed
outcomes in input order; `ExecPlan::first_ok` drives all children before
selecting the first input-order success. `cast().await` and `try_cast` admit
messages only; neither is a processing or commit acknowledgement. All eleven
`CancelKind` values remain policy-boundary input rather than being collapsed
early.

## Census verdict index

| Census row | Existing executable probe | Exact emitted verdict | Decision scope |
| --- | --- | --- | --- |
| Cancellation taxonomy | `tests/g0/asupersync_census/runtime_semantics.rs` | `cancelkind-eleven RESULT=RATIFIED` | The eleven pinned variants are constructed and round-tripped by the probe. |
| Ordered outcome selection | `tests/g0/asupersync_census/runtime_semantics.rs` | `first-ok-sequential RESULT=RATIFIED` | The classification surface is input-order sequential and stops at cancel/panic. |
| Concurrent plan behavior | `tests/g0/asupersync_census/runtime_semantics.rs` | `execplan-first-ok RESULT=RATIFIED` | The plan drives every child and does not cancel a loser. |
| Preset values | `tests/g0/asupersync_census/runtime_semantics.rs` | `preset-values RESULT=RATIFIED` | Constructed runtimes expose the recorded worker and steal-batch values. |
| `cast` acknowledgement | `tests/g0/asupersync_census/gen_server_semantics.rs` | `cast-async-ack RESULT=RATIFIED` | Awaiting the cast acknowledges enqueue, not handling. |
| `try_cast` policies | `tests/g0/asupersync_census/gen_server_semantics.rs` | `try-cast-policies RESULT=RATIFIED` | The probe distinguishes reject-full from declared DropOldest behavior. |

## Residuals and fallback

No record above supplies the required fixed-team formation, blocking-pool
envelope, post-latch spawn seal, cancellation-and-draining receipt, or
end-to-end durable recovery proof. `fnlp pull` therefore remains specified as
an explicit ordered `for`/`await` fallback, and durable state transitions
remain owned `call` or reserve/permit plus processing/commit acknowledgement.
No concurrent hedge, preset-derived CPU width, or drop-time cleanup claim is
authorized by these probes.
