# OQ-35 asupersync leverage census — pre-write

Status: **DRAFT — source census only; no item is ratified**
Owner: `franken_nlp-idt`
Suite pin: `8eb48575889c81b65f7556db4b26d47a8bc03197`
Source access rule: every observation below was read with `git show` at the listed commit, never from the ambient asupersync checkout HEAD.

## Decision boundary

This is a source-level inventory for OQ-35, not evidence that FrankenNLP uses or proves any facility. A row becomes `RATIFIED` only when its named G0 probe produces a retained artifact and its ADR is accepted. Until then, the product uses the stated fallback or keeps the surface closed.

| Census item | Pin observation | Pinned source | Required G0 evidence | Draft verdict | FNLP gate or fallback |
| --- | --- | --- | --- | --- | --- |
| OQ35-01 laboratory/replay/VirtualTcp | LabRuntime has seeded scheduling, virtual time, trace/replay, chaos, oracle suites, crashpack exports, and VirtualTcp. | `src/lab/mod.rs`; `src/lab/runtime.rs`; `src/net/tcp/virtual_tcp.rs` | `tests/g0/asupersync_lab_replay.rs` receipt for seed, oracles, chaos, crashpack, and VirtualTcp | OBSERVED@pin; unratified | Admit only after same-seed replay and oracle results. Scope is async lab state, never native CPU-team interleavings. |
| OQ35-02 least authority | Sealed type-level rows are `[SPAWN,TIME,RANDOM,IO,REMOTE]`; `Cx::restrict` is monotone and `Cx::current` applies the innermost restriction. | `src/cx/cap.rs`; `src/cx/cx.rs` | compile-fail fixtures plus nested-current regression | OBSERVED@pin; unratified | Pull gets only demonstrated spawn/time/IO/TLS-entropy needs and never REMOTE; inference has no network; greedy/leaves no RANDOM; leaves are `None` or checkpoint-only. |
| OQ35-03 budgets | `Budget` has deadline/poll/cost; `CapabilityBudget` has memory, abstract CPU, I/O, cleanup, and artifact dimensions, tightened by meet. | `src/types/budget.rs` | executable mapping/meet receipt and cleanup-reserve cases | OBSERVED@pin; unratified | No prose-only ms budget. Keep request token/output/page counters explicit and provide a checked product mapping. |
| OQ35-04 obligations | LabRuntime selects panic or log obligation-leak response from configuration. | `src/lab/runtime.rs`; `src/types/budget.rs` | lab/CI leak-panic test and production escalation contract | OBSERVED@pin; policy pending | Panic in lab/CI; log and escalate in production. Never leave a two-phase reservation untracked. |
| OQ35-05 presets/envelope | `current_thread` asks for one worker; `low_latency` tunes polling/steal; `high_throughput` doubles deterministic default workers. | `src/runtime/builder.rs` | preset contract and process-wide thread-envelope receipt | OBSERVED@pin; unratified | Never use `high_throughput()` as authority for a physical-core-wide scoped team. Inventory runtime workers, blocking coordinators, scoped children, and helpers together. |
| OQ35-06 scoped CPU seam | `scoped_cpu` joins children and belongs in synchronous code or `spawn_blocking`; its cap does not create a post-latch spawn seal. | `src/cx/scoped_cpu.rs` | bounded native team-state model and hostile native stress | OBSERVED@pin; hand-rolled protocol required | Form a fixed team before fallible work, seal it, then release workers. Resources stay charged through actual join/latch completion. |
| OQ35-07 `first_ok` | `first_ok_outcomes` classifies an already-complete ordered vector. `ExecPlan::first_ok` drives all children then chooses first input-order success. | `src/combinator/first_ok.rs`; `src/plan/execute.rs` | outcome-order and loser-lifecycle cases | OBSERVED@pin; unsuitable for ordered mirror fallback | Pull mirrors use explicit ordered `for`/`await`. Hedging is closed pending a first-success primitive, duplicate-work budget, and cancel-then-drain proof. |
| OQ35-08 bracket/durable cleanup | `bracket` releases on normal awaited completion; drop-time release is bounded best effort. | `src/combinator/bracket.rs` | cleanup ordering and interrupted-recovery receipts | OBSERVED@pin; unratified | Durable work requires RAII plus explicit await, journal/recovery, and atomic activation; drop alone is no durability claim. |
| OQ35-09 GenServer delivery | `call` waits for tracked reply; `cast().await` is enqueue only; `try_cast` follows declared reject/drop-oldest policy and tracks evictions. | `src/gen_server.rs` | acknowledgement and overflow receipts | OBSERVED@pin; unratified | State/output/journal transitions use `call` or reserve/permit plus processing/commit acknowledgement. Loss is declared and counted. |
| OQ35-10 cancellations | Exactly eleven `CancelKind` variants are present. | `src/types/cancel.rs` | exhaustive policy fixture | OBSERVED@pin; unratified | Preserve all eleven kinds through the library/CLI policy boundary. |
| OQ35-11 DPOR-style explorer | Explorer documents bounded seed sweeps and DPOR-style direction; reports include run/class/violation/coverage/saturation data. | `src/lab/explorer.rs` | exact seed/class/run/step/coverage/saturation receipt | OBSERVED@pin; bounded only | Never call exploration exhaustive. Record budget and saturation with every claim. |
| OQ35-12 TLA/TLC | A TLA export surface exists. Source presence is not a TLC execution. | `src/trace/tla_export.rs` | input, TLC version/config/command/result/property scope/counterexample | UNRATIFIED | No bounded-model-check claim without every named TLC artifact. |
| OQ35-13 supervision/registry | Supervisor policies and registry/name leases exist; lease resolution is explicit. | `src/supervision.rs`; `src/cx/registry.rs` | adoption ADR and restart/lease tests, or fallback-region-tree tests | OBSERVED@pin; adoption undecided | Adopt only through a narrow ADR. Otherwise use an owned region tree with explicit restart ownership. |
| OQ35-14 substrate crates | `franken-kernel`, `franken-evidence`, `franken-decision`, and `frankenlab` exist in the workspace. | four workspace `Cargo.toml` files | `SUITE.lock`, dependency review, and per-crate adoption ADR | PRESENT; no adoption | Add no Cargo dependency from this census. Each candidate needs a versioned lock and ratified use case. |

## Evidence receipt shape

Each completed row appends an immutable G0 receipt containing:

```text
suite_pin=8eb48575889c81b65f7556db4b26d47a8bc03197
census_item=OQ35-XX
source_paths=<pinned paths>
probe=<test or model command>
result=PASS|FAIL|ABSENT
scope=<exercised and excluded behavior>
artifact_digest=<sha256 or PENDING>
```

`ABSENT` is valid only when it names the hand-rolled fallback. `PASS` from a source read alone is invalid.

## Native-thread boundary

OQ35 retains two non-substitutable proof tracks: deterministic async lab/replay/obligation/loser-drain evidence, and a bounded native team-state model plus hostile native stress for fixed-team formation, cancellation, joining, and resource-latch ownership.
