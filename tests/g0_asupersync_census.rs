//! G0 probe 11 — OQ-35 asupersync leverage census target (franken_nlp-idt).
//!
//! Requires `--features asupersync-runtime`; the default suite never builds
//! this target (declared with `required-features` in Cargo.toml). Every test
//! emits `G0_CENSUS item=<name> RESULT=...` lines against the pinned
//! asupersync revision so verdicts are observations, not memory.
#[path = "g0/asupersync_census/runtime_semantics.rs"]
mod runtime_semantics;
