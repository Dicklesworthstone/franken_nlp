//! Weightless Phase-0 G0 probe harnesses.
//!
//! The records in `docs/adr/ADR-G0-01-*`, `02-*`, `03-*`, and `09-*` remain
//! BLOCKED until these bounded probes are run against their real target
//! environments and their raw transcripts are appended to the ADR evidence
//! directories.  These tests lock the hostile fixture and state-model cases
//! that every eventual ratification run must retain.
#![deny(unsafe_code)]

#[path = "g0/probe1_https_matrix.rs"]
mod probe1_https_matrix;
#[path = "g0/probe2_seam.rs"]
mod probe2_seam;
#[path = "g0/probe3_broker.rs"]
mod probe3_broker;
#[path = "g0/probe9_fs_crash.rs"]
mod probe9_fs_crash;
