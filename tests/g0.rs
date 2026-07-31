//! Weightless Phase-0 G0 probe harnesses.
//!
//! The records in `docs/adr/ADR-G0-01-*`, `02-*`, `03-*`, and `09-*` remain
//! BLOCKED until these bounded probes are run against their real target
//! environments and their raw transcripts are appended to the ADR evidence
//! directories.  These tests lock the hostile fixture and state-model cases
//! that every eventual ratification run must retain.
#![deny(unsafe_code)]

#[path = "g0/probe10_avx2_exact.rs"]
mod probe10_avx2_exact;
#[path = "g0/probe1_https_matrix.rs"]
mod probe1_https_matrix;
#[path = "g0/probe2_seam.rs"]
mod probe2_seam;
#[path = "g0/probe3_broker.rs"]
mod probe3_broker;
#[path = "g0/probe4_tok_tpl.rs"]
mod probe4_tok_tpl;
#[path = "g0/probe5_loop_boundary.rs"]
mod probe5_loop_boundary;
#[path = "g0/probe6_mask_memory.rs"]
mod probe6_mask_memory;
#[path = "g0/probe7_reduction_order.rs"]
mod probe7_reduction_order;
#[path = "g0/probe8_converter_rss.rs"]
mod probe8_converter_rss;
#[path = "g0/probe9_fs_crash.rs"]
mod probe9_fs_crash;
