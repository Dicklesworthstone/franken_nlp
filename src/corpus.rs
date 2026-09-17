//! Bounded long-document task composition. These surfaces reuse the native
//! task implementations and the shared map/reduce spine; they do not create
//! a loader, runtime, durable job, cache authority, or a second model backend.

pub mod native_summary;
pub mod summarize;
pub use crate::tasks::corpus_keyphrases as keyphrases;
