//! Bounded corpus and long-document task composition. These surfaces reuse
//! native task implementations; they do not create a loader, runtime, durable
//! job, cache authority, or a second model backend.

pub mod native_summary;
pub mod resolve;
pub mod summarize;
pub use crate::tasks::corpus_keyphrases as keyphrases;
