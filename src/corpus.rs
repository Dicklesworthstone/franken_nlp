//! Bounded corpus and long-document task composition. These surfaces reuse
//! native task implementations; they do not create a loader, runtime, durable
//! job, cache authority, or a second model backend.

pub mod entities;
pub mod native_resolve;
pub mod native_summary;
pub mod resolution_stream;
pub mod resolve;
pub mod summarize;
pub use crate::tasks::corpus_keyphrases as keyphrases;
