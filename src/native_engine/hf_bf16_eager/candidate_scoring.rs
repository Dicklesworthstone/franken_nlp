//! Concrete native finite-candidate execution over an already-admitted engine.
//!
//! `classify_eager_cached` uses one prefill, destructive KV prefix rewind and
//! true selected-row projection. The existing `classify_eager` replay route is
//! retained unchanged as an explicit reference/fallback surface; work receipts
//! distinguish the two algorithms. Neither route loads or activates a model,
//! spawns a runtime, clones model weights, or claims a measured speedup.

use super::{HfBf16EagerEngine, HfBf16EagerError};

#[path = "candidate_scoring/replay.rs"]
mod replay;
pub use replay::{
    EagerClassificationError, EagerClassificationRun, ReplayBudget, ReplayWork,
    classify_eager, classify_eager_with_control,
};
pub(crate) use replay::ClearCache;

pub mod prefix;
pub use prefix::{
    CachedClassificationRun, EagerPrefixSession, PREFIX_EXECUTION_VERSION,
    PrefixBudget, PrefixScoringError, PrefixWork,
    classify_eager_cached, classify_eager_cached_with_control,
};

struct Continue;
impl crate::native_engine::decode::DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<crate::native_engine::decode::DecodeCancellationKind> { None }
}
