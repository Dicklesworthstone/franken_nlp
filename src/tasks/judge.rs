//! Bounded judge tasks using the shared native finite-continuation scorer.
//!
//! Judge outputs are uncalibrated model judgments, not factuality certificates.
//! Pairwise ranking scores both presentation orders before making a decision.

mod common;
mod native;
mod pairwise;

pub use common::{JudgeError, JudgeLimits, JudgeLogits};
pub use native::{EagerJudgeRun, JudgeNativeError, JUDGE_NATIVE_EXECUTION};
pub use pairwise::{PairwiseDecision, PairwisePlan, PairwisePolicy, PairwiseResult};
