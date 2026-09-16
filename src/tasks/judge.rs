//! Bounded judge tasks using the shared native finite-continuation scorer.
//!
//! Judge outputs are uncalibrated model judgments, not factuality certificates.
//! Pairwise ranking scores both presentation orders before making a decision.

mod common;
mod pairwise;

pub use common::{JudgeError, JudgeLimits, JudgeLogits};
pub use pairwise::{PairwiseDecision, PairwisePlan, PairwisePolicy, PairwiseResult};
