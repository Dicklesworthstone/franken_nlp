//! Bounded judge tasks using the shared native finite-continuation scorer.
//!
//! Judge outputs are uncalibrated model judgments, not factuality certificates.
//! Pairwise ranking scores both orders; rubric criteria score independently.
//! Raw-text callers use JudgePlanner and retain PreparedJudge's private identity
//! for admission. No CLI activation, model loader or qualification is implied.

mod common;
mod native;
mod pairwise;
mod planning;
mod rubric;

pub use common::{JudgeError, JudgeLimits, JudgeLogits};
pub use native::{EagerJudgeRun, JudgeNativeError, JUDGE_NATIVE_EXECUTION};
pub use pairwise::{PairwiseDecision, PairwisePlan, PairwisePolicy, PairwiseResult};
pub use planning::{JudgePlanner, JudgeRequest, JudgeResult, PreparedJudge,
    RubricCriterion, RubricDefinition, JUDGE_PROMPT_VERSION};
pub use rubric::{RubricCriterionResult, RubricDecision, RubricHeadInput, RubricPlan, RubricPolicy, RubricResult};
