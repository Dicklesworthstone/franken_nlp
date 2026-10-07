//! Typed bulk-judgment defaults, shared by live streams and retained jobs.
//! Source text remains in each record. The host supplies every default budget.
use super::*;
use crate::{batch::judge::JudgeBatchArgs, tasks::judge::{PairwisePolicy, RubricDefinition,
    RubricPolicy, FaithfulnessPolicy}};

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Pairwise { criterion: String, b: String, policy: PairwisePolicy },
    Rubric { rubric: RubricDefinition, policy: RubricPolicy },
    Faithfulness { claim: String, policy: FaithfulnessPolicy },
}

pub(super) fn defaults(value: Option<serde_json::Value>, host: &ScoredArgs, budget: TaskBudget)
    -> Result<Defaults, CandidateError> {
    let Some(value) = value else { return Ok(Defaults::Judge(None)); };
    let input: Input = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    let text = |s: &str| !s.is_empty() && s.len() <= host.max_input_bytes;
    let args = match input {
        Input::Pairwise { criterion, b, policy } => {
            if !text(&criterion) || !text(&b) { return Err(CandidateError::Input); }
            // u32 milli-log-odds: do not reinterpret these values as ppm.
            JudgeBatchArgs::Pairwise { criterion, b, policy, budget }
        }
        Input::Rubric { rubric, policy } => {
            rubric.validate().map_err(|_| CandidateError::Input)?;
            if policy.minimum_peak_weight_ppm > 1_000_000 || policy.maximum_normalized_entropy_ppm > 1_000_000 {
                return Err(CandidateError::Input);
            }
            JudgeBatchArgs::Rubric { rubric, policy, budget }
        }
        Input::Faithfulness { claim, policy } => {
            if !text(&claim) { return Err(CandidateError::Input); }
            policy.validate().map_err(|_| CandidateError::Input)?;
            // The actual record's FULL source and evidence partition are
            // validated by the real task planner, never by a dummy document.
            JudgeBatchArgs::Faithfulness { claim, policy, budget }
        }
    };
    Ok(Defaults::Judge(Some(args)))
}

#[cfg(test)] mod tests;
