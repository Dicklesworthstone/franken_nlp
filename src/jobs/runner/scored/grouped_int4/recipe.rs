//! Private recipe projections derived from actual immutable task compilers.
use super::*;
use super::super::{ClassificationPlanningRecipe, SentimentPlanningRecipe, ScorerRecipe, check_budget};
use crate::{
    batch::{classify::ClassificationBatchArgs, judge::JudgeBatchArgs},
    jobs::manifest::bounded_json,
    native_engine::{lmhead::scoring::ScoringLimits, strict_int4::{Int4MemoryRequirement, Int4Work}},
    tasks::{
        classify::{ClassificationLimits, ClassificationMode, CLASSIFICATION_PROMPT_VERSION,
            grouped_int4::{Int4ClassificationPlanner, INT4_CLASSIFICATION_EXECUTION}},
        sentiment::{SentimentAxis, SentimentLimits, SENTIMENT_PROMPT_VERSION,
            batch::SentimentBatchArgs, grouped_int4::{Int4SentimentPlanner, INT4_SENTIMENT_EXECUTION}},
        judge::{JudgeLimits, JUDGE_PROMPT_VERSION,
            grouped_int4::{Int4JudgePlanner, INT4_JUDGE_EXECUTION}},
    },
};
use serde::Serialize;

const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// The module is private: only the three native planner implementations below
/// can derive a durable recipe. External callers cannot implement this bound.
pub trait Sealed: Int4ScoredTask {
    fn freeze_recipe(&self, defaults: Option<Self::Args>, native: Int4ScoredBatchLimits)
        -> Result<(Int4ScoredJobRecipe, Option<Self::Args>), BatchFault>;
}

/// Private content is serializable only for the existing keyed job commitment.
/// No Debug, Deserialize, public fields or constructor can become an authority
/// to change a model-bound factory after it has been created.
#[derive(Serialize)]
pub struct Int4ScoredJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    task_spec: &'static str,
    group_width: usize,
    task_ceiling: TaskBudget,
    native: NativeRecipe,
    settings: Settings,
}
#[derive(Serialize)]
struct NativeRecipe { max_item_work: Int4Work, max_model_work: Int4Work }
#[derive(Serialize)]
#[serde(tag = "task", rename_all = "snake_case")]
enum Settings {
    Classification { planning: ClassificationPlanningRecipe, defaults: Option<ClassificationBatchArgs> },
    Sentiment { planning: SentimentPlanningRecipe, defaults: Option<SentimentBatchArgs> },
    Judge { planning: JudgePlanningRecipe, defaults: Option<JudgeBatchArgs> },
}
#[derive(Serialize)]
struct JudgePlanningRecipe {
    per_head: ScorerRecipe, max_total_prompt_tokens: usize,
    max_total_candidate_tokens: usize, max_total_projected_logits: u64,
    max_output_bytes: u64,
}
impl From<JudgeLimits> for JudgePlanningRecipe {
    fn from(limits: JudgeLimits) -> Self {
        // Exhaustive destructuring forces a future field into resume authority.
        let JudgeLimits { per_head, max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes } = limits;
        Self { per_head: per_head.into(), max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes }
    }
}

impl Int4ScoredJobRecipe {
    fn finish(task_spec: &'static str, execution: &'static str, prompt_version: &'static str,
        group_width: usize, task_ceiling: TaskBudget, native: Int4ScoredBatchLimits, settings: Settings)
        -> Result<Self, BatchFault> {
        task_ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        native.validate()?;
        Int4MemoryRequirement::for_context(1, group_width).map_err(|_| BatchCode::InvalidLimits)?;
        let Int4ScoredBatchLimits { max_item_work, max_model_work } = native;
        let recipe = Self { version: 1, dependency_scope: "item-local",
            input_policy: "typed-scored-records-fixed-task-full-resident-kv-v1",
            execution, prompt_version, task_spec, group_width, task_ceiling,
            native: NativeRecipe { max_item_work, max_model_work }, settings };
        // Leave room for JobRunner's protocol and input-profile wrapper.
        // Defaults are still single-owned here; clone only after this bound.
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    pub(super) fn task_ceiling(&self) -> TaskBudget { self.task_ceiling }
    pub(super) fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if limits.max_work != JobWork::from_int4(self.native.max_model_work, 0)
            || self.task_ceiling.max_output_bytes > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub(super) fn classification(width: usize, ceiling: TaskBudget, limits: ClassificationLimits,
        defaults: Option<ClassificationBatchArgs>, native: Int4ScoredBatchLimits) -> Result<Self, BatchFault> {
        check_classification(limits, defaults.as_ref(), ceiling)?;
        Self::finish("classify-v1", INT4_CLASSIFICATION_EXECUTION, CLASSIFICATION_PROMPT_VERSION,
            width, ceiling, native, Settings::Classification { planning: limits.into(), defaults })
    }
    pub(super) fn sentiment(width: usize, ceiling: TaskBudget, limits: SentimentLimits,
        defaults: Option<SentimentBatchArgs>, native: Int4ScoredBatchLimits) -> Result<Self, BatchFault> {
        check_scoring(limits.per_axis)?;
        if limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
            || limits.max_total_nodes == 0 || limits.max_total_projected_logits == 0 || limits.max_output_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        if let Some(args) = &defaults {
            check_budget(args.budget, ceiling)?;
            if args.axes.is_empty() || args.axes.len() > SentimentAxis::ALL.len()
                || args.axes.iter().enumerate().any(|(i, axis)| args.axes[..i].contains(axis)) {
                return Err(BatchCode::InvalidLimits.into());
            }
        }
        Self::finish("sentiment-v1", INT4_SENTIMENT_EXECUTION, SENTIMENT_PROMPT_VERSION,
            width, ceiling, native, Settings::Sentiment { planning: limits.into(), defaults })
    }
    pub(super) fn judge(width: usize, ceiling: TaskBudget, limits: JudgeLimits,
        defaults: Option<JudgeBatchArgs>, native: Int4ScoredBatchLimits) -> Result<Self, BatchFault> {
        check_scoring(limits.per_head)?;
        if limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
            || limits.max_total_projected_logits == 0 || limits.max_output_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        if let Some(args) = &defaults {
            let budget = match args {
                JudgeBatchArgs::Pairwise { criterion, b, budget, .. } => {
                    if criterion.is_empty() || b.is_empty() { return Err(BatchCode::InvalidLimits.into()); }
                    *budget
                }
                JudgeBatchArgs::Rubric { rubric, budget, .. } => {
                    rubric.validate().map_err(|_| BatchCode::InvalidLimits)?;
                    *budget
                }
                JudgeBatchArgs::Faithfulness { claim, policy, budget } => {
                    if claim.is_empty() { return Err(BatchCode::InvalidLimits.into()); }
                    policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
                    *budget
                }
            };
            check_budget(budget, ceiling)?;
        }
        Self::finish("judge-v1", INT4_JUDGE_EXECUTION, JUDGE_PROMPT_VERSION,
            width, ceiling, native, Settings::Judge { planning: limits.into(), defaults })
    }
}

impl Sealed for Int4ClassificationPlanner {
    fn freeze_recipe(&self, defaults: Option<ClassificationBatchArgs>, native: Int4ScoredBatchLimits)
        -> Result<(Int4ScoredJobRecipe, Option<ClassificationBatchArgs>), BatchFault> {
        let (ceiling, limits) = self.durable_configuration();
        let recipe = Int4ScoredJobRecipe::classification(self.group_width(), ceiling, limits, defaults, native)?;
        let defaults = match &recipe.settings { Settings::Classification { defaults, .. } => defaults.clone(),
            _ => return Err(BatchCode::InvalidExecution.into()) };
        Ok((recipe, defaults))
    }
}
impl Int4ScoredJobTask for Int4ClassificationPlanner {}
impl Sealed for Int4SentimentPlanner {
    fn freeze_recipe(&self, defaults: Option<SentimentBatchArgs>, native: Int4ScoredBatchLimits)
        -> Result<(Int4ScoredJobRecipe, Option<SentimentBatchArgs>), BatchFault> {
        let (ceiling, limits) = self.durable_configuration();
        // Actual mode/EOS/policy already live in this planner's template digest.
        // No second caller-supplied options object can diverge from them.
        let recipe = Int4ScoredJobRecipe::sentiment(self.group_width(), ceiling, limits, defaults, native)?;
        let defaults = match &recipe.settings { Settings::Sentiment { defaults, .. } => defaults.clone(),
            _ => return Err(BatchCode::InvalidExecution.into()) };
        Ok((recipe, defaults))
    }
}
impl Int4ScoredJobTask for Int4SentimentPlanner {}
impl Sealed for Int4JudgePlanner {
    fn freeze_recipe(&self, defaults: Option<JudgeBatchArgs>, native: Int4ScoredBatchLimits)
        -> Result<(Int4ScoredJobRecipe, Option<JudgeBatchArgs>), BatchFault> {
        let (ceiling, limits) = self.durable_configuration();
        let recipe = Int4ScoredJobRecipe::judge(self.group_width(), ceiling, limits, defaults, native)?;
        let defaults = match &recipe.settings { Settings::Judge { defaults, .. } => defaults.clone(),
            _ => return Err(BatchCode::InvalidExecution.into()) };
        Ok((recipe, defaults))
    }
}
impl Int4ScoredJobTask for Int4JudgePlanner {}

fn check_scoring(limits: ScoringLimits) -> Result<(), BatchFault> {
    let ScoringLimits { max_candidates, max_total_tokens, max_nodes, max_depth,
        max_candidate_id_bytes, max_projected_logits } = limits;
    if max_candidates == 0 || max_total_tokens == 0 || max_nodes == 0 || max_depth == 0
        || max_candidate_id_bytes == 0 || max_projected_logits == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn check_classification(limits: ClassificationLimits, defaults: Option<&ClassificationBatchArgs>, ceiling: TaskBudget)
    -> Result<(), BatchFault> {
    check_scoring(limits.scoring)?;
    if !(1..=4096).contains(&limits.max_labels) || !(1..=64 * 1024 * 1024).contains(&limits.max_input_bytes)
        || !(1..=1024).contains(&limits.max_label_id_bytes) || limits.max_label_description_bytes > 1024 * 1024
        || !(1..=8 * 1024 * 1024).contains(&limits.max_total_label_bytes)
        || !(1..=8192).contains(&limits.max_context_tokens)
        || !(1..=16 * 1024 * 1024).contains(&limits.max_total_prompt_tokens)
        || limits.max_work.forward_positions == 0 || limits.max_work.projected_logits == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    if let Some(args) = defaults {
        check_budget(args.budget, ceiling)?;
        if args.labels.is_empty() || args.labels.len() > limits.max_labels
            || (args.mode == ClassificationMode::Exclusive && args.labels.len() < 2)
            || args.policy.minimum_candidate_weight_ppm > 1_000_000 || args.policy.minimum_margin_ppm > 1_000_000 {
            return Err(BatchCode::InvalidLimits.into());
        }
        let mut ids = std::collections::BTreeSet::new(); let mut bytes = 0_usize;
        for label in &args.labels {
            bytes = bytes.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
                .ok_or(BatchCode::InvalidLimits)?;
            if label.id.trim().is_empty() || label.id.len() > limits.max_label_id_bytes
                || label.id.chars().any(char::is_control) || label.description.len() > limits.max_label_description_bytes
                || bytes > limits.max_total_label_bytes || !ids.insert(label.id.as_str()) {
                return Err(BatchCode::InvalidLimits.into());
            }
        }
    }
    Ok(())
}
