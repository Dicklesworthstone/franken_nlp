//! Durable finite scoring on the SAME resident native batch adapters.
//! Recipes are immutable private projections, committed with the job secret.
//! No generated-label shortcut, alternate scorer or public confidence claim.
use serde::Serialize;
use crate::{
    batch::{BatchCode, BatchFault, BatchWork, classify::ClassificationBatchArgs},
    jobs::manifest::bounded_json,
    native_engine::{lmhead::scoring::ScoringLimits, strict_int8::Int8Work},
    tasks::{classify::{ClassificationLimits, CLASSIFICATION_PROMPT_VERSION,
            quantized::INT8_CLASSIFICATION_EXECUTION},
        ir::TaskBudget,
        sentiment::{SentimentLimits, SENTIMENT_PROMPT_VERSION,
            batch::{SentimentBatchArgs, SentimentBatchConfig}, quantized::INT8_SENTIMENT_EXECUTION}},
};

mod classify;
mod sentiment;
pub use classify::{Int8ClassificationJobPlanner, Int8ClassificationJobProcessor};
pub use sentiment::{Int8SentimentJobPlanner, Int8SentimentJobProcessor};

const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// Exhaustive projection: a new scorer limit must be deliberately frozen here.
#[derive(Serialize)]
struct ScorerRecipe {
    max_candidates: usize, max_total_tokens: usize, max_nodes: usize,
    max_depth: usize, max_candidate_id_bytes: usize, max_projected_logits: u64,
}
impl From<ScoringLimits> for ScorerRecipe {
    fn from(limits: ScoringLimits) -> Self {
        let ScoringLimits { max_candidates, max_total_tokens, max_nodes, max_depth,
            max_candidate_id_bytes, max_projected_logits } = limits;
        Self { max_candidates, max_total_tokens, max_nodes, max_depth,
            max_candidate_id_bytes, max_projected_logits }
    }
}
#[derive(Serialize)]
struct ClassificationPlanningRecipe {
    max_labels: usize, max_input_bytes: usize, max_label_id_bytes: usize,
    max_label_description_bytes: usize, max_total_label_bytes: usize,
    max_context_tokens: usize, max_total_prompt_tokens: usize,
    max_work: BatchWork, scoring: ScorerRecipe,
}
impl From<ClassificationLimits> for ClassificationPlanningRecipe {
    fn from(limits: ClassificationLimits) -> Self {
        let ClassificationLimits { max_labels, max_input_bytes, max_label_id_bytes,
            max_label_description_bytes, max_total_label_bytes, max_context_tokens,
            max_total_prompt_tokens, max_work, scoring } = limits;
        Self { max_labels, max_input_bytes, max_label_id_bytes, max_label_description_bytes,
            max_total_label_bytes, max_context_tokens, max_total_prompt_tokens,
            max_work, scoring: scoring.into() }
    }
}
#[derive(Serialize)]
struct SentimentPlanningRecipe {
    per_axis: ScorerRecipe, max_total_prompt_tokens: usize, max_total_candidate_tokens: usize,
    max_total_nodes: usize, max_total_projected_logits: u64, max_output_bytes: u64,
}
impl From<SentimentLimits> for SentimentPlanningRecipe {
    fn from(limits: SentimentLimits) -> Self {
        let SentimentLimits { per_axis, max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_nodes, max_total_projected_logits, max_output_bytes } = limits;
        Self { per_axis: per_axis.into(), max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_nodes, max_total_projected_logits, max_output_bytes }
    }
}

/// Labels/descriptions/default policies are private settings. Only the existing
/// domain-separated HMAC recipe commitment is persisted by JobRunner.
#[derive(Serialize)]
pub struct ClassificationJobRecipe {
    version: u32, dependency_scope: &'static str, execution: &'static str,
    prompt_version: &'static str, task_ceiling: TaskBudget,
    planning: ClassificationPlanningRecipe, max_model_work: Int8Work,
    defaults: Option<ClassificationBatchArgs>,
}
impl ClassificationJobRecipe {
    fn new(ceiling: TaskBudget, planning: ClassificationLimits,
        defaults: Option<ClassificationBatchArgs>, max_model_work: Int8Work) -> Result<Self, BatchFault> {
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        check_work(max_model_work)?;
        let recipe = Self { version: 1, dependency_scope: "item-local",
            execution: INT8_CLASSIFICATION_EXECUTION, prompt_version: CLASSIFICATION_PROMPT_VERSION,
            task_ceiling: ceiling, planning: planning.into(), max_model_work, defaults };
        // Bound retained settings BEFORE making the native compiler's clone.
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        if let Some(args) = &recipe.defaults {
            check_budget(args.budget, ceiling)?;
            let l = &recipe.planning;
            if args.labels.is_empty() || args.labels.len() > l.max_labels
                || (args.mode == crate::tasks::classify::ClassificationMode::Exclusive && args.labels.len() < 2)
                || args.policy.minimum_candidate_weight_ppm > 1_000_000 || args.policy.minimum_margin_ppm > 1_000_000 {
                return Err(BatchCode::InvalidLimits.into());
            }
            let mut ids = std::collections::BTreeSet::new(); let mut bytes = 0_usize;
            for label in &args.labels {
                bytes = bytes.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
                    .ok_or(BatchCode::InvalidLimits)?;
                if label.id.trim().is_empty() || label.id.len() > l.max_label_id_bytes
                    || label.id.chars().any(char::is_control) || label.description.len() > l.max_label_description_bytes
                    || bytes > l.max_total_label_bytes || !ids.insert(label.id.as_str()) {
                    return Err(BatchCode::InvalidLimits.into());
                }
            }
        }
        Ok(recipe)
    }
}

/// Score mode, EOS and sentiment policy are already part of the actual pinned
/// planner's template digest in ExecutionIdentity. They cannot be supplied as
/// a second, possibly divergent recipe. Defaults and every limit freeze here.
#[derive(Serialize)]
pub struct SentimentJobRecipe {
    version: u32, dependency_scope: &'static str, execution: &'static str,
    prompt_version: &'static str, task_ceiling: TaskBudget,
    planning: SentimentPlanningRecipe, max_item_work: Int8Work,
    max_model_work: Int8Work, defaults: Option<SentimentBatchArgs>,
}
impl SentimentJobRecipe {
    fn new(config: &SentimentBatchConfig) -> Result<Self, BatchFault> {
        // Validation by the real batch compiler precedes this constructor.
        let recipe = Self { version: 1, dependency_scope: "item-local", execution: INT8_SENTIMENT_EXECUTION,
            prompt_version: SENTIMENT_PROMPT_VERSION, task_ceiling: config.task_ceiling,
            planning: config.planning.into(), max_item_work: config.max_item_work,
            max_model_work: config.max_model_work, defaults: config.defaults.clone() };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}
fn check_budget(b: TaskBudget, ceiling: TaskBudget) -> Result<(), BatchFault> {
    b.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes > ceiling.max_kv_bytes { return Err(BatchCode::InvalidLimits.into()); }
    Ok(())
}
fn check_work(work: Int8Work) -> Result<(), BatchFault> {
    if work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

#[cfg(test)] mod tests;
