//! A pinned text planner derived from actual loaded grouped-INT4 model facts.
use super::*;
use crate::{execution_identity::{ThinkingMode, ToolMode},
    native_engine::strict_int4::STRICT_INT4_EXECUTION, tokenizer::pinned_controls};

/// Retains model identity and the pinned compiler, never resident weights.
/// Construction does not grant memory admission or artifact qualification.
pub struct Int4SentimentPlanner {
    compiler: SentimentPlanner,
    identity: ExecutionIdentity,
    model: Int4ModelIdentity,
    ceiling: TaskBudget,
    limits: SentimentLimits,
}
impl Int4SentimentPlanner {
    pub fn for_model(model: &Int4ModelIdentity, options: SentimentOptions,
        ceiling: TaskBudget, limits: SentimentLimits) -> Result<Self, Int4SentimentError> {
        ceiling.validate().map_err(|_| SentimentPlanningError::Budget)?;
        let registry = pinned_controls::pinned().map_err(|_| SentimentPlanningError::ControlRegistry)?;
        let compiler = SentimentPlanner::pinned(registry.template_controls(), options)?;
        let identity = bind_candidate_model(uncompiled_identity(&compiler), model)
            .map_err(|_| Int4SentimentError::Identity)?;
        Ok(Self { compiler, identity, model: model.clone(), ceiling, limits })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    #[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn durable_configuration(&self) -> (TaskBudget, SentimentLimits) {
        (self.ceiling, self.limits)
    }
    pub fn plan(&self, request: &SentimentRequest) -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        self.plan_with_control(request, &mut Continue)
    }
    pub fn plan_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest, control: &mut C)
        -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        let context = PlanContext::new(&self.identity, self.ceiling).map_err(|_| SentimentPlanningError::Budget)?;
        self.compiler.plan_int4_with_control(request, &context, self.limits, &self.model, control)
    }
}
fn uncompiled_identity(compiler: &SentimentPlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    // The sealed binder replaces all model fields before this is exposed. The
    // task compiler subsequently binds every request field before admission.
    ExecutionIdentity { schema_version: 1, source_revision: "unbound".to_owned(),
        logical_model_digest: absent, artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(),
        packing_set_digest: absent, tokenizer_digest: compiler.tokenizer_digest(), template_digest: *compiler.template_digest(),
        task_spec: "sentiment-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
