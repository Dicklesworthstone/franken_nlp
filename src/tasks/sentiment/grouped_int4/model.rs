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
    /// Configure untrusted input before deriving a prepared identity. A
    /// complete-document map shares these ceilings across probes and plans.
    pub fn with_document_encoding(mut self, encoding: crate::tokenizer::untrusted::UntrustedEncoding)
        -> Result<Self, Int4SentimentError> {
        self.compiler = self.compiler.with_document_encoding(encoding)?;
        self.identity.template_digest = *self.compiler.template_digest();
        Ok(self)
    }
    pub fn document_encoding(&self) -> &crate::tokenizer::untrusted::UntrustedEncoding {
        self.compiler.document_encoding()
    }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    /// Immutable request authority retained by this model-bound compiler.
    pub fn task_budget(&self) -> TaskBudget { self.ceiling }
    /// Immutable finite-candidate planning bounds, without exposing prompts.
    pub fn planning_limits(&self) -> SentimentLimits { self.limits }
    /// Exact maximum source bytes across the selected axes, request/planner
    /// prompt ceilings and native continuation context. No reserve heuristic.
    pub fn max_document_bytes(&self, axes: &[crate::tasks::sentiment::SentimentAxis],
        budget: TaskBudget, context_tokens: usize) -> Result<usize, Int4SentimentError> {
        self.compiler.source_capacity(axes, budget, self.ceiling, self.limits, context_tokens)
            .map_err(Int4SentimentError::from)
    }
    pub(crate) fn source_token_capacity(&self, axes: &[crate::tasks::sentiment::SentimentAxis],
        budget: TaskBudget, context_tokens: usize) -> Result<usize, SentimentPlanningError> {
        self.compiler.source_token_capacity(axes, budget, self.ceiling, self.limits, context_tokens)
    }
    pub(super) fn source_tokens_with_work<C: DecodeStepControl>(&self, source: &str,
        work: &mut SafeBpeWork, control: &mut C) -> Result<usize, SentimentPlanningError> {
        self.compiler.source_tokens_with_work(source, work, control)
    }
    pub(super) fn plan_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        control: &mut C, work: &mut SafeBpeWork) -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        let context = PlanContext::new(&self.identity, self.ceiling).map_err(|_| SentimentPlanningError::Budget)?;
        check_model(context.execution_identity(), &self.model)?;
        self.compiler.compile_int4_with_work(request, &context, self.limits, self.model.group_width(), control, work)
    }
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
