//! Model-bound pinned judge planning without retaining resident weights.
use super::*;
use crate::tokenizer::pinned_controls;

pub struct Int4JudgePlanner {
    compiler: JudgePlanner,
    identity: ExecutionIdentity,
    model: Int4ModelIdentity,
    ceiling: TaskBudget,
    limits: JudgeLimits,
}
impl Int4JudgePlanner {
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, limits: JudgeLimits)
        -> Result<Self, Int4JudgeError> {
        ceiling.validate().map_err(|_| JudgeError::Contract("judge task budget"))?;
        let registry = pinned_controls::pinned().map_err(|_| Int4JudgeError::Identity)?;
        let mut ends = registry.template_controls().entries().iter().filter(|e| e.special && e.surface == IM_END);
        let eos = ends.next().ok_or(Int4JudgeError::Identity)?.id;
        if ends.next().is_some() { return Err(Int4JudgeError::Identity); }
        let compiler = JudgePlanner::pinned(registry.template_controls(), eos)?;
        let identity = bind_candidate_model(uncompiled_identity(&compiler), model).map_err(|_| Int4JudgeError::Identity)?;
        Ok(Self { compiler, identity, model: model.clone(), ceiling, limits })
    }
    /// Model/asset-bound base; each prepared plan binds its whole request.
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    pub(crate) fn semantic_configuration(&self) -> (TaskBudget, JudgeLimits) {
        (self.ceiling, self.limits)
    }
    #[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn durable_configuration(&self) -> (TaskBudget, JudgeLimits) {
        (self.ceiling, self.limits)
    }
    pub fn plan(&self, request: &JudgeRequest) -> Result<PreparedInt4Judge, Int4JudgeError> {
        self.plan_with_control(request, &mut Continue)
    }
    pub fn plan_with_control<C: DecodeStepControl>(&self, request: &JudgeRequest, control: &mut C)
        -> Result<PreparedInt4Judge, Int4JudgeError> {
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| JudgeError::Contract("judge plan context"))?;
        self.compiler.plan_int4_with_control(request, &context, self.limits, &self.model, control)
    }
    /// A composed second reader can only tighten prompt/logit work. Model,
    /// task ceiling and every other complete-head limit remain factory-owned.
    pub(crate) fn plan_semantic_with_control<C: DecodeStepControl>(&self, request: &JudgeRequest,
        prompt_tokens: usize, projected_logits: u64, control: &mut C)
        -> Result<PreparedInt4Judge, Int4JudgeError> {
        if !matches!(request, JudgeRequest::Faithfulness { .. }) {
            return Err(JudgeError::Contract("semantic reader requires faithfulness").into());
        }
        let limits = JudgeLimits {
            max_total_prompt_tokens: self.limits.max_total_prompt_tokens.min(prompt_tokens),
            max_total_projected_logits: self.limits.max_total_projected_logits.min(projected_logits),
            ..self.limits
        };
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| JudgeError::Contract("judge plan context"))?;
        self.compiler.plan_int4_with_control(request, &context, limits, &self.model, control)
    }
}
fn uncompiled_identity(compiler: &JudgePlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    // The sealed binder replaces model fields before exposing this identity;
    // the shared judge compiler supplies all request and policy bindings.
    ExecutionIdentity { schema_version: 1, source_revision: "unbound".to_owned(),
        logical_model_digest: absent, artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(),
        packing_set_digest: absent, tokenizer_digest: compiler.tokenizer_digest(), template_digest: *compiler.template_digest(),
        task_spec: "judge-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
