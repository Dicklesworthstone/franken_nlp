//! A pinned raw-text planner derived from actual source/snapshot-loaded weights.
use super::*;
use crate::{execution_identity::{Sha256Digest, ThinkingMode, ToolMode},
    native_engine::strict_int4::STRICT_INT4_EXECUTION,
    tokenizer::pinned_controls, template::IM_END};

/// Retains only model identity and the one pinned compiler, never model weights.
/// The caller separately admits tokenizer/request preparation storage. Creation
/// does not load a model, create a runtime or grant artifact qualification.
pub struct Int4ClassificationPlanner {
    compiler: ClassificationPlanner, identity: ExecutionIdentity, model: Int4ModelIdentity,
    ceiling: TaskBudget, limits: ClassificationLimits,
}
impl Int4ClassificationPlanner {
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, limits: ClassificationLimits)
        -> Result<Self, Int4ClassificationError> {
        ceiling.validate().map_err(|_| ClassificationPlanningError::InvalidLimits)?;
        let registry = pinned_controls::pinned().map_err(|_| ClassificationPlanningError::ControlRegistry)?;
        let mut matches = registry.template_controls().entries().iter().filter(|e| e.special && e.surface == IM_END);
        let eos = matches.next().ok_or(ClassificationPlanningError::ControlRegistry)?.id;
        if matches.next().is_some() { return Err(ClassificationPlanningError::ControlRegistry.into()); }
        let compiler = ClassificationPlanner::pinned(registry.template_controls(), eos)?;
        let identity = bind_candidate_model(uncompiled_identity(&compiler), model)
            .map_err(|_| Int4ClassificationError::Identity)?;
        Ok(Self { compiler, identity, model: model.clone(), ceiling, limits })
    }
    /// Model/asset-bound base, not a compiled request identity. Each prepared
    /// task adds its complete prompt, label, score-space and policy bindings.
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    pub fn plan(&self, request: &ClassificationRequest) -> Result<PreparedInt4Classification, Int4ClassificationError> {
        self.plan_with_control(request, &mut Continue)
    }
    pub fn plan_with_control<C: DecodeStepControl>(&self, request: &ClassificationRequest, control: &mut C)
        -> Result<PreparedInt4Classification, Int4ClassificationError> {
        let context = PlanContext::new(&self.identity, self.ceiling).map_err(|_| ClassificationPlanningError::InvalidLimits)?;
        self.compiler.plan_int4_with_control(request, &context, self.limits, &self.model, control)
    }
}
fn uncompiled_identity(compiler: &ClassificationPlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    // Never exposed with unbound model fields. The sealed model binder replaces
    // source/content/packing fields; the compiler replaces request fields.
    ExecutionIdentity { schema_version: 1, source_revision: "unbound".to_owned(),
        logical_model_digest: absent, artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(),
        packing_set_digest: absent, tokenizer_digest: compiler.tokenizer_digest(), template_digest: *compiler.template_digest(),
        task_spec: "classify-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
