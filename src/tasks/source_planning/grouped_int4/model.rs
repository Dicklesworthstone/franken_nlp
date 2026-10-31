//! Pinned source planning derived from actual loaded INT4 model identities.
//!
//! Paired callers share immutable tokenizer/vocabulary assets while every
//! compiled request remains bound to its own explicitly selected model.

use super::*;
use std::sync::Arc;
use crate::{
    native_engine::{generation::grouped_int4::bind_candidate_model,
        strict_int4::{Int4ModelIdentity, STRICT_INT4_EXECUTION}},
    tokenizer::pinned_controls,
};

struct Assets { compiler: SourceTaskPlanner, vocabulary: ExtractionVocabulary }

/// Model and asset binding only; this object owns no weights, runtime or
/// qualification authority. Its caller must admit preparation storage before
/// construction. Clones made with `with_model` share the one pinned asset set.
pub struct Int4SourcePlanner {
    assets: Arc<Assets>,
    identity: ExecutionIdentity,
    model: Int4ModelIdentity,
    task: BuiltInTask,
    ceiling: TaskBudget,
    limits: SourcePlanningLimits,
}
impl Int4SourcePlanner {
    pub fn for_model(model: &Int4ModelIdentity, task: BuiltInTask, ceiling: TaskBudget,
        limits: SourcePlanningLimits) -> Result<Self, Int4SourceError> {
        ceiling.validate().map_err(|_| SourcePlanningError::Contract("source task budget"))?;
        limits.validate()?;
        if !matches!(task, BuiltInTask::Ner | BuiltInTask::Keyphrases | BuiltInTask::Summarize | BuiltInTask::Answer) {
            return Err(SourcePlanningError::Contract("unsupported model-bound source task").into());
        }
        let registry = pinned_controls::pinned()
            .map_err(|_| SourcePlanningError::Contract("pinned source control registry"))?;
        let mut endings = registry.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == IM_END);
        let eos = endings.next().ok_or(SourcePlanningError::Contract("source EOS census"))?.id;
        if endings.next().is_some() { return Err(SourcePlanningError::Contract("source EOS census").into()); }
        let compiler = SourceTaskPlanner::pinned(registry.template_controls(), eos)?;
        let vocabulary = ExtractionVocabulary::pinned(registry.template_controls())?;
        let identity = bind_candidate_model(uncompiled_identity(&compiler, task), model)
            .map_err(|_| SourcePlanningError::Contract("source model identity"))?;
        Ok(Self { assets: Arc::new(Assets { compiler, vocabulary }), identity,
            model: model.clone(), task, ceiling, limits })
    }

    /// Explicitly select another loaded model while reusing only the immutable
    /// pinned compiler and vocabulary. Request and native engine state are
    /// never shared, and model/packing fields are freshly bound.
    pub fn with_model(&self, model: &Int4ModelIdentity) -> Result<Self, Int4SourceError> {
        let identity = bind_candidate_model(uncompiled_identity(&self.assets.compiler, self.task), model)
            .map_err(|_| SourcePlanningError::Contract("source model identity"))?;
        Ok(Self { assets: self.assets.clone(), identity, model: model.clone(), task: self.task,
            ceiling: self.ceiling, limits: self.limits })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    pub fn vocabulary(&self) -> &ExtractionVocabulary { &self.assets.vocabulary }

    pub fn plan_with_control<C: DecodeStepControl>(&self, request: &SourceTaskRequest, control: &mut C)
        -> Result<PreparedInt4SourceTask, Int4SourceError> {
        if request.task() != self.task {
            return Err(SourcePlanningError::Contract("model-bound source task differs").into());
        }
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| SourcePlanningError::Contract("source task context"))?;
        self.assets.compiler.plan_int4_with_control(request, &context, self.limits,
            self.group_width(), control)
    }
}

fn uncompiled_identity(compiler: &SourceTaskPlanner, task: BuiltInTask) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    // These placeholders never escape: the sealed actual-model binder fills
    // model fields, and the existing source compiler fills request fields.
    ExecutionIdentity { schema_version: 1, source_revision: "unbound".to_owned(),
        logical_model_digest: absent, artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(),
        packing_set_digest: absent, tokenizer_digest: compiler.tokenizer_digest(), template_digest: *compiler.template_digest(),
        task_spec: task.spec().identity().to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
