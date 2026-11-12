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
    /// Configure the immutable policy before preparing any request. All data
    /// segments, evidence windows and composed plans share its declared work
    /// limits; callers separately retain a ledger across reconstruction.
    pub fn with_document_encoding(mut self, encoding: UntrustedEncoding) -> Result<Self, Int4JudgeError> {
        self.compiler = self.compiler.with_document_encoding(encoding)?;
        self.identity.template_digest = *self.compiler.template_digest();
        Ok(self)
    }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn document_encoding_workspace_bytes(&self) -> Result<u64, Int4JudgeError> {
        self.compiler.document_encoding_workspace_bytes().map_err(Into::into)
    }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    /// Claim-aware source capacity of one complete faithfulness chunk. Exact
    /// evidence-head and whole-document work admission follow compilation.
    pub fn max_faithfulness_source_bytes(&self, request: &JudgeRequest, context_tokens: usize)
        -> Result<usize, Int4JudgeError> {
        if self.document_encoding() != &UntrustedEncoding::ByteFallback {
            return Err(JudgeError::Contract("compact faithfulness requires token and byte capacity admission").into());
        }
        let JudgeRequest::Faithfulness { claim, policy, budget, .. } = request else {
            return Err(JudgeError::Contract("faithfulness map requires faithfulness mode").into());
        };
        self.compiler.faithfulness.source_capacity(claim, *policy, *budget, self.ceiling,
            self.limits, context_tokens).map_err(Into::into)
    }
    pub(super) fn faithfulness_source_token_capacity_with_work<C: DecodeStepControl>(&self,
        request: &JudgeRequest, context_tokens: usize, work: &mut SafeBpeWork, control: &mut C)
        -> Result<usize, Int4JudgeError> {
        let JudgeRequest::Faithfulness { claim, policy, budget, .. } = request else {
            return Err(JudgeError::Contract("faithfulness map requires faithfulness mode").into());
        };
        self.compiler.faithfulness.source_token_capacity_with_work(claim, *policy, *budget,
            self.ceiling, self.limits, context_tokens, self.document_encoding(), work, control).map_err(Into::into)
    }
    pub(super) fn faithfulness_source_byte_limit(&self, request: &JudgeRequest) -> Result<usize, Int4JudgeError> {
        let JudgeRequest::Faithfulness { policy, .. } = request else {
            return Err(JudgeError::Contract("faithfulness map requires faithfulness mode").into());
        };
        FaithfulnessCompiler::source_byte_limit(*policy).map_err(Into::into)
    }
    pub(super) fn source_tokens_with_work<C: DecodeStepControl>(&self, source: &str,
        work: &mut SafeBpeWork, control: &mut C) -> Result<usize, Int4JudgeError> {
        self.compiler.faithfulness.source_tokens_with_work(source, self.document_encoding(), work, control).map_err(Into::into)
    }
    pub(crate) fn plan_with_encoding_work<C: DecodeStepControl>(&self, request: &JudgeRequest,
        control: &mut C, work: &mut SafeBpeWork) -> Result<PreparedInt4Judge, Int4JudgeError> {
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| JudgeError::Contract("judge plan context"))?;
        check_model(context.execution_identity(), &self.model)?;
        self.compiler.compile_int4_with_work(request, &context, self.limits, self.group_width(), control, work)
    }
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
        self.plan_with_encoding_work(request, control, &mut SafeBpeWork::default())
    }
    /// A composed second reader can only tighten prompt/logit work. Model,
    /// task ceiling and every other complete-head limit remain factory-owned.
    pub(crate) fn plan_semantic_with_control<C: DecodeStepControl>(&self, request: &JudgeRequest,
        prompt_tokens: usize, projected_logits: u64, control: &mut C)
        -> Result<PreparedInt4Judge, Int4JudgeError> {
        if !matches!(request, JudgeRequest::Faithfulness { .. }) {
            return Err(JudgeError::Contract("semantic reader requires faithfulness").into());
        }
        if self.document_encoding() != &UntrustedEncoding::ByteFallback {
            return Err(JudgeError::Contract("composed semantic reader requires byte input admission").into());
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
