//! Explicit experimental semantic second reader (plan 7.1/7.6).
//! The original extraction is retained byte-for-byte. Every configured scalar
//! claim runs the exact public faithfulness judge, never a hidden validation
//! prompt. Same-model correlation remains visible, and no signal authorizes
//! acceptance, repairs a field, or manufactures a correctness certificate.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::lmhead::scoring::ScoringWork,
    tasks::{ir::{PlanContext, PromptSegmentKind, TaskBudget, TaskPlan}, judge::{
        FaithfulnessPolicy, FaithfulnessRelation, FaithfulnessResult, FAITHFULNESS_VERSION,
        JudgeError, JudgeLimits, JudgeLogits, JudgePlanner, JudgeRequest, JudgeResult, PreparedJudge}},
    validation::{JsonLimits, parse_json_with_limits},
};
use super::{ExtractError, ExtractPlan, ExtractResult, SourceDocument};
mod claims;
pub use claims::{ClaimPathStep, ClaimRule, ClaimValueKind, CLAIM_RENDER_VERSION};
use claims::RenderedField;

pub const SEMANTIC_VERIFICATION_VERSION: &str = "extract-explicit-correlated-faithfulness-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticLimits {
    pub max_fields: usize,
    pub max_claim_bytes: usize,
    pub max_total_claim_bytes: usize,
    pub max_template_bytes: usize,
    pub max_walk_nodes: usize,
    pub max_input_json_bytes: usize,
    pub max_total_prompt_tokens: usize,
    pub max_total_projected_logits: u64,
    pub max_output_bytes: u64,
}
impl Default for SemanticLimits {
    fn default() -> Self {
        Self { max_fields: 256, max_claim_bytes: 4096, max_total_claim_bytes: 65536,
            max_template_bytes: 65536, max_walk_nodes: 4096, max_input_json_bytes: 1024 * 1024,
            max_total_prompt_tokens: 1_000_000, max_total_projected_logits: 100_000_000,
            max_output_bytes: 8 * 1024 * 1024 }
    }
}
/// There is no default-on setting. A caller must explicitly opt into this
/// unqualified diagnostic and supply schema-aware proposition templates.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticVerificationSpec {
    pub schema_version: u32,
    pub experimental_opt_in: bool,
    pub revision: String,
    pub claims: Vec<ClaimRule>,
    pub judge_policy: FaithfulnessPolicy,
    pub judge_budget: TaskBudget,
}
impl SemanticVerificationSpec {
    pub fn from_json(source: &str, max_bytes: usize) -> Result<Self, SemanticError> {
        if source.len() > max_bytes { return Err(SemanticError::Limit("semantic_spec_bytes")); }
        let value = canonjson::parse_str(source).map_err(|_| SemanticError::Contract("invalid semantic specification JSON"))?;
        serde_json::from_value(value).map_err(|_| SemanticError::Contract("invalid semantic specification shape"))
    }
}
#[derive(Debug)]
pub enum SemanticError {
    Contract(&'static str), Limit(&'static str), Extraction(ExtractError), Judge(JudgeError),
    AllocationRefused, Serialization,
}
impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Contract(r) => write!(f, "semantic verification refused: {r}"),
            Self::Limit(axis) => write!(f, "semantic verification budget exceeded: {axis}"),
            Self::Extraction(_) => f.write_str("semantic verification extraction binding refused"),
            Self::Judge(_) => f.write_str("semantic verification judge failed"),
            Self::AllocationRefused => f.write_str("semantic verification allocation refused"),
            Self::Serialization => f.write_str("semantic verification serialization failed") }
    }
}
impl Error for SemanticError {}
impl From<ExtractError> for SemanticError { fn from(e: ExtractError) -> Self { Self::Extraction(e) } }
impl From<JudgeError> for SemanticError { fn from(e: JudgeError) -> Self { Self::Judge(e) } }

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticFieldStatus { Entailed, Contradicted, Unsupported, NotChecked }
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticNotChecked { VerbatimMembershipOnly, NoClaimRule, NullValue, EmptyContainer, JudgeAbstained }
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFieldResult {
    /// Uses the validator's root '$' and escaped JSON pointer child convention.
    pub json_pointer: String,
    pub status: SemanticFieldStatus,
    pub not_checked: Option<SemanticNotChecked>,
    pub claim_rule_id: Option<String>,
    pub rendered_claim: Option<String>,
    pub same_model_correlated: Option<bool>,
    pub judge: Option<FaithfulnessResult>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticVerificationReceipt {
    pub schema_version: u32,
    pub algorithm: String,
    pub claim_rendering: String,
    pub judge_algorithm: String,
    pub specification_revision: String,
    pub calibration: String,
    pub same_model_correlated: bool,
    pub attempted_fields: usize,
    pub checked_fields: usize,
    pub not_checked_fields: usize,
    pub fields: Vec<SemanticFieldResult>,
    pub work: ScoringWork,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticExtractionResult {
    pub schema_version: u32,
    pub extraction: ExtractResult,
    pub semantic: SemanticVerificationReceipt,
}

/// All claims and identities are prepared before any second-reader callback.
/// Neither the content-derived binding nor private identities enter results.
pub struct PreparedSemanticVerification {
    extraction: ExtractResult,
    fields: Vec<RenderedField>,
    judges: Vec<PreparedJudge>,
    revision: String,
    binding: Sha256Digest,
    max_output_bytes: u64,
    work: ScoringWork,
}
impl ExtractPlan {
    /// Attach second-reader planning to THIS admitted extraction, its exact
    /// TaskPlan and its single original source document. Multiple document
    /// segments are refused rather than guessed or concatenated after the fact.
    pub fn prepare_semantic_verification(&self, task: &TaskPlan, source: &SourceDocument,
        extraction: &ExtractResult, extraction_identity: &ExecutionIdentity,
        spec: &SemanticVerificationSpec, planner: &JudgePlanner, judge_context: &PlanContext<'_>,
        judge_limits: JudgeLimits, limits: SemanticLimits) -> Result<PreparedSemanticVerification, SemanticError> {
        if !spec.experimental_opt_in || spec.schema_version != 1 || limits.max_fields == 0 || limits.max_fields > 4096 {
            return Err(SemanticError::Contract("explicit experimental opt-in and supported semantic specification required"));
        }
        claims::check_id(&spec.revision)?; spec.judge_policy.validate()?;
        self.verify_identity(extraction_identity)?;
        if self.task_identity != "extract-v1" || task.task_spec_identity() != "extract-v1"
            || task.ir().digest().map_err(|_| SemanticError::Serialization)? != self.taskir_digest {
            return Err(SemanticError::Contract("semantic source task differs from extraction"));
        }
        let mut docs = task.ir().prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document);
        if docs.next().is_none_or(|s| s.token_ids() != source.token_ids()) || docs.next().is_some()
            || source.text().is_empty() || source.text().len() != source.token_ids().len() {
            return Err(SemanticError::Contract("semantic source does not bind the complete extraction document"));
        }
        ensure_same_engine(extraction_identity, judge_context.execution_identity())?;
        if extraction.output.json.len() > limits.max_input_json_bytes || extraction.output.token_ids.len() > self.options.max_new_tokens {
            return Err(SemanticError::Limit("extraction_input"));
        }
        // Re-run the existing independent finalizer, including exact original
        // source membership. Reject forged/replaced metadata, not just bad JSON.
        let checked = self.finalize(extraction.output.clone())?;
        if &checked != extraction { return Err(SemanticError::Contract("extraction envelope differs from independent finalization")); }
        let value = parse_json_with_limits(&extraction.output.json, JsonLimits {
            max_input_bytes: limits.max_input_json_bytes, max_string_lexeme_bytes: limits.max_input_json_bytes,
            max_container_entries: limits.max_walk_nodes, ..JsonLimits::default()
        }).map_err(|_| SemanticError::Contract("semantic extraction JSON refused"))?;
        let fields = claims::render_fields(self.program.declarative_schema(), &value, &spec.claims, limits)?;
        let count = fields.iter().filter(|field| field.claim.is_some()).count();
        let mut judges = Vec::new(); judges.try_reserve_exact(count).map_err(|_| SemanticError::AllocationRefused)?;
        let mut remaining_logits = limits.max_total_projected_logits;
        let mut work = ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 };
        // Equal deterministic prompt slices bound aggregate allocation before
        // compiling each claim. This is conservative, never a renewable budget.
        let prompt_share = limits.max_total_prompt_tokens / count.max(1);
        for field in &fields {
            let Some(claim) = &field.claim else { continue; };
            let per_claim = JudgeLimits { max_total_prompt_tokens: judge_limits.max_total_prompt_tokens.min(prompt_share),
                max_total_projected_logits: judge_limits.max_total_projected_logits.min(remaining_logits), ..judge_limits };
            let request = JudgeRequest::Faithfulness { source: source.text().to_owned(), claim: claim.clone(),
                policy: spec.judge_policy, budget: spec.judge_budget };
            let plan = planner.plan(&request, judge_context, per_claim)?;
            let cost = plan.planned_work();
            remaining_logits = remaining_logits.checked_sub(cost.projected_logits).ok_or(SemanticError::Limit("projected_logits"))?;
            work = add_work(work, cost)?; judges.push(plan);
        }
        let binding = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(SEMANTIC_VERIFICATION_VERSION,
            CLAIM_RENDER_VERSION, extraction_identity, extraction, spec, limits,
            judges.iter().map(PreparedJudge::execution_identity).collect::<Vec<_>>())).map_err(|_| SemanticError::Serialization)?);
        Ok(PreparedSemanticVerification { extraction: checked, fields, judges, revision: spec.revision.clone(), binding,
            max_output_bytes: limits.max_output_bytes.min(self.max_result_bytes), work })
    }
}
impl PreparedSemanticVerification {
    /// Prompt-derived private commitment; do not export it in public telemetry.
    pub fn binding_digest(&self) -> &Sha256Digest { &self.binding }
    pub fn judge_identities(&self) -> impl Iterator<Item = &ExecutionIdentity> { self.judges.iter().map(PreparedJudge::execution_identity) }
    pub fn planned_work(&self) -> ScoringWork { self.work }
    fn verify_identities(&self, admitted: &[ExecutionIdentity]) -> Result<(), SemanticError> {
        if admitted.len() != self.judges.len() { return Err(SemanticError::Contract("semantic admitted identity count")); }
        for (judge, identity) in self.judges.iter().zip(admitted) { judge.verify_identity(identity)?; }
        Ok(())
    }
    /// The provider receives each exact claim/head prompt through JudgeLogits.
    /// No field failure is converted into unsupported, skipped, or retried.
    pub fn execute<M: JudgeLogits>(&self, admitted: &[ExecutionIdentity], model: &mut M)
        -> Result<SemanticExtractionResult, SemanticError> {
        self.verify_identities(admitted)?;
        let mut results = Vec::new(); results.try_reserve_exact(self.judges.len()).map_err(|_| SemanticError::AllocationRefused)?;
        for (judge, identity) in self.judges.iter().zip(admitted) {
            let JudgeResult::Faithfulness(result) = judge.execute(identity, model)? else {
                return Err(SemanticError::Contract("semantic execution returned a different judge task"));
            };
            results.push(result);
        }
        self.finish(results)
    }
    fn finish(&self, results: Vec<FaithfulnessResult>) -> Result<SemanticExtractionResult, SemanticError> {
        if results.len() != self.judges.len() { return Err(SemanticError::Contract("incomplete semantic result set")); }
        let mut results = results.into_iter(); let mut fields = Vec::new();
        fields.try_reserve_exact(self.fields.len()).map_err(|_| SemanticError::AllocationRefused)?;
        let mut checked = 0;
        for field in &self.fields {
            let judge = if field.claim.is_some() { Some(results.next().ok_or(SemanticError::Contract("missing semantic field result"))?) } else { None };
            let status = match judge.as_ref().and_then(|result| result.relation) {
                Some(FaithfulnessRelation::Entailed) => SemanticFieldStatus::Entailed,
                Some(FaithfulnessRelation::Contradicted) => SemanticFieldStatus::Contradicted,
                Some(FaithfulnessRelation::Unsupported) => SemanticFieldStatus::Unsupported,
                None => SemanticFieldStatus::NotChecked,
            };
            if status != SemanticFieldStatus::NotChecked { checked += 1; }
            let not_checked = if judge.is_some() && status == SemanticFieldStatus::NotChecked { Some(SemanticNotChecked::JudgeAbstained) } else { field.not_checked };
            fields.push(SemanticFieldResult { json_pointer: field.pointer.clone(), status, not_checked,
                claim_rule_id: field.rule_id.clone(), rendered_claim: field.claim.clone(),
                same_model_correlated: judge.as_ref().map(|_| true), judge });
        }
        let result = SemanticExtractionResult { schema_version: 1, extraction: self.extraction.clone(),
            semantic: SemanticVerificationReceipt { schema_version: 1, algorithm: SEMANTIC_VERIFICATION_VERSION.to_owned(),
                claim_rendering: CLAIM_RENDER_VERSION.to_owned(), judge_algorithm: FAITHFULNESS_VERSION.to_owned(),
                specification_revision: self.revision.clone(), calibration: "experimental_uncalibrated_diagnostic_only".to_owned(),
                same_model_correlated: true, attempted_fields: self.judges.len(), checked_fields: checked,
                not_checked_fields: fields.len() - checked, fields, work: self.work } };
        self.check_output(&result)?;
        Ok(result)
    }
    fn check_output<T: Serialize>(&self, result: &T) -> Result<(), SemanticError> {
        let bytes = canonjson::canonical_bytes(result).map_err(|_| SemanticError::Serialization)?;
        if bytes.len() as u64 > self.max_output_bytes { return Err(SemanticError::Limit("complete_semantic_output")); }
        Ok(())
    }
}
fn ensure_same_engine(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), SemanticError> {
    if a.logical_model_digest != b.logical_model_digest || a.artifact_format != b.artifact_format
        || a.quant_recipe != b.quant_recipe || a.packing_set_digest != b.packing_set_digest
        || a.tokenizer_digest != b.tokenizer_digest || a.numerics_profile != b.numerics_profile || a.kv_dtype != b.kv_dtype
        || a.backend_semantic_version != b.backend_semantic_version || a.source_revision != b.source_revision
        || a.thinking_mode != b.thinking_mode || a.tool_mode != b.tool_mode || a.host_class != b.host_class || a.compiler_identity != b.compiler_identity {
        return Err(SemanticError::Contract("semantic v1 requires the same admitted model and execution profile"));
    }
    Ok(())
}
fn add_work(a: ScoringWork, b: ScoringWork) -> Result<ScoringWork, SemanticError> {
    Ok(ScoringWork { prefix_evaluations: a.prefix_evaluations.checked_add(b.prefix_evaluations).ok_or(SemanticError::Limit("prefixes"))?,
        scored_edges: a.scored_edges.checked_add(b.scored_edges).ok_or(SemanticError::Limit("edges"))?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits).ok_or(SemanticError::Limit("projected_logits"))? })
}
