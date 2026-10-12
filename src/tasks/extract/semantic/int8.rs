//! Native INT8 extraction followed by the existing full-source faithfulness judge.
//! One consumed plan and engine, exact scalar claims, no rewritten extraction,
//! no model-receipt injection and no automatic acceptance or field repair.
use super::*;
use std::sync::Arc;
use crate::{
    batch::extract::quantized::PreparedInt8BatchExtraction,
    native_engine::{constrained::JsonWorkBudget, constrained_int8::Int8JsonBudget,
        decode::{DecodeCancellationKind, DecodeStepControl}, kv::KV_BYTES_PER_TOKEN,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE,
            scoring::Int8ScoringBudget}},
    tasks::{extract::{ExtractionVocabulary, quantized::{Int8ExtractError, Int8ExtractRun, check_size}},
        judge::{partition_evidence, quantized::{Int8JudgeError, Int8JudgeRun, PreparedInt8Judge, INT8_JUDGE_EXECUTION}}},
    grammar::mask::MaskWorkLimits,
};

pub const INT8_SEMANTIC_EXECUTION: &str = "portable-int8-extraction-correlated-second-reader-v1";

/// Explicit, closed configuration; the complete native ceiling covers BOTH
/// stages. The original extraction reservation is never refunded after EOS.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8SemanticOptions {
    pub specification: SemanticVerificationSpec,
    pub limits: SemanticLimits,
    pub max_model_work: Int8Work,
}
impl Int8SemanticOptions {
    pub fn validate(&self) -> Result<(), Int8SemanticError> {
        let s = &self.specification; let l = self.limits; let w = self.max_model_work;
        if s.schema_version != 1 || !s.experimental_opt_in { return Err(Int8SemanticError::Contract); }
        claims::check_id(&s.revision)?;
        s.judge_policy.validate().map_err(SemanticError::from)?;
        s.judge_budget.validate().map_err(|_| Int8SemanticError::Contract)?;
        if !(1..=4096).contains(&l.max_fields) || !(1..=1_048_576).contains(&l.max_claim_bytes)
            || !(1..=64 * 1024 * 1024).contains(&l.max_total_claim_bytes)
            || !(1..=1024 * 1024).contains(&l.max_template_bytes)
            || !(1..=1_000_000).contains(&l.max_walk_nodes)
            || !(1..=64 * 1024 * 1024).contains(&l.max_input_json_bytes)
            || !(1..=16_777_216).contains(&l.max_total_prompt_tokens)
            || l.max_total_projected_logits == 0 || !(1..=64 * 1024 * 1024).contains(&l.max_output_bytes)
            || w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
            || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
            return Err(Int8SemanticError::Contract);
        }
        check_size(s, 1024 * 1024)?;
        Ok(())
    }
}

pub enum Int8SemanticError {
    Contract, WorkBudget, Accounting, Cancelled(DecodeCancellationKind),
    Extraction(Int8ExtractError), Semantic(SemanticError), Judge(Int8JudgeError),
}
impl fmt::Display for Int8SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Contract => "int8 semantic extraction contract refused",
            Self::WorkBudget => "int8 semantic extraction whole-request work exceeded",
            Self::Accounting => "int8 semantic extraction completion mismatch",
            Self::Cancelled(_) => "int8 semantic extraction cancelled",
            Self::Extraction(_) => "int8 semantic extraction native extraction failed",
            Self::Semantic(_) => "int8 semantic extraction claim or result refused",
            Self::Judge(_) => "int8 semantic extraction complete judgment failed",
        })
    }
}
impl fmt::Debug for Int8SemanticError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) } }
impl Error for Int8SemanticError {}
impl From<Int8ExtractError> for Int8SemanticError { fn from(e: Int8ExtractError) -> Self { Self::Extraction(e) } }
impl From<ExtractError> for Int8SemanticError { fn from(e: ExtractError) -> Self { Self::Extraction(e.into()) } }
impl From<SemanticError> for Int8SemanticError { fn from(e: SemanticError) -> Self { Self::Semantic(e) } }
impl From<Int8JudgeError> for Int8SemanticError { fn from(e: Int8JudgeError) -> Self { Self::Judge(e) } }
impl Int8SemanticError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) => Some(*c), Self::Extraction(e) => e.cancellation(),
            Self::Judge(e) => e.cancellation(), _ => None }
    }
}

/// Construct from the real pinned extraction factory, not a saved JSON result.
/// The exact original source and schema stay attached throughout both stages.
pub struct PreparedInt8SemanticExtraction {
    extraction: PreparedInt8BatchExtraction,
    judge: Arc<JudgePlanner>,
    judge_identity: ExecutionIdentity,
    judge_limits: JudgeLimits,
    options: Int8SemanticOptions,
}
impl PreparedInt8SemanticExtraction {
    pub fn new<C: DecodeStepControl>(extraction: PreparedInt8BatchExtraction, judge: Arc<JudgePlanner>,
        judge_identity: ExecutionIdentity, judge_limits: JudgeLimits,
        options: Int8SemanticOptions, control: &mut C) -> Result<Self, Int8SemanticError> {
        poll(control)?;
        options.validate()?;
        judge_identity.validate().map_err(|_| Int8SemanticError::Contract)?;
        let a = extraction.execution_identity();
        crate::native_engine::constrained_int8::check_profile(a).map_err(Int8ExtractError::from)?;
        crate::native_engine::constrained_int8::check_profile(&judge_identity).map_err(Int8ExtractError::from)?;
        if judge_identity.task_spec != "judge-v1" || judge_identity.tokenizer_digest != judge.tokenizer_digest()
            || judge_identity.template_digest != *judge.template_digest() { return Err(Int8SemanticError::Contract); }
        same_model(a, &judge_identity)?;
        let schema = extraction.extraction_plan().semantic_schema(extraction.task_plan(), extraction.source())?;
        claims::validate_rules(schema, &options.specification.claims, options.limits)?;
        // Refuse source/evidence-policy mismatch BEFORE extraction. Claims and
        // their dynamic prompts do not exist yet; do not invent placeholder ones.
        partition_evidence(extraction.source().text(), options.specification.judge_policy)
            .map_err(SemanticError::from)?;
        if !fits(extraction.planned_work(), options.max_model_work) { return Err(Int8SemanticError::WorkBudget); }
        check_size(&judge_identity, 16 * 1024)?;
        poll(control)?;
        Ok(Self { extraction, judge, judge_identity, judge_limits, options })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.extraction.execution_identity() }
    pub fn judge_identity(&self) -> &ExecutionIdentity { &self.judge_identity }
    pub fn extraction_reserved_work(&self) -> Int8Work { self.extraction.planned_work() }
    pub fn max_model_work(&self) -> Int8Work { self.options.max_model_work }
    pub fn extraction_budget(&self) -> TaskBudget { *self.extraction.task_plan().ir().budget() }
    pub fn judge_budget(&self) -> TaskBudget { self.options.specification.judge_budget }
    pub fn semantic_limits(&self) -> SemanticLimits { self.options.limits }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }

    /// The caller owns native/preparation/output admission. All actual judge
    /// prompts are compiled and preflighted BEFORE the first second-reader
    /// forward. Any field failure returns no combined result, never Unsupported.
    pub fn execute_with_control<C: DecodeStepControl>(self, engine: &mut StrictInt8Engine<'_>,
        vocabulary: &ExtractionVocabulary, masks: MaskWorkLimits, mask_visits: u64, control: &mut C)
        -> Result<Int8SemanticExtractionRun, Int8SemanticError> {
        poll(control)?;
        if masks.max_trie_node_visits == 0 || masks.checkpoint_interval_nodes == 0 || mask_visits == 0 {
            return Err(Int8SemanticError::Contract);
        }
        let kv = self.check_capacity(engine)?;
        let reserved = self.extraction.planned_work();
        let extraction = self.extraction.extraction_plan().execute(engine, self.execution_identity(), vocabulary,
            Int8JsonBudget { native: Int8RunBudget::exact(reserved), json: JsonWorkBudget {
                max_forward_positions: reserved.forward_positions, max_projected_logits: reserved.projected_logits,
                max_kv_bytes: kv, max_total_mask_node_visits: mask_visits, mask_limits: masks } }, control)?;
        self.check_capacity(engine)?;
        let review = self.prepare_review(extraction, control)?;
        for plan in &review.judges {
            poll(control)?;
            plan.preflight(plan.execution_identity(), engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: kv })?;
        }
        let mut results = vector(review.judges.len())?;
        for plan in &review.judges {
            poll(control)?;
            results.push(plan.execute_with_control(plan.execution_identity(), engine,
                Int8ScoringBudget { native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: kv }, control)?);
            self.check_capacity(engine)?;
        }
        self.finish(review, results, control)
    }
    fn check_capacity(&self, engine: &StrictInt8Engine<'_>) -> Result<u64, Int8SemanticError> {
        let positions = engine.kv_cache().capacity_positions();
        let cap = self.extraction_budget().max_kv_bytes.min(self.judge_budget().max_kv_bytes);
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || self.extraction_reserved_work().forward_positions > positions as u64 {
            return Err(Int8SemanticError::Contract);
        }
        (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).filter(|&n| n != 0 && n <= cap)
            .ok_or(Int8SemanticError::Contract)
    }
    // Private composition seam. Public execution above always obtains this
    // value from the real native extraction, never from caller-supplied output.
    fn prepare_review<C: DecodeStepControl>(&self, extraction: Int8ExtractRun, control: &mut C)
        -> Result<Review, Int8SemanticError> {
        poll(control)?;
        self.extraction.extraction_plan().verify_completed(&extraction)?;
        let l = self.options.limits;
        let value = parse_json_with_limits(&extraction.result.output.json, JsonLimits {
            max_input_bytes: l.max_input_json_bytes, max_string_lexeme_bytes: l.max_input_json_bytes,
            max_container_entries: l.max_walk_nodes, ..JsonLimits::default()
        }).map_err(|_| Int8SemanticError::Contract)?;
        let schema = self.extraction.extraction_plan().semantic_schema(self.extraction.task_plan(), self.extraction.source())?;
        let fields = claims::render_fields(schema, &value, &self.options.specification.claims, l)?;
        poll(control)?;
        let count = fields.iter().filter(|f| f.claim.is_some()).count();
        let mut judges = vector(count)?;
        let mut work = Int8Work::default();
        let s = &self.options.specification;
        let context = PlanContext::new(&self.judge_identity, s.judge_budget).map_err(|_| Int8SemanticError::Contract)?;
        let mut logits_left = l.max_total_projected_logits;
        // Conservative per-field shares bound all retained plans. No prompt
        // tokens or projected-logit allowance is renewed for each field.
        let prompt_share = l.max_total_prompt_tokens / count.max(1);
        for field in &fields {
            let Some(claim) = &field.claim else { continue; };
            poll(control)?;
            let limits = JudgeLimits {
                max_total_prompt_tokens: self.judge_limits.max_total_prompt_tokens.min(prompt_share),
                max_total_projected_logits: self.judge_limits.max_total_projected_logits.min(logits_left),
                ..self.judge_limits };
            let request = JudgeRequest::Faithfulness { source: copy(self.extraction.source().text())?, claim: copy(claim)?,
                policy: s.judge_policy, budget: s.judge_budget };
            let plan = self.judge.plan_int8_with_control(&request, &context, limits, control)?;
            logits_left = logits_left.checked_sub(plan.planned_work().projected_logits).ok_or(Int8SemanticError::WorkBudget)?;
            work = plus(work, plan.planned_work())?;
            // Charge the original maximum extraction reservation, NOT its
            // smaller early-EOS actual work. No partial set of judges can run.
            if !fits(plus(self.extraction_reserved_work(), work)?, self.options.max_model_work) {
                return Err(Int8SemanticError::WorkBudget);
            }
            judges.push(plan);
        }
        poll(control)?;
        Ok(Review { extraction, fields, judges, work })
    }
    fn finish<C: DecodeStepControl>(&self, review: Review, results: Vec<Int8JudgeRun>, control: &mut C)
        -> Result<Int8SemanticExtractionRun, Int8SemanticError> {
        poll(control)?;
        if results.len() != review.judges.len() { return Err(Int8SemanticError::Accounting); }
        let mut work = Int8Work::default(); let mut scoring = ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 };
        let (mut head_count, mut rewound) = (0_usize, 0_u64);
        let mut verified = vector(results.len())?;
        for (plan, run) in review.judges.iter().zip(results) {
            poll(control)?;
            if run.schema_version != 1 || run.execution != INT8_JUDGE_EXECUTION || run.numerics_profile != STRICT_INT8_PROFILE
                || run.model_work != plan.planned_work() || run.head_count != plan.head_count() {
                return Err(Int8SemanticError::Accounting);
            }
            let JudgeResult::Faithfulness(result) = run.result else { return Err(Int8SemanticError::Accounting); };
            work = plus(work, run.model_work)?;
            scoring = super::add_work(scoring, result.work)?;
            head_count = head_count.checked_add(run.head_count).ok_or(Int8SemanticError::Accounting)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int8SemanticError::Accounting)?;
            verified.push(result);
        }
        if work != review.work { return Err(Int8SemanticError::Accounting); }
        let attempted = verified.len(); let mut verified = verified.into_iter(); let mut fields = vector(review.fields.len())?;
        let mut checked = 0;
        for field in review.fields {
            poll(control)?;
            let judge = if field.claim.is_some() { Some(verified.next().ok_or(Int8SemanticError::Accounting)?) } else { None };
            let status = match judge.as_ref().and_then(|j| j.relation) {
                Some(FaithfulnessRelation::Entailed) => SemanticFieldStatus::Entailed,
                Some(FaithfulnessRelation::Contradicted) => SemanticFieldStatus::Contradicted,
                Some(FaithfulnessRelation::Unsupported) => SemanticFieldStatus::Unsupported,
                None => SemanticFieldStatus::NotChecked,
            };
            if status != SemanticFieldStatus::NotChecked { checked += 1; }
            let not_checked = if judge.is_some() && status == SemanticFieldStatus::NotChecked {
                Some(SemanticNotChecked::JudgeAbstained) } else { field.not_checked };
            fields.push(SemanticFieldResult { json_pointer: field.pointer, status, not_checked,
                claim_rule_id: field.rule_id, rendered_claim: field.claim,
                same_model_correlated: judge.as_ref().map(|_| true), judge });
        }
        if verified.next().is_some() { return Err(Int8SemanticError::Accounting); }
        let reserved = plus(self.extraction_reserved_work(), work)?;
        let actual = plus(review.extraction.model_work, work)?;
        if !fits(actual, reserved) || !fits(reserved, self.options.max_model_work) { return Err(Int8SemanticError::Accounting); }
        let run = Int8SemanticExtractionRun { schema_version: 1, execution: INT8_SEMANTIC_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, extraction_reserved_work: self.extraction_reserved_work(),
            semantic_model_work: work, reserved_model_work: reserved, model_work: actual,
            judge_head_count: head_count, rewound_positions: rewound, extraction: review.extraction,
            semantic: SemanticVerificationReceipt { schema_version: 1, algorithm: SEMANTIC_VERIFICATION_VERSION.to_owned(),
                claim_rendering: CLAIM_RENDER_VERSION.to_owned(), judge_algorithm: FAITHFULNESS_VERSION.to_owned(),
                specification_revision: self.options.specification.revision.clone(),
                calibration: "experimental_uncalibrated_diagnostic_only".to_owned(), same_model_correlated: true,
                attempted_fields: attempted, checked_fields: checked, not_checked_fields: fields.len() - checked, fields, work: scoring } };
        check_size(&run, self.max_result_bytes())?;
        poll(control)?;
        Ok(run)
    }
}
struct Review { extraction: Int8ExtractRun, fields: Vec<RenderedField>, judges: Vec<PreparedInt8Judge>, work: Int8Work }

#[derive(Serialize)]
pub struct Int8SemanticExtractionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub extraction_reserved_work: Int8Work,
    pub semantic_model_work: Int8Work,
    pub reserved_model_work: Int8Work,
    pub model_work: Int8Work,
    pub judge_head_count: usize,
    pub rewound_positions: u64,
    /// Original JSON and native evidence are never rewritten or replaced.
    pub extraction: Int8ExtractRun,
    pub semantic: SemanticVerificationReceipt,
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), Int8SemanticError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int8SemanticError::Cancelled(c)), None => Ok(()) }
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int8SemanticError> {
    // Normalize ONLY task-owned identities; all model, host and future fields
    // remain in structural equality. Do not merely compare a short model ID.
    let mut normalized = a.clone();
    normalized.task_spec = b.task_spec.clone(); normalized.template_digest = b.template_digest;
    normalized.taskir_digest = b.taskir_digest; normalized.prompt_digest = b.prompt_digest;
    normalized.schema_digest = b.schema_digest; normalized.grammar_compiler_version = b.grammar_compiler_version.clone();
    normalized.sampler_version = b.sampler_version.clone(); normalized.calibration_digest = b.calibration_digest;
    normalized.decision_policy_digest = b.decision_policy_digest;
    if &normalized != b { return Err(Int8SemanticError::Contract); }
    Ok(())
}
fn fits(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn plus(a: Int8Work, b: Int8Work) -> Result<Int8Work, Int8SemanticError> {
    a.checked_add(b).map_err(|_| Int8SemanticError::WorkBudget)
}
fn vector<T>(n: usize) -> Result<Vec<T>, Int8SemanticError> {
    let mut out = Vec::new(); out.try_reserve_exact(n).map_err(|_| SemanticError::AllocationRefused)?; Ok(out)
}
fn copy(text: &str) -> Result<String, Int8SemanticError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| SemanticError::AllocationRefused)?;
    out.push_str(text); Ok(out)
}

#[cfg(test)] mod tests;
