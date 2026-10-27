//! Real grouped-INT4 extraction followed by its full-source faithfulness judge.
//! Original schema output stays unchanged; second-reader signals are explicitly
//! correlated, experimental diagnostics with no repair or acceptance authority.
use super::*;
use crate::{
    batch::extract::grouped_int4::PreparedInt4BatchExtraction,
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget,
        constrained_int4::{self, Int4JsonBudget},
        decode::{DecodeCancellationKind, DecodeStepControl}, kv::KV_BYTES_PER_TOKEN,
        strict_int4::{Int4RunBudget, Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE,
            scoring::Int4ScoringBudget}},
    tasks::{extract::{ExtractionVocabulary, grouped_int4::{Int4ExtractError, Int4ExtractRun},
        quantized::check_size},
        judge::{partition_evidence, grouped_int4::{Int4JudgeError, Int4JudgePlanner,
            Int4JudgeRun, PreparedInt4Judge, INT4_JUDGE_EXECUTION}}},
};

pub const INT4_SEMANTIC_EXECUTION: &str = "grouped-int4-extraction-correlated-second-reader-v1";

/// One explicit whole-request ceiling covers extraction and every field's
/// complete judgment. Early EOS never refunds the extraction reservation.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4SemanticOptions {
    pub specification: SemanticVerificationSpec,
    pub limits: SemanticLimits,
    pub max_model_work: Int4Work,
}
impl Int4SemanticOptions {
    pub fn validate(&self) -> Result<(), Int4SemanticError> {
        let s = &self.specification; let l = self.limits; let w = self.max_model_work;
        if s.schema_version != 1 || !s.experimental_opt_in { return Err(Int4SemanticError::Contract); }
        claims::check_id(&s.revision)?;
        s.judge_policy.validate().map_err(SemanticError::from)?;
        s.judge_budget.validate().map_err(|_| Int4SemanticError::Contract)?;
        if !(1..=4096).contains(&l.max_fields) || !(1..=1_048_576).contains(&l.max_claim_bytes)
            || !(1..=64 * 1024 * 1024).contains(&l.max_total_claim_bytes)
            || !(1..=1024 * 1024).contains(&l.max_template_bytes)
            || !(1..=1_000_000).contains(&l.max_walk_nodes)
            || !(1..=64 * 1024 * 1024).contains(&l.max_input_json_bytes)
            || !(1..=16_777_216).contains(&l.max_total_prompt_tokens)
            || l.max_total_projected_logits == 0 || !(1..=64 * 1024 * 1024).contains(&l.max_output_bytes)
            || w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
            || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
            return Err(Int4SemanticError::Contract);
        }
        check_size(s, 1024 * 1024)?;
        Ok(())
    }
    /// Pure bounded preflight for callers that compile schemas before loading
    /// weights. This grants no execution authority: the consumed preparation
    /// repeats these checks against its own sealed schema and original source.
    pub fn validate_source_schema(&self, schema: &crate::grammar::SchemaNode, source: &str)
        -> Result<(), Int4SemanticError> {
        self.validate()?;
        claims::validate_rules(schema, &self.specification.claims, self.limits)?;
        partition_evidence(source, self.specification.judge_policy).map_err(SemanticError::from)?;
        Ok(())
    }
}

pub enum Int4SemanticError {
    Contract, WorkBudget, Accounting, Cancelled(DecodeCancellationKind),
    Extraction(Int4ExtractError), Semantic(SemanticError), Judge(Int4JudgeError),
}
impl fmt::Display for Int4SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Contract => "int4 semantic extraction contract refused",
            Self::WorkBudget => "int4 semantic extraction whole-request work exceeded",
            Self::Accounting => "int4 semantic extraction completion mismatch",
            Self::Cancelled(_) => "int4 semantic extraction cancelled",
            Self::Extraction(_) => "int4 semantic extraction native extraction failed",
            Self::Semantic(_) => "int4 semantic extraction claim or result refused",
            Self::Judge(_) => "int4 semantic extraction complete judgment failed",
        })
    }
}
impl fmt::Debug for Int4SemanticError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) } }
impl Error for Int4SemanticError {}
impl From<Int4ExtractError> for Int4SemanticError { fn from(e: Int4ExtractError) -> Self { Self::Extraction(e) } }
impl From<ExtractError> for Int4SemanticError { fn from(e: ExtractError) -> Self { Self::Extraction(e.into()) } }
impl From<SemanticError> for Int4SemanticError { fn from(e: SemanticError) -> Self { Self::Semantic(e) } }
impl From<Int4JudgeError> for Int4SemanticError { fn from(e: Int4JudgeError) -> Self { Self::Judge(e) } }
impl Int4SemanticError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) => Some(*c), Self::Extraction(e) => e.cancellation(),
            Self::Judge(e) => e.cancellation(), _ => None }
    }
}

/// Consumes the actual prepared extraction and model-bound pinned judge. No
/// deserialized output, replacement schema, identity or logits provider can
/// supply the public execution path. The original source stays attached.
pub struct PreparedInt4SemanticExtraction {
    extraction: PreparedInt4BatchExtraction,
    judge: Int4JudgePlanner,
    options: Int4SemanticOptions,
}
impl PreparedInt4SemanticExtraction {
    pub fn new<C: DecodeStepControl>(extraction: PreparedInt4BatchExtraction, judge: Int4JudgePlanner,
        options: Int4SemanticOptions, control: &mut C) -> Result<Self, Int4SemanticError> {
        poll(control)?;
        let (ceiling, limits) = judge.semantic_configuration();
        check_configuration(&extraction, judge.execution_identity(), judge.group_width(), ceiling, limits, &options)?;
        poll(control)?;
        Ok(Self { extraction, judge, options })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.extraction.execution_identity() }
    pub fn judge_identity(&self) -> &ExecutionIdentity { self.judge.execution_identity() }
    pub fn group_width(&self) -> usize { self.judge.group_width() }
    pub fn extraction_reserved_work(&self) -> Int4Work { self.extraction.planned_work() }
    pub fn max_model_work(&self) -> Int4Work { self.options.max_model_work }
    pub fn extraction_budget(&self) -> TaskBudget { *self.extraction.task_plan().ir().budget() }
    pub fn judge_budget(&self) -> TaskBudget { self.options.specification.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge.semantic_configuration().1 }
    pub fn source_bytes(&self) -> usize { self.extraction.source().text().len() }
    pub fn semantic_limits(&self) -> SemanticLimits { self.options.limits }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }

    /// The host admits all retained preparation, intermediate results, native
    /// workspace and final output. Every real field plan is compiled and
    /// preflighted before ANY second-reader forward. A failure returns no
    /// combined result; it never becomes an Unsupported verdict or a repair.
    pub fn execute_with_control<C: DecodeStepControl>(self, engine: &mut StrictInt4Engine<'_>,
        vocabulary: &ExtractionVocabulary, masks: MaskWorkLimits, mask_visits: u64, control: &mut C)
        -> Result<Int4SemanticExtractionRun, Int4SemanticError> {
        poll(control)?;
        if masks.max_trie_node_visits == 0 || masks.checkpoint_interval_nodes == 0 || mask_visits == 0 {
            return Err(Int4SemanticError::Contract);
        }
        let kv = self.check_capacity(engine)?;
        let reserved = self.extraction_reserved_work();
        let extraction = self.extraction.extraction_plan().execute(engine, self.execution_identity(), vocabulary,
            Int4JsonBudget { native: Int4RunBudget::exact(reserved), json: JsonWorkBudget {
                max_forward_positions: reserved.forward_positions, max_projected_logits: reserved.projected_logits,
                max_kv_bytes: kv, max_total_mask_node_visits: mask_visits, mask_limits: masks } }, control)?;
        self.check_capacity(engine)?;
        let review = self.prepare_review(extraction, control)?;
        for plan in &review.judges {
            poll(control)?;
            plan.preflight(plan.execution_identity(), engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: kv })?;
        }
        let mut results = vector(review.judges.len())?;
        for plan in &review.judges {
            poll(control)?;
            results.push(plan.execute_with_control(plan.execution_identity(), engine,
                Int4ScoringBudget { native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: kv }, control)?);
            self.check_capacity(engine)?;
        }
        self.finish(review, results, control)
    }
    fn check_capacity(&self, engine: &StrictInt4Engine<'_>) -> Result<u64, Int4SemanticError> {
        // Check both real model bindings even when all fields are not_checked
        // and therefore no judge forward will run to repeat this validation.
        for identity in [self.execution_identity(), self.judge_identity()] {
            constrained_int4::check_model(identity, engine.model_identity())
                .map_err(|_| Int4SemanticError::Contract)?;
        }
        let positions = engine.kv_cache().capacity_positions();
        let cap = self.extraction_budget().max_kv_bytes.min(self.judge_budget().max_kv_bytes);
        if engine.profile() != STRICT_INT4_PROFILE || engine.model_identity().group_width() != self.group_width()
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || self.extraction_reserved_work().forward_positions > positions as u64 {
            return Err(Int4SemanticError::Contract);
        }
        (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).filter(|&n| n != 0 && n <= cap)
            .ok_or(Int4SemanticError::Contract)
    }
    // Private composition seam; the public path only supplies native output.
    fn prepare_review<C: DecodeStepControl>(&self, extraction: Int4ExtractRun, control: &mut C)
        -> Result<Review, Int4SemanticError> {
        poll(control)?;
        let fields = prepare_fields(&self.extraction, &extraction, &self.options)?;
        poll(control)?;
        let count = fields.iter().filter(|f| f.claim.is_some()).count();
        let mut judges = vector(count)?;
        let mut work = Int4Work::default();
        let s = &self.options.specification;
        let mut logits_left = self.options.limits.max_total_projected_logits;
        // Fixed per-field shares bound the sum of retained exact prompts.
        // Neither the original planner ceiling nor this total renews per field.
        let prompt_share = self.options.limits.max_total_prompt_tokens / count.max(1);
        for field in &fields {
            let Some(claim) = &field.claim else { continue; };
            poll(control)?;
            let request = JudgeRequest::Faithfulness { source: copy(self.extraction.source().text())?, claim: copy(claim)?,
                policy: s.judge_policy, budget: s.judge_budget };
            let plan = self.judge.plan_semantic_with_control(&request, prompt_share, logits_left, control)?;
            if plan.group_width() != self.group_width() { return Err(Int4SemanticError::Accounting); }
            same_model(self.execution_identity(), plan.execution_identity())?;
            logits_left = logits_left.checked_sub(plan.planned_work().projected_logits).ok_or(Int4SemanticError::WorkBudget)?;
            work = plus(work, plan.planned_work())?;
            reserve_combined(self.extraction_reserved_work(), work, self.options.max_model_work)?;
            judges.push(plan);
        }
        poll(control)?;
        Ok(Review { extraction, fields, judges, work })
    }
    fn finish<C: DecodeStepControl>(&self, review: Review, results: Vec<Int4JudgeRun>, control: &mut C)
        -> Result<Int4SemanticExtractionRun, Int4SemanticError> {
        poll(control)?;
        if results.len() != review.judges.len() { return Err(Int4SemanticError::Accounting); }
        let mut work = Int4Work::default(); let mut scoring = ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 };
        let (mut head_count, mut rewound) = (0_usize, 0_u64);
        let mut verified = vector(results.len())?;
        for (plan, run) in review.judges.iter().zip(results) {
            poll(control)?;
            if run.schema_version != 1 || run.execution != INT4_JUDGE_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
                || run.group_width != self.group_width() || run.model_work != plan.planned_work()
                || run.head_count != plan.head_count() {
                return Err(Int4SemanticError::Accounting);
            }
            let JudgeResult::Faithfulness(result) = run.result else { return Err(Int4SemanticError::Accounting); };
            if result.work.projected_logits != run.model_work.projected_logits {
                return Err(Int4SemanticError::Accounting);
            }
            work = plus(work, run.model_work)?;
            scoring = super::add_work(scoring, result.work)?;
            head_count = head_count.checked_add(run.head_count).ok_or(Int4SemanticError::Accounting)?;
            rewound = rewound.checked_add(run.rewound_positions).ok_or(Int4SemanticError::Accounting)?;
            verified.push(result);
        }
        if work != review.work { return Err(Int4SemanticError::Accounting); }
        let attempted = verified.len(); let mut verified = verified.into_iter(); let mut fields = vector(review.fields.len())?;
        let mut checked = 0;
        for field in review.fields {
            poll(control)?;
            let judge = if field.claim.is_some() { Some(verified.next().ok_or(Int4SemanticError::Accounting)?) } else { None };
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
        if verified.next().is_some() { return Err(Int4SemanticError::Accounting); }
        let reserved = reserve_combined(self.extraction_reserved_work(), work, self.options.max_model_work)?;
        let actual = plus(review.extraction.model_work, work)?;
        if !fits(actual, reserved) { return Err(Int4SemanticError::Accounting); }
        let run = Int4SemanticExtractionRun { schema_version: 1, execution: INT4_SEMANTIC_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width: self.group_width(),
            extraction_reserved_work: self.extraction_reserved_work(), semantic_model_work: work,
            reserved_model_work: reserved, model_work: actual, judge_head_count: head_count,
            rewound_positions: rewound, extraction: review.extraction,
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

fn check_configuration(extraction: &PreparedInt4BatchExtraction, judge_identity: &ExecutionIdentity,
    width: usize, ceiling: TaskBudget, limits: JudgeLimits, options: &Int4SemanticOptions)
    -> Result<(), Int4SemanticError> {
    options.validate()?;
    let a = extraction.execution_identity();
    for identity in [a, judge_identity] {
        constrained_int4::check_profile(identity, width).map_err(|_| Int4SemanticError::Contract)?;
    }
    if extraction.extraction_plan().group_width() != width || judge_identity.task_spec != "judge-v1" {
        return Err(Int4SemanticError::Contract);
    }
    same_model(a, judge_identity)?;
    let b = options.specification.judge_budget;
    if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes > ceiling.max_kv_bytes || limits.max_total_prompt_tokens == 0
        || limits.max_total_candidate_tokens == 0 || limits.max_total_projected_logits == 0
        || limits.max_output_bytes == 0 || limits.per_head.max_candidates == 0
        || limits.per_head.max_total_tokens == 0 || limits.per_head.max_nodes == 0
        || limits.per_head.max_depth == 0 || limits.per_head.max_candidate_id_bytes == 0
        || limits.per_head.max_projected_logits == 0 {
        return Err(Int4SemanticError::Contract);
    }
    let schema = extraction.extraction_plan().semantic_schema(extraction.task_plan(), extraction.source())?;
    // The policy must partition the complete source before extraction spends
    // work; dynamic field claims are never guessed or supplied as placeholders.
    options.validate_source_schema(schema, extraction.source().text())?;
    reserve_combined(extraction.planned_work(), Int4Work::default(), options.max_model_work)?;
    check_size(judge_identity, 16 * 1024)?;
    Ok(())
}
fn prepare_fields(prepared: &PreparedInt4BatchExtraction, extraction: &Int4ExtractRun,
    options: &Int4SemanticOptions) -> Result<Vec<RenderedField>, Int4SemanticError> {
    prepared.extraction_plan().verify_completed(extraction)?;
    let l = options.limits;
    let value = parse_json_with_limits(&extraction.result.output.json, JsonLimits {
        max_input_bytes: l.max_input_json_bytes, max_string_lexeme_bytes: l.max_input_json_bytes,
        max_container_entries: l.max_walk_nodes, ..JsonLimits::default()
    }).map_err(|_| Int4SemanticError::Contract)?;
    let schema = prepared.extraction_plan().semantic_schema(prepared.task_plan(), prepared.source())?;
    claims::render_fields(schema, &value, &options.specification.claims, l).map_err(Into::into)
}
struct Review { extraction: Int4ExtractRun, fields: Vec<RenderedField>, judges: Vec<PreparedInt4Judge>, work: Int4Work }

#[derive(Serialize)]
pub struct Int4SemanticExtractionRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub extraction_reserved_work: Int4Work,
    pub semantic_model_work: Int4Work,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub judge_head_count: usize,
    pub rewound_positions: u64,
    /// Original JSON, native tokens and source evidence are never rewritten.
    pub extraction: Int4ExtractRun,
    pub semantic: SemanticVerificationReceipt,
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), Int4SemanticError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int4SemanticError::Cancelled(c)), None => Ok(()) }
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int4SemanticError> {
    // Normalize task-owned fields only. All model/partition/tokenizer/host and
    // future fields remain part of equality, beyond a short model name.
    let mut normalized = a.clone();
    normalized.task_spec = b.task_spec.clone(); normalized.template_digest = b.template_digest;
    normalized.taskir_digest = b.taskir_digest; normalized.prompt_digest = b.prompt_digest;
    normalized.schema_digest = b.schema_digest; normalized.grammar_compiler_version = b.grammar_compiler_version.clone();
    normalized.sampler_version = b.sampler_version.clone(); normalized.calibration_digest = b.calibration_digest;
    normalized.decision_policy_digest = b.decision_policy_digest;
    if &normalized != b { return Err(Int4SemanticError::Contract); }
    Ok(())
}
fn fits(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn plus(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4SemanticError> {
    a.checked_add(b).map_err(|_| Int4SemanticError::WorkBudget)
}
fn reserve_combined(extraction: Int4Work, semantic: Int4Work, ceiling: Int4Work) -> Result<Int4Work, Int4SemanticError> {
    let reserved = plus(extraction, semantic)?;
    if !fits(reserved, ceiling) { return Err(Int4SemanticError::WorkBudget); }
    Ok(reserved)
}
fn vector<T>(n: usize) -> Result<Vec<T>, Int4SemanticError> {
    let mut out = Vec::new(); out.try_reserve_exact(n).map_err(|_| SemanticError::AllocationRefused)?; Ok(out)
}
fn copy(text: &str) -> Result<String, Int4SemanticError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| SemanticError::AllocationRefused)?;
    out.push_str(text); Ok(out)
}

#[cfg(test)] mod tests;
