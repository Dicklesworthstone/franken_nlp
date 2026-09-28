//! Opt-in neural answer synthesis from verified verbatim map evidence ONLY.
//! Generated passage answers are retained for inspection, never fed back as facts.
//! This is evidence compression with unestablished recall, not full-context parity.
use super::*;
use crate::native_engine::portable_int8::ProjectionWork;
mod evidence;

pub const INT8_QUESTION_SYNTHESIS_EXECUTION: &str = "portable-int8-question-verbatim-evidence-synthesis-v1";
const SEMANTICS: &str = "answer-from-collected-quotes-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuestionSynthesisLimits {
    /// ALL distinct (chunk, quote) passages must fit. Never top-k or truncate.
    pub max_evidence_passages: usize,
    /// Logical verbatim passage bytes; question/manifest/prompt limits also apply.
    pub max_evidence_bytes: usize,
}
/// Private text, deliberately not Debug/Serialize. Existing SourceQuestion and
/// independent-passage APIs are unchanged unless this concrete mode is selected.
pub struct SourceQuestionSynthesis {
    pub question: SourceQuestion,
    pub limits: QuestionSynthesisLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionSynthesisPreflight {
    discovery: Int8QuestionPreflight,
    synthesis_reserve: Int8Work,
    synthesis_masks: u64,
    limits: QuestionSynthesisLimits,
}
impl Int8QuestionSynthesisPreflight {
    pub fn discovery(&self) -> Int8QuestionPreflight { self.discovery }
    pub fn synthesis_reserved_work(&self) -> Int8Work { self.synthesis_reserve }
    pub fn reserved_model_work(&self) -> Result<Int8Work, Int8SourceMapError> {
        add_work(self.discovery.planned_work(), self.synthesis_reserve).ok_or(Int8SourceMapError::WorkLimit)
    }
    pub fn reserved_mask_visits(&self) -> Result<u64, Int8SourceMapError> {
        self.discovery.reserved_mask_visits().checked_add(self.synthesis_masks).ok_or(Int8SourceMapError::WorkLimit)
    }
    pub fn verify_completed(&self, run: &Int8QuestionSynthesisRun) -> Result<(), Int8SourceMapError> {
        self.discovery.verify_completed(&run.discovery)?;
        let actual = add_work(run.discovery.model_work, run.synthesis_model_work).ok_or(Int8SourceMapError::WorkLimit)?;
        let masks = run.discovery.mask_node_visit_charge.checked_add(run.synthesis_mask_node_visits)
            .ok_or(Int8SourceMapError::WorkLimit)?;
        let v = self.discovery.verification;
        if run.schema_version != 1 || run.execution != INT8_QUESTION_SYNTHESIS_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.semantics != SEMANTICS
            || run.calibration != AnswerCalibration::Uncalibrated
            || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || run.semantic_support != SummarySemanticSupport::NotAssessed
            || run.untrusted_fields != ["synthesis", "discovery.mapped.root.value"]
            || run.synthesis_reserved_work != self.synthesis_reserve
            || run.reserved_model_work != self.reserved_model_work()? || run.model_work != actual
            || !within(run.synthesis_planned_work, self.synthesis_reserve)
            || !within(run.synthesis_model_work, run.synthesis_planned_work)
            || run.reserved_mask_node_visits != self.reserved_mask_visits()? || run.mask_node_visit_charge != masks
            || run.synthesis_mask_node_visits > self.synthesis_masks
            || run.evidence_passages > self.limits.max_evidence_passages || run.evidence_bytes > self.limits.max_evidence_bytes
            || run.verification_used.fields > v.max_fields || run.verification_used.matches > v.max_matches
            || run.verification_used.scan_steps > v.max_scan_steps
            || run.verification_used.scan_steps < run.discovery.verification_scan_steps {
            return Err(invalid());
        }
        match run.synthesis.status {
            SynthesisStatus::NoEvidenceCollected => {
                if run.evidence_passages != 0 || run.evidence_bytes != 0 || run.discovery.answered_chunks != 0
                    || run.synthesis_planned_work != Int8Work::default() || run.synthesis_model_work != Int8Work::default()
                    || run.synthesis_mask_node_visits != 0 || run.synthesis.answer.is_some() || !run.synthesis.citations.is_empty() {
                    return Err(invalid());
                }
            }
            SynthesisStatus::Answered | SynthesisStatus::Abstained => {
                if run.evidence_passages == 0 || run.evidence_bytes == 0 || run.synthesis_model_work.forward_positions == 0 {
                    return Err(invalid());
                }
                if run.synthesis.status == SynthesisStatus::Answered {
                    if !run.synthesis.answer.as_deref().is_some_and(has_text) || run.synthesis.citations.is_empty() { return Err(invalid()); }
                } else if run.synthesis.answer.is_some() || !run.synthesis.citations.is_empty() { return Err(invalid()); }
            }
        }
        Ok(())
    }
}

/// Consumes the map commitment and keeps its exact question/source binding.
/// No public API accepts a caller-constructed collection of inference receipts.
pub struct PreparedInt8QuestionSynthesis<'s> {
    source: &'s str,
    question: String,
    map: PreparedInt8Question<'s>,
    identity: ExecutionIdentity,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int8SourceMapLimits,
    options: AnswerOptions,
    expected: Int8QuestionSynthesisPreflight,
}
impl SourceTaskPlanner {
    /// Reserve a maximum-context synthesis pass BEFORE discovering any evidence.
    /// This is a work upper bound, not a fabricated task/prompt/native receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int8_question_synthesis_with_control<C: DecodeStepControl>(&self, source: &str,
        request: &SourceQuestionSynthesis, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<Int8QuestionSynthesisPreflight, Int8SourceMapError> {
        checkpoint(control)?;
        let (discovery_limits, reserve) = reserve(request.limits, budget, planning, mapping)?;
        let discovery = self.preflight_int8_question_with_control(source, &request.question, budget, context,
            planning, discovery_limits, control)?;
        expectation(discovery, reserve, request.limits, mapping)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int8_question_synthesis_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &SourceQuestionSynthesis, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<PreparedInt8QuestionSynthesis<'s>, Int8SourceMapError> {
        checkpoint(control)?;
        let (discovery_limits, reserve) = reserve(request.limits, budget, planning, mapping)?;
        let map = self.plan_int8_question_with_control(source, &request.question, budget, context,
            planning, discovery_limits, control)?;
        let expected = expectation(map.preflight_metadata(), reserve, request.limits, mapping)?;
        Ok(PreparedInt8QuestionSynthesis { source, question: copy(&request.question.question)?, map,
            identity: context.execution_identity().clone(), budget, planning, mapping, options: request.question.options, expected })
    }
}
fn expectation(discovery: Int8QuestionPreflight, synthesis_reserve: Int8Work, limits: QuestionSynthesisLimits,
    mapping: Int8SourceMapLimits) -> Result<Int8QuestionSynthesisPreflight, Int8SourceMapError> {
    let value = Int8QuestionSynthesisPreflight { discovery, synthesis_reserve, synthesis_masks: mapping.mask_visits_per_chunk, limits };
    if !within(value.reserved_model_work()?, mapping.max_model_work) || value.reserved_mask_visits()? > mapping.max_mask_visits {
        return Err(Int8SourceMapError::WorkLimit);
    }
    Ok(value)
}
fn reserve(limits: QuestionSynthesisLimits, budget: TaskBudget, planning: SourcePlanningLimits,
    mut mapping: Int8SourceMapLimits) -> Result<(Int8SourceMapLimits, Int8Work), Int8SourceMapError> {
    mapping.validate(budget, planning)?;
    if !(1..=1024).contains(&limits.max_evidence_passages) || limits.max_evidence_passages > planning.max_passages
        || limits.max_evidence_bytes == 0 || limits.max_evidence_bytes > planning.max_input_bytes {
        return Err(Int8SourceMapError::InvalidLimits);
    }
    let output = budget.max_output_tokens as usize;
    let prompt = planning.max_context_tokens.checked_sub(output).ok_or(Int8SourceMapError::InvalidLimits)?
        .min(budget.max_input_tokens as usize);
    if prompt == 0 { return Err(Int8SourceMapError::InvalidLimits); }
    let reserved = constrained_int8::planned_work(prompt, output).map_err(|_| Int8SourceMapError::WorkLimit)?;
    mapping.max_model_work = subtract(mapping.max_model_work, reserved)?;
    mapping.max_mask_visits = mapping.max_mask_visits.checked_sub(mapping.mask_visits_per_chunk)
        .ok_or(Int8SourceMapError::WorkLimit)?;
    Ok((mapping, reserved))
}
impl PreparedInt8QuestionSynthesis<'_> {
    pub fn preflight_metadata(&self) -> Int8QuestionSynthesisPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.map.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
    /// One real engine and controller for discovery AND the final evidence-only
    /// question. Insufficient evidence/context refuses; nothing is silently ranked
    /// away. All generated map answers remain in the completed output.
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8QuestionSynthesisRun, Int8SourceMapError> {
        checkpoint(control)?;
        if planner.tokenizer_digest() != self.identity.tokenizer_digest || *planner.template_digest() != self.identity.template_digest {
            return Err(Int8SourceMapError::Admission);
        }
        self.map.preflight(admitted, engine)?;
        let Self { source, question, map, identity, budget, planning, mapping, options, expected } = self;
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        expected.discovery.verify_completed(&discovery)?;
        let mut remaining = remaining_verification(&discovery, expected.discovery.verification)?;
        let collection = evidence::collect(source, discovery.mapped.root().value(), expected.limits, &mut remaining, control)?;
        let passages = collection.passages.len(); let bytes = collection.bytes;
        let (answer, planned, work, masks) = if passages == 0 {
            (SynthesisAnswer { status: SynthesisStatus::NoEvidenceCollected, answer: None, citations: Vec::new() },
                Int8Work::default(), Int8Work::default(), 0)
        } else {
            let context = PlanContext::new(&identity, budget).map_err(|_| Int8SourceMapError::Admission)?;
            let request = SourceTaskRequest::Answer { question, passages: collection.copy_passages()?, options, budget };
            let plan = planner.plan_int8_with_control(&request, &context, planning, control)?;
            let planned = plan.planned_work();
            if !within(planned, expected.synthesis_reserve) { return Err(Int8SourceMapError::WorkLimit); }
            // Admit the actual dynamic evidence identity and full resident KV
            // before the final forward; discovery success cannot waive this.
            preflight_plans(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), engine)?;
            let mut driver = NativeDriver { engine, vocabulary, control, limits: mapping };
            let native = driver.run(&plan, plan.execution_identity())?;
            check_run(&plan, &native, mapping.mask_visits_per_chunk)?;
            extract_int8::check_size(&native, plan.max_result_bytes()).map_err(Int8SourceError::from)?;
            let work = native.model_work;
            let SourceTaskResult::Answer(raw) = native.result else { return Err(invalid()); };
            let masks = raw.mask_node_visit_charge;
            let answer = evidence::final_answer(source, raw, &collection, options, &mut remaining, control)?;
            (answer, planned, work, masks)
        };
        drop(collection);
        finish(discovery, answer, passages, bytes, planned, work, masks, remaining, expected,
            mapping.reduction.max_result_bytes as u64, control)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisStatus { NoEvidenceCollected, Answered, Abstained }
#[derive(Serialize)]
pub struct SynthesisAnswer {
    pub status: SynthesisStatus,
    pub answer: Option<String>,
    /// Original-document spans reachable through the collected evidence only.
    /// Equal text elsewhere is NOT newly asserted as support.
    pub citations: Vec<SourceCitation>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct QuestionVerificationWork { pub fields: usize, pub matches: usize, pub scan_steps: u64 }
#[derive(Serialize)]
pub struct Int8QuestionSynthesisRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub semantics: &'static str,
    pub evidence_passages: usize,
    pub evidence_bytes: usize,
    pub synthesis_reserved_work: Int8Work,
    pub synthesis_planned_work: Int8Work,
    pub synthesis_model_work: Int8Work,
    pub reserved_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub synthesis_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: QuestionVerificationWork,
    pub calibration: AnswerCalibration,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 2],
    pub synthesis: SynthesisAnswer,
    /// All alternatives/abstentions remain visible, even after synthesis.
    pub discovery: Int8QuestionRun,
}
fn remaining_verification(discovery: &Int8QuestionRun, total: GroundingBudget) -> Result<GroundingBudget, Int8SourceMapError> {
    let stats = execution::statistics(discovery.mapped.root().value())?;
    Ok(GroundingBudget { max_fields: total.max_fields.checked_sub(stats.citations).ok_or(Int8SourceMapError::WorkLimit)?,
        max_matches: total.max_matches.checked_sub(stats.spans).ok_or(Int8SourceMapError::WorkLimit)?,
        max_scan_steps: total.max_scan_steps.checked_sub(discovery.verification_scan_steps).ok_or(Int8SourceMapError::WorkLimit)? })
}
#[allow(clippy::too_many_arguments)]
fn finish<C: DecodeStepControl>(discovery: Int8QuestionRun, synthesis: SynthesisAnswer, evidence_passages: usize,
    evidence_bytes: usize, synthesis_planned_work: Int8Work, synthesis_model_work: Int8Work, synthesis_mask_node_visits: u64,
    remaining: GroundingBudget, expected: Int8QuestionSynthesisPreflight, cap: u64, control: &mut C)
    -> Result<Int8QuestionSynthesisRun, Int8SourceMapError> {
    checkpoint(control)?;
    let total = expected.discovery.verification;
    let verification_used = QuestionVerificationWork {
        fields: total.max_fields.checked_sub(remaining.max_fields).ok_or_else(invalid)?,
        matches: total.max_matches.checked_sub(remaining.max_matches).ok_or_else(invalid)?,
        scan_steps: total.max_scan_steps.checked_sub(remaining.max_scan_steps).ok_or_else(invalid)?,
    };
    let model_work = add_work(discovery.model_work, synthesis_model_work).ok_or(Int8SourceMapError::WorkLimit)?;
    let mask_node_visit_charge = discovery.mask_node_visit_charge.checked_add(synthesis_mask_node_visits)
        .ok_or(Int8SourceMapError::WorkLimit)?;
    let run = Int8QuestionSynthesisRun { schema_version: 1, execution: INT8_QUESTION_SYNTHESIS_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, semantics: SEMANTICS, evidence_passages, evidence_bytes,
        synthesis_reserved_work: expected.synthesis_reserve, synthesis_planned_work, synthesis_model_work,
        reserved_model_work: expected.reserved_model_work()?, model_work,
        reserved_mask_node_visits: expected.reserved_mask_visits()?, synthesis_mask_node_visits, mask_node_visit_charge,
        verification_used, calibration: AnswerCalibration::Uncalibrated,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        untrusted_fields: ["synthesis", "discovery.mapped.root.value"], synthesis, discovery };
    expected.verify_completed(&run)?;
    extract_int8::check_size(&run, cap).map_err(Int8SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, Int8SourceMapError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int8SourceMapError::WorkLimit);
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn invalid() -> Int8SourceMapError { Int8SourceError::InvalidResult.into() }
fn copy(text: &str) -> Result<String, Int8SourceMapError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| Int8SourceMapError::Allocation)?;
    out.push_str(text); Ok(out)
}
fn vector<T>(count: usize) -> Result<Vec<T>, Int8SourceMapError> {
    let mut out = Vec::new(); out.try_reserve_exact(count).map_err(|_| Int8SourceMapError::Allocation)?; Ok(out)
}

#[cfg(test)] mod tests;
