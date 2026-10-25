//! One final INT4 answer from verified verbatim discovery quotations only.
//! All original passage answers remain inspectable. Their generated assertions
//! never enter the evidence input, and absent evidence is not calibrated abstention.
use super::*;
use super::super::execution::{NativeDriver, SourceDriver, check_run, preflight};
use crate::native_engine::portable_int8::ProjectionWork;
mod evidence;

pub const INT4_QUESTION_SYNTHESIS_EXECUTION: &str = "grouped-int4-question-verbatim-evidence-synthesis-v1";
const SEMANTICS: &str = "answer-from-collected-quotes-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuestionSynthesisLimits {
    /// Every distinct (chunk, quote) passage must fit; no top-k or truncation.
    pub max_evidence_passages: usize,
    /// Verbatim passage bytes; the question and complete manifest must fit too.
    pub max_evidence_bytes: usize,
}
impl QuestionSynthesisLimits {
    pub fn validate(self, planning: SourcePlanningLimits) -> Result<(), Int4SourceMapError> {
        if !(1..=1024).contains(&self.max_evidence_passages) || self.max_evidence_passages > planning.max_passages
            || self.max_evidence_bytes == 0 || self.max_evidence_bytes > planning.max_input_bytes {
            return Err(Int4SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}
/// Private source/question bytes deliberately have no Debug or wire constructor.
pub struct SourceQuestionSynthesis {
    pub question: SourceQuestion,
    pub limits: QuestionSynthesisLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionSynthesisPreflight {
    discovery: Int4QuestionPreflight,
    synthesis_reserve: Int4Work,
    synthesis_masks: u64,
    limits: QuestionSynthesisLimits,
    options: AnswerOptions,
    max_output_tokens: usize,
}
impl Int4QuestionSynthesisPreflight {
    pub fn discovery(&self) -> Int4QuestionPreflight { self.discovery }
    pub fn synthesis_reserved_work(&self) -> Int4Work { self.synthesis_reserve }
    pub fn reserved_model_work(&self) -> Result<Int4Work, Int4SourceMapError> {
        add_work(self.discovery.planned_work(), self.synthesis_reserve).ok_or(Int4SourceMapError::WorkLimit)
    }
    pub fn reserved_mask_visits(&self) -> Result<u64, Int4SourceMapError> {
        self.discovery.reserved_mask_visits().checked_add(self.synthesis_masks).ok_or(Int4SourceMapError::WorkLimit)
    }
    pub fn verify_completed(&self, run: &Int4QuestionSynthesisRun) -> Result<(), Int4SourceMapError> {
        self.discovery.verify_completed(&run.discovery)?;
        let actual = add_work(run.discovery.model_work, run.synthesis_model_work).ok_or(Int4SourceMapError::WorkLimit)?;
        let masks = run.discovery.mask_node_visit_charge.checked_add(run.synthesis_mask_node_visits)
            .ok_or(Int4SourceMapError::WorkLimit)?;
        let verification = run.discovery.verification_used;
        if run.schema_version != 1 || run.execution != INT4_QUESTION_SYNTHESIS_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.discovery.group_width()
            || run.semantics != SEMANTICS || run.calibration != AnswerCalibration::Uncalibrated
            || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || run.semantic_support != SummarySemanticSupport::NotAssessed
            || run.untrusted_fields != ["synthesis", "discovery.mapped.root.value"]
            || run.synthesis_reserved_work != self.synthesis_reserve
            || run.reserved_model_work != self.reserved_model_work()? || run.model_work != actual
            || !fits(run.synthesis_planned_work, self.synthesis_reserve)
            || !fits(run.synthesis_model_work, run.synthesis_planned_work)
            || run.reserved_mask_node_visits != self.reserved_mask_visits()? || run.mask_node_visit_charge != masks
            || run.synthesis_mask_node_visits > self.synthesis_masks
            || run.evidence_passages > self.limits.max_evidence_passages || run.evidence_bytes > self.limits.max_evidence_bytes
            || !run.verification_used.fits(self.discovery.verification)
            || run.verification_used.fields < verification.fields || run.verification_used.matches < verification.matches
            || run.verification_used.scan_steps < verification.scan_steps {
            return Err(invalid());
        }
        match run.synthesis.status {
            SynthesisStatus::NoEvidenceCollected => {
                if run.evidence_passages != 0 || run.evidence_bytes != 0 || run.discovery.answered_chunks != 0
                    || run.synthesis_prompt_tokens != 0 || run.synthesis_generated_tokens != 0
                    || run.synthesis_planned_work != Int4Work::default() || run.synthesis_model_work != Int4Work::default()
                    || run.synthesis_mask_node_visits != 0 || run.synthesis.answer.is_some() || !run.synthesis.citations.is_empty() {
                    return Err(invalid());
                }
            }
            SynthesisStatus::Answered | SynthesisStatus::Abstained => {
                if run.evidence_passages == 0 || run.evidence_bytes == 0 || run.discovery.answered_chunks == 0
                    || run.synthesis_prompt_tokens == 0 || run.synthesis_generated_tokens == 0
                    || run.synthesis_generated_tokens > self.max_output_tokens { return Err(invalid()); }
                let planned = constrained_int4::planned_work(run.group_width, run.synthesis_prompt_tokens, self.max_output_tokens)
                    .map_err(Int4SourceError::from)?;
                let work = constrained_int4::planned_work(run.group_width, run.synthesis_prompt_tokens, run.synthesis_generated_tokens)
                    .map_err(Int4SourceError::from)?;
                if planned != run.synthesis_planned_work || work != run.synthesis_model_work { return Err(invalid()); }
                if run.synthesis.status == SynthesisStatus::Answered {
                    if !run.synthesis.answer.as_deref().is_some_and(|answer| has_text(answer)
                        && answer.chars().count() <= self.options.max_answer_scalars)
                        || run.synthesis.citations.is_empty() || run.synthesis.citations.len() > self.options.max_citations {
                        return Err(invalid());
                    }
                    for citation in &run.synthesis.citations {
                        evidence::check_original_citation(citation, self.options, self.discovery.source_span())?;
                    }
                } else if run.synthesis.answer.is_some() || !run.synthesis.citations.is_empty() { return Err(invalid()); }
            }
        }
        Ok(())
    }
}

/// Consumes the sealed source/question discovery plan. Callers cannot substitute
/// fabricated map results or an independently selected evidence subset.
pub struct PreparedInt4QuestionSynthesis<'s> {
    source: &'s str,
    question: String,
    map: PreparedInt4Question<'s>,
    identity: ExecutionIdentity,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits,
    options: AnswerOptions,
    expected: Int4QuestionSynthesisPreflight,
}
impl SourceTaskPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int4_question_synthesis_with_control<C: DecodeStepControl>(&self, source: &str,
        request: &SourceQuestionSynthesis, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, group_width: usize, control: &mut C)
        -> Result<Int4QuestionSynthesisPreflight, Int4SourceMapError> {
        checkpoint(control)?;
        let (discovery_limits, reserve) = reserve(request.limits, budget, planning, mapping, group_width)?;
        let discovery = self.preflight_int4_question_with_control(source, &request.question, budget, context,
            planning, discovery_limits, group_width, control)?;
        expectation(discovery, reserve, request, budget, mapping)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int4_question_synthesis_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &SourceQuestionSynthesis, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, group_width: usize, control: &mut C)
        -> Result<PreparedInt4QuestionSynthesis<'s>, Int4SourceMapError> {
        checkpoint(control)?;
        let (discovery_limits, reserve) = reserve(request.limits, budget, planning, mapping, group_width)?;
        let map = self.plan_int4_question_with_control(source, &request.question, budget, context,
            planning, discovery_limits, group_width, control)?;
        let expected = expectation(map.preflight_metadata(), reserve, request, budget, mapping)?;
        checkpoint(control)?;
        Ok(PreparedInt4QuestionSynthesis { source, question: copy(&request.question.question)?, map,
            identity: context.execution_identity().clone(), budget, planning, mapping, options: request.question.options, expected })
    }
}
fn expectation(discovery: Int4QuestionPreflight, synthesis_reserve: Int4Work, request: &SourceQuestionSynthesis,
    budget: TaskBudget, mapping: Int4SourceMapLimits) -> Result<Int4QuestionSynthesisPreflight, Int4SourceMapError> {
    let value = Int4QuestionSynthesisPreflight { discovery, synthesis_reserve, synthesis_masks: mapping.mask_visits_per_chunk,
        limits: request.limits, options: request.question.options, max_output_tokens: budget.max_output_tokens as usize };
    if !fits(value.reserved_model_work()?, mapping.max_model_work) || value.reserved_mask_visits()? > mapping.max_mask_visits {
        return Err(Int4SourceMapError::WorkLimit);
    }
    Ok(value)
}
fn reserve(limits: QuestionSynthesisLimits, budget: TaskBudget, planning: SourcePlanningLimits,
    mut mapping: Int4SourceMapLimits, group_width: usize) -> Result<(Int4SourceMapLimits, Int4Work), Int4SourceMapError> {
    mapping.validate(budget, planning)?;
    limits.validate(planning)?;
    let output = budget.max_output_tokens as usize;
    let prompt = planning.max_context_tokens.checked_sub(output).ok_or(Int4SourceMapError::InvalidLimits)?
        .min(budget.max_input_tokens as usize);
    if prompt == 0 { return Err(Int4SourceMapError::InvalidLimits); }
    let reserved = constrained_int4::planned_work(group_width, prompt, output).map_err(Int4SourceError::from)?;
    mapping.max_model_work = subtract(mapping.max_model_work, reserved)?;
    mapping.max_mask_visits = mapping.max_mask_visits.checked_sub(mapping.mask_visits_per_chunk)
        .ok_or(Int4SourceMapError::WorkLimit)?;
    Ok((mapping, reserved))
}
impl PreparedInt4QuestionSynthesis<'_> {
    pub fn preflight_metadata(&self) -> Int4QuestionSynthesisPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.map.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
    pub fn max_kv_bytes(&self) -> u64 { self.budget.max_kv_bytes }
    pub fn group_width(&self) -> usize { self.expected.discovery.group_width() }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4QuestionSynthesisRun, Int4SourceMapError> {
        checkpoint(control)?;
        if planner.tokenizer_digest() != self.identity.tokenizer_digest || *planner.template_digest() != self.identity.template_digest {
            return Err(Int4SourceMapError::Admission);
        }
        self.map.preflight(admitted, engine)?;
        let Self { source, question, map, identity, budget, planning, mapping, options, expected } = self;
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        expected.discovery.verify_completed(&discovery)?;
        let mut remaining = remaining_verification(expected.discovery.verification, discovery.verification_used)?;
        let collection = evidence::collect(source, discovery.mapped.root().value(), expected.limits, &mut remaining, control)?;
        let passages = collection.passages.len(); let bytes = collection.bytes;
        let (answer, prompt, tokens, planned, work, masks) = if passages == 0 {
            (SynthesisAnswer { status: SynthesisStatus::NoEvidenceCollected, answer: None, citations: Vec::new() },
                0, 0, Int4Work::default(), Int4Work::default(), 0)
        } else {
            let context = PlanContext::new(&identity, budget).map_err(|_| Int4SourceMapError::Admission)?;
            let request = SourceTaskRequest::Answer { question, passages: collection.copy_passages()?, options, budget };
            let plan = planner.plan_int4_with_control(&request, &context, planning, expected.discovery.group_width(), control)?;
            let planned = plan.planned_work();
            if !fits(planned, expected.synthesis_reserve) { return Err(Int4SourceMapError::WorkLimit); }
            let mut driver = NativeDriver { engine, vocabulary, control, limits: mapping };
            preflight(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), &driver)?;
            let native = driver.run(&plan, plan.execution_identity())?;
            check_run(&plan, &native, mapping.mask_visits_per_chunk)?;
            let work = native.model_work;
            let SourceTaskResult::Answer(raw) = native.result else { return Err(invalid()); };
            let masks = raw.mask_node_visit_charge; let tokens = raw.generated_token_ids.len();
            let answer = evidence::final_answer(source, raw, &collection, options, &mut remaining, control)?;
            (answer, plan.prompt_tokens(), tokens, planned, work, masks)
        };
        drop(collection);
        finish(discovery, answer, passages, bytes, prompt, tokens, planned, work, masks, remaining, expected,
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
    /// Original-document spans reachable through verified evidence only.
    pub citations: Vec<SourceCitation>,
}
#[derive(Serialize)]
pub struct Int4QuestionSynthesisRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub semantics: &'static str,
    pub evidence_passages: usize,
    pub evidence_bytes: usize,
    pub synthesis_prompt_tokens: usize,
    pub synthesis_generated_tokens: usize,
    pub synthesis_reserved_work: Int4Work,
    pub synthesis_planned_work: Int4Work,
    pub synthesis_model_work: Int4Work,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub synthesis_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SourceMapVerification,
    pub calibration: AnswerCalibration,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 2],
    pub synthesis: SynthesisAnswer,
    /// All alternatives and abstentions remain present after final synthesis.
    pub discovery: Int4QuestionRun,
}
fn remaining_verification(total: GroundingBudget, used: SourceMapVerification) -> Result<GroundingBudget, Int4SourceMapError> {
    Ok(GroundingBudget { max_fields: total.max_fields.checked_sub(used.fields).ok_or(Int4SourceMapError::WorkLimit)?,
        max_matches: total.max_matches.checked_sub(used.matches).ok_or(Int4SourceMapError::WorkLimit)?,
        max_scan_steps: total.max_scan_steps.checked_sub(used.scan_steps).ok_or(Int4SourceMapError::WorkLimit)? })
}
#[allow(clippy::too_many_arguments)]
fn finish<C: DecodeStepControl>(discovery: Int4QuestionRun, synthesis: SynthesisAnswer, evidence_passages: usize,
    evidence_bytes: usize, synthesis_prompt_tokens: usize, synthesis_generated_tokens: usize,
    synthesis_planned_work: Int4Work, synthesis_model_work: Int4Work, synthesis_mask_node_visits: u64,
    remaining: GroundingBudget, expected: Int4QuestionSynthesisPreflight, cap: u64, control: &mut C)
    -> Result<Int4QuestionSynthesisRun, Int4SourceMapError> {
    checkpoint(control)?;
    let verification_used = SourceMapVerification::used(expected.discovery.verification, remaining)?;
    let model_work = add_work(discovery.model_work, synthesis_model_work).ok_or(Int4SourceMapError::WorkLimit)?;
    let mask_node_visit_charge = discovery.mask_node_visit_charge.checked_add(synthesis_mask_node_visits)
        .ok_or(Int4SourceMapError::WorkLimit)?;
    let run = Int4QuestionSynthesisRun { schema_version: 1, execution: INT4_QUESTION_SYNTHESIS_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: expected.discovery.group_width(), semantics: SEMANTICS,
        evidence_passages, evidence_bytes, synthesis_prompt_tokens, synthesis_generated_tokens,
        synthesis_reserved_work: expected.synthesis_reserve, synthesis_planned_work, synthesis_model_work,
        reserved_model_work: expected.reserved_model_work()?, model_work,
        reserved_mask_node_visits: expected.reserved_mask_visits()?, synthesis_mask_node_visits, mask_node_visit_charge,
        verification_used, calibration: AnswerCalibration::Uncalibrated,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        untrusted_fields: ["synthesis", "discovery.mapped.root.value"], synthesis, discovery };
    expected.verify_completed(&run)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}
fn subtract(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4SourceMapError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int4SourceMapError::WorkLimit);
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn invalid() -> Int4SourceMapError { Int4SourceError::InvalidResult.into() }
fn copy(text: &str) -> Result<String, Int4SourceMapError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| Int4SourceMapError::Allocation)?;
    out.push_str(text); Ok(out)
}
fn vector<T>(count: usize) -> Result<Vec<T>, Int4SourceMapError> {
    let mut out = Vec::new(); out.try_reserve_exact(count).map_err(|_| Int4SourceMapError::Allocation)?; Ok(out)
}

#[cfg(test)] mod tests;
