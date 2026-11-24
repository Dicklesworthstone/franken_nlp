//! Explicit lossy quotation selection for evidence exceeding one QA context.
//! Every pass sees the original question; generated answers never become source.
use super::*;
use std::ops::Range;
mod execution;
mod receipt;
mod transport;
/// Shared checked admission for complete hosted and CLI hierarchy ownership.
pub mod memory;

pub const INT8_QUESTION_HIERARCHY_EXECUTION: &str = "portable-int8-hierarchical-quote-answer-v1";
const SEMANTICS: &str = "lossy-question-quote-selection-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct QuestionHierarchyLimits {
    /// Includes the final single-group level, never renewed by a smaller frontier.
    pub max_levels: usize,
    /// ALL additional native calls across ALL levels, including the final call.
    pub max_passes: usize,
    /// Exact grouping attempts over the question, manifest and complete quotes.
    /// Native compilation is separately bounded to two preparations per pass.
    pub max_tokenizer_calls: usize,
    pub max_tokenizer_bytes: u64,
}
impl Default for QuestionHierarchyLimits {
    fn default() -> Self {
        Self { max_levels: 8, max_passes: 16, max_tokenizer_calls: 8192, max_tokenizer_bytes: 64 * 1024 * 1024 }
    }
}
impl QuestionHierarchyLimits {
    pub fn validate(self) -> Result<(), Int8SourceMapError> {
        if !(1..=16).contains(&self.max_levels) || !(1..=256).contains(&self.max_passes)
            || !(1..=65_536).contains(&self.max_tokenizer_calls)
            || !(1..=1_000_000_000).contains(&self.max_tokenizer_bytes) {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Capacity { scaffold: usize, prompt: usize, input_bytes: usize, question_bytes: usize, eos: u32 }
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int8QuestionHierarchyPreflight {
    base: Int8QuestionSynthesisPreflight,
    limits: QuestionHierarchyLimits,
    capacity: Capacity,
    options: AnswerOptions,
    output_tokens: usize,
    reserved_work: Int8Work,
    reserved_masks: u64,
    request_binding: [u8; 32],
    max_result_bytes: u64,
}
impl fmt::Debug for Int8QuestionHierarchyPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8QuestionHierarchyPreflight").field("base", &self.base)
            .field("limits", &self.limits).field("capacity", &self.capacity)
            .field("reserved_work", &self.reserved_work).field("reserved_masks", &self.reserved_masks)
            .finish_non_exhaustive()
    }
}
impl Int8QuestionHierarchyPreflight {
    pub fn discovery(&self) -> Int8QuestionPreflight { self.base.discovery }
    pub fn limits(&self) -> QuestionHierarchyLimits { self.limits }
    pub fn reserved_model_work(&self) -> Int8Work { self.reserved_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.reserved_masks }
    pub fn verify_completed(&self, run: &Int8QuestionHierarchyRun) -> Result<(), Int8SourceMapError> {
        self.verify_completed_with_control(run, &mut binding::Continue)
    }
    /// Verify every intermediate and final payload with its exact discovery owner.
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8QuestionHierarchyRun,
        control: &mut C) -> Result<(), Int8SourceMapError> {
        checkpoint(control)?;
        if run.request_binding != self.request_binding || run.result_binding != binding::composed_result(
            b"fnlp-int8-question-hierarchy-result-v1", self.request_binding,
            &[run.discovery.request_binding, run.discovery.result_binding], run, self.max_result_bytes, control)? {
            return Err(invalid());
        }
        receipt::verify(self, run, control)?;
        checkpoint(control)?;
        Ok(())
    }
}
pub struct PreparedInt8QuestionHierarchy<'s> {
    base: PreparedInt8QuestionSynthesis<'s>,
    expected: Int8QuestionHierarchyPreflight,
    mapping: Int8SourceMapLimits,
}
struct HierarchyContinuation<'s> {
    base: Continuation<'s>,
    expected: Int8QuestionHierarchyPreflight,
    /// Original whole-hierarchy envelope, before reserving every extra pass.
    mapping: Int8SourceMapLimits,
}
impl SourceTaskPlanner {
    /// Weight-free discovery preflight with every permitted hierarchy pass funded.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int8_question_hierarchy_with_control<C: DecodeStepControl>(&self, source: &str,
        request: &SourceQuestionSynthesis, hierarchy: QuestionHierarchyLimits, budget: TaskBudget,
        context: &PlanContext<'_>, planning: SourcePlanningLimits, mapping: Int8SourceMapLimits,
        control: &mut C) -> Result<Int8QuestionHierarchyPreflight, Int8SourceMapError> {
        let reduced = reserve_hierarchy(request, hierarchy, budget, planning, mapping, control)?;
        let base = self.preflight_int8_question_synthesis_with_control(source, request, budget, context,
            planning, reduced, control)?;
        hierarchy_expectation(self, request, hierarchy, base, budget, planning, mapping, control)
    }
    /// All actual group contexts are compiled and admitted before the first
    /// forward in their level. Callers cannot replace sealed discovery evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int8_question_hierarchy_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &SourceQuestionSynthesis, hierarchy: QuestionHierarchyLimits, budget: TaskBudget,
        context: &PlanContext<'_>, planning: SourcePlanningLimits, mapping: Int8SourceMapLimits,
        control: &mut C) -> Result<PreparedInt8QuestionHierarchy<'s>, Int8SourceMapError> {
        let reduced = reserve_hierarchy(request, hierarchy, budget, planning, mapping, control)?;
        let base = self.plan_int8_question_synthesis_with_control(source, request, budget, context,
            planning, reduced, control)?;
        let expected = hierarchy_expectation(self, request, hierarchy, base.expected, budget, planning, mapping, control)?;
        Ok(PreparedInt8QuestionHierarchy { base, expected, mapping })
    }
}
fn reserve_hierarchy<C: DecodeStepControl>(request: &SourceQuestionSynthesis, hierarchy: QuestionHierarchyLimits,
    budget: TaskBudget, planning: SourcePlanningLimits, mut mapping: Int8SourceMapLimits, control: &mut C)
    -> Result<Int8SourceMapLimits, Int8SourceMapError> {
    checkpoint(control)?;
    hierarchy.validate()?;
    for _ in 1..hierarchy.max_passes {
        checkpoint(control)?;
        mapping = reserve(request.limits, budget, planning, mapping)?.0;
    }
    Ok(mapping)
}
#[allow(clippy::too_many_arguments)]
fn hierarchy_expectation<C: DecodeStepControl>(planner: &SourceTaskPlanner, request: &SourceQuestionSynthesis,
    limits: QuestionHierarchyLimits, base: Int8QuestionSynthesisPreflight, budget: TaskBudget,
    planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
    -> Result<Int8QuestionHierarchyPreflight, Int8SourceMapError> {
    let mut reserved_work = base.discovery.planned_work();
    for _ in 0..limits.max_passes {
        checkpoint(control)?;
        reserved_work = add_work(reserved_work, base.synthesis_reserve).ok_or(Int8SourceMapError::WorkLimit)?;
    }
    let reserved_masks = base.synthesis_masks.checked_mul(limits.max_passes as u64)
        .and_then(|n| n.checked_add(base.discovery.reserved_mask_visits())).ok_or(Int8SourceMapError::WorkLimit)?;
    if !within(reserved_work, mapping.max_model_work) || reserved_masks > mapping.max_mask_visits {
        return Err(Int8SourceMapError::WorkLimit);
    }
    let schema = request.question.options.schema_source().map_err(Int8SourceError::from)?;
    let mut scaffold = 0_usize;
    for fragment in render_fragments(BuiltInTask::Answer, &schema).map_err(Int8SourceError::from)? {
        checkpoint(control)?;
        let ids = planner.tokenizer.tokenizer().encode_ids_with_options(&fragment,
            EncodeOptions { add_bos: false, add_eos: false })
            .map_err(|_| Int8SourceError::from(SourcePlanningError::Contract("INT8 answer hierarchy scaffold encoding")))?;
        scaffold = scaffold.checked_add(ids.len()).ok_or(Int8SourceMapError::WorkLimit)?;
    }
    let prompt = planning.max_context_tokens.checked_sub(budget.max_output_tokens as usize)
        .ok_or(Int8SourceMapError::InvalidLimits)?.min(budget.max_input_tokens as usize);
    if scaffold == 0 || scaffold >= prompt { return Err(Int8SourceMapError::InvalidLimits); }
    let input_bytes = match planner.document_encoding() {
        crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback =>
            planning.max_input_bytes.min(budget.max_input_tokens as usize),
        crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(policy) =>
            planning.max_input_bytes.min(policy.max_input_bytes),
    };
    let capacity = Capacity { scaffold, prompt, input_bytes,
        question_bytes: request.question.question.len(), eos: planner.eos };
    let request_binding = binding::composed_request(b"fnlp-int8-question-hierarchy-request-v1",
        base.request_binding, &(limits, (capacity.scaffold, capacity.prompt, capacity.input_bytes,
            capacity.question_bytes, capacity.eos), request.question.options, budget.max_output_tokens,
            reserved_work, reserved_masks), mapping, control)?;
    checkpoint(control)?;
    Ok(Int8QuestionHierarchyPreflight { base, limits, capacity, options: request.question.options,
        output_tokens: budget.max_output_tokens as usize, reserved_work, reserved_masks,
        request_binding, max_result_bytes: mapping.reduction.max_result_bytes as u64 })
}
impl<'s> PreparedInt8QuestionHierarchy<'s> {
    pub fn preflight_metadata(&self) -> Int8QuestionHierarchyPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.base.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
    fn into_phases(self) -> (PreparedInt8Question<'s>, HierarchyContinuation<'s>) {
        let Self { base, expected, mapping } = self;
        let (map, base) = base.into_phases();
        (map, HierarchyContinuation { base, expected, mapping })
    }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8QuestionHierarchyRun, Int8SourceMapError> {
        self.execute_scheduled(admitted, planner, engine, vocabulary, None, control)
    }
    /// Apply bounded prompt morsels to every discovery and hierarchy pass on
    /// the same engine, without renewing evidence, tokenizer or native budgets.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
        prefill: Int8PrefillLimits, control: &mut C) -> Result<Int8QuestionHierarchyRun, Int8SourceMapError> {
        self.execute_scheduled(admitted, planner, engine, vocabulary, Some(prefill), control)
    }
    fn execute_scheduled<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
        prefill: Option<Int8PrefillLimits>, control: &mut C) -> Result<Int8QuestionHierarchyRun, Int8SourceMapError> {
        checkpoint(control)?;
        if let Some(prefill) = prefill { check_prefill(prefill)?; }
        self.base.check_planner(planner)?;
        self.base.map.preflight(admitted, engine)?;
        let (map, continuation) = self.into_phases();
        let discovery = match prefill {
            Some(prefill) => map.execute_layer_major_with_control(admitted, engine, vocabulary, prefill, control),
            None => map.execute_with_control(admitted, engine, vocabulary, control),
        }?;
        continuation.execute(discovery, control, |continuation, collection, remaining, control| {
            let base = &continuation.base;
            let mut driver = execution::Native { planner: &base.operation, identity: &base.identity, question: &base.question,
                budget: base.budget, planning: base.planning, options: base.options, mapping: continuation.mapping,
                engine, vocabulary, prefill, expected: continuation.expected };
            execution::reduce(base.source, collection, continuation.expected, continuation.mapping.reduction.max_result_bytes as u64,
                remaining, &mut driver, control)
        })
    }
}
impl HierarchyContinuation<'_> {
    fn execute<C: DecodeStepControl, F>(self, discovery: Int8QuestionRun, control: &mut C, reduce: F)
        -> Result<Int8QuestionHierarchyRun, Int8SourceMapError>
        where F: FnOnce(&Self, evidence::Collection, &mut GroundingBudget, &mut C)
            -> Result<execution::Reduced, Int8SourceMapError> {
        let expected = self.expected;
        expected.discovery().verify_completed_with_control(&discovery, control)?;
        let mut remaining = remaining_verification(&discovery, expected.base.discovery.verification, control)?;
        let collection = evidence::collect(self.base.source, discovery.mapped.root().value(),
            expected.base.limits, &mut remaining, control)?;
        let initial = EvidenceSize::of(&collection);
        let reduced = if collection.passages.is_empty() {
            execution::empty(expected, control)?
        } else { reduce(&self, collection, &mut remaining, control)? };
        finish(discovery, initial, reduced, remaining, expected, control)
    }
}
fn finish<C: DecodeStepControl>(discovery: Int8QuestionRun, initial_evidence: EvidenceSize,
    reduced: execution::Reduced, remaining: GroundingBudget, expected: Int8QuestionHierarchyPreflight,
    control: &mut C) -> Result<Int8QuestionHierarchyRun, Int8SourceMapError> {
    checkpoint(control)?;
    let total = expected.base.discovery.verification;
    let mut run = Int8QuestionHierarchyRun { schema_version: 1, execution: INT8_QUESTION_HIERARCHY_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, semantics: SEMANTICS, limits: expected.limits,
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, untrusted_fields: ["passes", "discovery.mapped.root.value"],
        initial_evidence, status: reduced.status, final_pass: reduced.final_pass, reserved_model_work: expected.reserved_work,
        planned_model_work: add_work(discovery.planned_model_work, reduced.planned).ok_or(Int8SourceMapError::WorkLimit)?,
        model_work: add_work(discovery.model_work, reduced.actual).ok_or(Int8SourceMapError::WorkLimit)?,
        reserved_mask_node_visits: expected.reserved_masks,
        mask_node_visit_charge: discovery.mask_node_visit_charge.checked_add(reduced.masks).ok_or(Int8SourceMapError::WorkLimit)?,
        verification_used: QuestionVerificationWork {
            fields: total.max_fields.checked_sub(remaining.max_fields).ok_or_else(invalid)?,
            matches: total.max_matches.checked_sub(remaining.max_matches).ok_or_else(invalid)?,
            scan_steps: total.max_scan_steps.checked_sub(remaining.max_scan_steps).ok_or_else(invalid)? },
        tokenizer_work: reduced.tokenizer, levels: reduced.levels, passes: reduced.passes, discovery,
        request_binding: expected.request_binding, result_binding: [0; 32] };
    run.result_binding = binding::composed_result(b"fnlp-int8-question-hierarchy-result-v1",
        run.request_binding, &[run.discovery.request_binding, run.discovery.result_binding],
        &run, expected.max_result_bytes, control)?;
    expected.verify_completed_with_control(&run, control)?;
    checkpoint(control)?;
    Ok(run)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EvidenceSize { pub passages: usize, pub bytes: usize }
impl EvidenceSize {
    fn of(collection: &evidence::Collection) -> Self { Self { passages: collection.passages.len(), bytes: collection.bytes } }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct HierarchyTokenizerWork { pub calls: usize, pub bytes: u64 }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionHierarchyStatus { NoEvidenceCollected, EvidenceExhausted, Answered, Abstained }
#[derive(Serialize)]
pub struct QuestionHierarchyPass {
    pub level: usize,
    pub group: usize,
    pub input_passages: Range<usize>,
    /// Quote bytes only, excluding question, manifest and synthetic joins.
    pub input_bytes: usize,
    /// Exact aggregate untrusted bytes: question, joined quotes and manifest.
    pub context_bytes: usize,
    pub prompt_tokens: usize,
    pub planned_model_work: Int8Work,
    /// Full native receipt with passage-local citations and generated token ids.
    pub native: Int8SourceTaskRun,
    /// The same answer with independently lifted ORIGINAL-document citations.
    pub answer: SynthesisAnswer,
}
#[derive(Serialize)]
pub struct QuestionHierarchyLevel {
    pub input: EvidenceSize,
    pub passes: Range<usize>,
    /// None is the final single-group answer; Some is a quote-selection level.
    pub next_evidence: Option<EvidenceSize>,
}
#[derive(Serialize)]
pub struct Int8QuestionHierarchyRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub semantics: &'static str,
    pub limits: QuestionHierarchyLimits,
    pub calibration: AnswerCalibration,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 2],
    pub initial_evidence: EvidenceSize,
    pub status: QuestionHierarchyStatus,
    /// No duplicate final payload. None means no evidence or exhausted selection.
    pub final_pass: Option<usize>,
    pub reserved_model_work: Int8Work,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: QuestionVerificationWork,
    pub tokenizer_work: HierarchyTokenizerWork,
    pub levels: Vec<QuestionHierarchyLevel>,
    pub passes: Vec<QuestionHierarchyPass>,
    pub discovery: Int8QuestionRun,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl Int8QuestionHierarchyRun {
    pub fn answer(&self) -> Option<&SynthesisAnswer> {
        self.final_pass.and_then(|index| self.passes.get(index)).map(|pass| &pass.answer)
    }
}

#[cfg(test)] mod tests;
