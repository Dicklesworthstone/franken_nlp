//! Explicit lossy quotation selection for evidence exceeding one QA context.
//! Every pass sees the original question; generated answers never become source.
use super::*;
use std::ops::Range;
mod execution;
mod receipt;
mod transport;
/// Native discovery cohorts with the existing bounded quotation hierarchy.
pub mod cohort;

pub const INT4_QUESTION_HIERARCHY_EXECUTION: &str = "grouped-int4-hierarchical-quote-answer-v1";
const SEMANTICS: &str = "lossy-question-quote-selection-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct QuestionHierarchyLimits {
    /// Includes the final single-group level, never renewed for a smaller frontier.
    pub max_levels: usize,
    /// ALL additional native calls across ALL levels, including the final call.
    pub max_passes: usize,
    /// Exact grouping attempts over the complete question, manifest and source.
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
    pub fn validate(self) -> Result<(), Int4SourceMapError> {
        if !(1..=16).contains(&self.max_levels) || !(1..=256).contains(&self.max_passes)
            || !(1..=65_536).contains(&self.max_tokenizer_calls)
            || !(1..=1_000_000_000).contains(&self.max_tokenizer_bytes) { return Err(Int4SourceMapError::InvalidLimits); }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Capacity { scaffold: usize, prompt: usize, input_bytes: usize, question_bytes: usize, eos: u32 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionHierarchyPreflight {
    base: Int4QuestionSynthesisPreflight,
    limits: QuestionHierarchyLimits,
    capacity: Capacity,
    reserved_work: Int4Work,
    reserved_masks: u64,
}
impl Int4QuestionHierarchyPreflight {
    pub fn discovery(&self) -> Int4QuestionPreflight { self.base.discovery }
    pub fn group_width(&self) -> usize { self.base.discovery.group_width() }
    pub fn limits(&self) -> QuestionHierarchyLimits { self.limits }
    pub fn reserved_model_work(&self) -> Int4Work { self.reserved_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.reserved_masks }
    pub fn verify_completed(&self, run: &Int4QuestionHierarchyRun) -> Result<(), Int4SourceMapError> { receipt::verify(self, run) }
}
pub struct PreparedInt4QuestionHierarchy<'s> {
    base: PreparedInt4QuestionSynthesis<'s>,
    expected: Int4QuestionHierarchyPreflight,
    mapping: Int4SourceMapLimits,
}
struct HierarchyContinuation<'s> {
    base: Continuation<'s>,
    expected: Int4QuestionHierarchyPreflight,
    // This is the original whole-hierarchy envelope, before every additional
    // pass was reserved out of the sealed discovery mapping limits.
    mapping: Int4SourceMapLimits,
}
impl SourceTaskPlanner {
    /// Weight-free discovery preflight with all permitted hierarchy work reserved.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int4_question_hierarchy_with_control<C: DecodeStepControl>(&self, source: &str,
        request: &SourceQuestionSynthesis, hierarchy: QuestionHierarchyLimits, budget: TaskBudget,
        context: &PlanContext<'_>, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits,
        group_width: usize, control: &mut C) -> Result<Int4QuestionHierarchyPreflight, Int4SourceMapError> {
        let reduced = reserve_hierarchy(request, hierarchy, budget, planning, mapping, group_width, control)?;
        let base = self.preflight_int4_question_synthesis_with_control(source, request, budget, context,
            planning, reduced, group_width, control)?;
        hierarchy_expectation(self, request, hierarchy, base, budget, planning, mapping, control)
    }
    /// Every actual group is subsequently compiled and admitted before any
    /// forward in its level. The sealed discovery cannot be replaced by callers.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int4_question_hierarchy_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &SourceQuestionSynthesis, hierarchy: QuestionHierarchyLimits, budget: TaskBudget,
        context: &PlanContext<'_>, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits,
        group_width: usize, control: &mut C) -> Result<PreparedInt4QuestionHierarchy<'s>, Int4SourceMapError> {
        let reduced = reserve_hierarchy(request, hierarchy, budget, planning, mapping, group_width, control)?;
        let base = self.plan_int4_question_synthesis_with_control(source, request, budget, context,
            planning, reduced, group_width, control)?;
        let expected = hierarchy_expectation(self, request, hierarchy, base.expected, budget, planning, mapping, control)?;
        Ok(PreparedInt4QuestionHierarchy { base, expected, mapping })
    }
}
#[allow(clippy::too_many_arguments)]
fn reserve_hierarchy<C: DecodeStepControl>(request: &SourceQuestionSynthesis, hierarchy: QuestionHierarchyLimits,
    budget: TaskBudget, planning: SourcePlanningLimits, mut mapping: Int4SourceMapLimits, width: usize, control: &mut C)
    -> Result<Int4SourceMapLimits, Int4SourceMapError> {
    checkpoint(control)?;
    hierarchy.validate()?;
    for _ in 1..hierarchy.max_passes {
        checkpoint(control)?;
        mapping = reserve(request.limits, budget, planning, mapping, width)?.0;
    }
    Ok(mapping)
}
#[allow(clippy::too_many_arguments)]
fn hierarchy_expectation<C: DecodeStepControl>(planner: &SourceTaskPlanner, request: &SourceQuestionSynthesis,
    limits: QuestionHierarchyLimits, base: Int4QuestionSynthesisPreflight, budget: TaskBudget,
    planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, control: &mut C)
    -> Result<Int4QuestionHierarchyPreflight, Int4SourceMapError> {
    let mut reserved_work = base.discovery.planned_work();
    for _ in 0..limits.max_passes { reserved_work = add_work(reserved_work, base.synthesis_reserve).ok_or(Int4SourceMapError::WorkLimit)?; }
    let reserved_masks = base.synthesis_masks.checked_mul(limits.max_passes as u64)
        .and_then(|n| n.checked_add(base.discovery.reserved_mask_visits())).ok_or(Int4SourceMapError::WorkLimit)?;
    if !fits(reserved_work, mapping.max_model_work) || reserved_masks > mapping.max_mask_visits { return Err(Int4SourceMapError::WorkLimit); }
    let schema = request.question.options.schema_source().map_err(Int4SourceError::from)?;
    let mut scaffold = 0_usize;
    for fragment in render_fragments(BuiltInTask::Answer, &schema).map_err(Int4SourceError::from)? {
        checkpoint(control)?;
        let ids = planner.tokenizer.tokenizer().encode_ids_with_options(&fragment,
            EncodeOptions { add_bos: false, add_eos: false })
            .map_err(|_| Int4SourceError::from(SourcePlanningError::Contract("INT4 answer hierarchy scaffold encoding")))?;
        scaffold = scaffold.checked_add(ids.len()).ok_or(Int4SourceMapError::WorkLimit)?;
    }
    let prompt = planning.max_context_tokens.checked_sub(budget.max_output_tokens as usize)
        .ok_or(Int4SourceMapError::InvalidLimits)?.min(budget.max_input_tokens as usize);
    if scaffold == 0 || scaffold >= prompt { return Err(Int4SourceMapError::InvalidLimits); }
    let capacity = Capacity { scaffold, prompt, input_bytes: planning.max_input_bytes.min(budget.max_input_tokens as usize),
        question_bytes: request.question.question.len(), eos: planner.eos };
    checkpoint(control)?;
    Ok(Int4QuestionHierarchyPreflight { base, limits, capacity, reserved_work, reserved_masks })
}
impl<'s> PreparedInt4QuestionHierarchy<'s> {
    pub fn preflight_metadata(&self) -> Int4QuestionHierarchyPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.base.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.base.max_result_bytes() }
    pub fn max_kv_bytes(&self) -> u64 { self.base.max_kv_bytes() }
    pub fn group_width(&self) -> usize { self.expected.group_width() }
    fn into_phases(self) -> (PreparedInt4Question<'s>, HierarchyContinuation<'s>) {
        let Self { base, expected, mapping } = self;
        let (map, base) = base.into_phases();
        (map, HierarchyContinuation { base, expected, mapping })
    }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4QuestionHierarchyRun, Int4SourceMapError> {
        checkpoint(control)?;
        self.base.check_planner(planner)?;
        self.base.map.preflight(admitted, engine)?;
        let (map, continuation) = self.into_phases();
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        continuation.execute(discovery, control, |continuation, collection, remaining, control| {
            continuation.reduce(collection, remaining, planner, engine, vocabulary, control)
        })
    }
}
impl HierarchyContinuation<'_> {
    fn execute<C: DecodeStepControl, F>(self, discovery: Int4QuestionRun, control: &mut C, reduce: F)
        -> Result<Int4QuestionHierarchyRun, Int4SourceMapError>
        where F: FnOnce(&Self, evidence::Collection, &mut GroundingBudget, &mut C)
            -> Result<execution::Reduced, Int4SourceMapError> {
        let expected = self.expected;
        expected.discovery().verify_completed(&discovery)?;
        let mut remaining = remaining_verification(expected.base.discovery.verification, discovery.verification_used)?;
        let collection = evidence::collect(self.base.source, discovery.mapped.root().value(), expected.base.limits, &mut remaining, control)?;
        let initial = EvidenceSize::of(&collection);
        let reduced = if collection.passages.is_empty() {
            execution::empty(expected, control)?
        } else {
            reduce(&self, collection, &mut remaining, control)?
        };
        finish(discovery, initial, reduced, remaining, expected, self.mapping.reduction.max_result_bytes as u64, control)
    }
    #[allow(clippy::too_many_arguments)]
    fn reduce<C: DecodeStepControl>(&self, collection: evidence::Collection, remaining: &mut GroundingBudget,
        planner: &SourceTaskPlanner, engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<execution::Reduced, Int4SourceMapError> {
        let base = &self.base;
        let mut driver = execution::Native { planner, identity: &base.identity, question: &base.question,
            budget: base.budget, planning: base.planning, options: base.options, mapping: self.mapping,
            engine, vocabulary, expected: self.expected };
        execution::reduce(base.source, collection, self.expected, self.mapping.reduction.max_result_bytes as u64,
            remaining, &mut driver, control)
    }
}
fn finish<C: DecodeStepControl>(discovery: Int4QuestionRun, initial_evidence: EvidenceSize, reduced: execution::Reduced,
    remaining: GroundingBudget, expected: Int4QuestionHierarchyPreflight, cap: u64, control: &mut C)
    -> Result<Int4QuestionHierarchyRun, Int4SourceMapError> {
    checkpoint(control)?;
    let run = Int4QuestionHierarchyRun { schema_version: 1, execution: INT4_QUESTION_HIERARCHY_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: expected.group_width(), semantics: SEMANTICS, limits: expected.limits,
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, untrusted_fields: ["passes", "discovery.mapped.root.value"],
        initial_evidence, status: reduced.status, final_pass: reduced.final_pass, reserved_model_work: expected.reserved_work,
        planned_model_work: add_work(discovery.planned_model_work, reduced.planned).ok_or(Int4SourceMapError::WorkLimit)?,
        model_work: add_work(discovery.model_work, reduced.actual).ok_or(Int4SourceMapError::WorkLimit)?,
        reserved_mask_node_visits: expected.reserved_masks,
        mask_node_visit_charge: discovery.mask_node_visit_charge.checked_add(reduced.masks).ok_or(Int4SourceMapError::WorkLimit)?,
        verification_used: SourceMapVerification::used(expected.base.discovery.verification, remaining)?,
        tokenizer_work: reduced.tokenizer, levels: reduced.levels, passes: reduced.passes, discovery };
    expected.verify_completed(&run)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
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
    /// Quotation bytes only, excluding synthetic joins, question and manifest.
    pub input_bytes: usize,
    /// Exact aggregate untrusted bytes: question, joined quotations and manifest.
    pub context_bytes: usize,
    pub prompt_tokens: usize,
    pub planned_model_work: Int4Work,
    /// Full native receipt with passage-local citations and generated token ids.
    pub native: Int4SourceTaskRun,
    /// The same answer with independently lifted ORIGINAL-document citations.
    pub answer: SynthesisAnswer,
}
#[derive(Serialize)]
pub struct QuestionHierarchyLevel {
    pub input: EvidenceSize,
    pub passes: Range<usize>,
    /// None denotes the final single-group answer, Some denotes quote selection.
    pub next_evidence: Option<EvidenceSize>,
}
#[derive(Serialize)]
pub struct Int4QuestionHierarchyRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
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
    pub reserved_model_work: Int4Work,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SourceMapVerification,
    pub tokenizer_work: HierarchyTokenizerWork,
    pub levels: Vec<QuestionHierarchyLevel>,
    pub passes: Vec<QuestionHierarchyPass>,
    pub discovery: Int4QuestionRun,
}
impl Int4QuestionHierarchyRun {
    pub fn answer(&self) -> Option<&SynthesisAnswer> { self.final_pass.and_then(|i| self.passes.get(i)).map(|pass| &pass.answer) }
}

#[cfg(test)] mod tests;
