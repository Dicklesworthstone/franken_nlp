//! Explicit lossy reduction through verified source quotations only. Every level
//! consumes its complete frontier; generated assertions never feed another pass.
use super::*;
use crate::tasks::source_planning::grouped_int4::capacity::Int4SourceMapCapacity;
use std::ops::Range;
mod execution;
mod receipt;
mod transport;

pub const INT4_SUMMARY_HIERARCHY_EXECUTION: &str = "grouped-int4-hierarchical-quote-summary-v1";
const SEMANTICS: &str = "lossy-quote-selection-every-level-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SummaryHierarchyLimits {
    /// Includes the final single-group level; never reset for a smaller frontier.
    pub max_levels: usize,
    /// Total additional native calls, including the final call, NOT per level.
    pub max_passes: usize,
    /// Additional exact grouping token counts. Compilation is separately bounded
    /// to two preparations per permitted pass plus the discovery map.
    pub max_tokenizer_calls: usize,
    pub max_tokenizer_bytes: u64,
}
impl Default for SummaryHierarchyLimits {
    fn default() -> Self {
        Self { max_levels: 8, max_passes: 16, max_tokenizer_calls: 8192, max_tokenizer_bytes: 64 * 1024 * 1024 }
    }
}
impl SummaryHierarchyLimits {
    pub fn validate(self) -> Result<(), Int4SourceMapError> {
        if !(1..=16).contains(&self.max_levels) || !(1..=256).contains(&self.max_passes)
            || !(1..=65_536).contains(&self.max_tokenizer_calls)
            || !(1..=1_000_000_000).contains(&self.max_tokenizer_bytes) { return Err(Int4SourceMapError::InvalidLimits); }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SummaryHierarchyPreflight {
    discovery: Int4SummarySynthesisPreflight,
    limits: SummaryHierarchyLimits,
    capacity: Int4SourceMapCapacity,
    reserved_work: Int4Work,
    reserved_masks: u64,
}
impl Int4SummaryHierarchyPreflight {
    pub fn chunk_count(&self) -> usize { self.discovery.chunk_count() }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.discovery.source_span() }
    pub fn group_width(&self) -> usize { self.discovery.group_width() }
    pub fn reserved_model_work(&self) -> Int4Work { self.reserved_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.reserved_masks }
    pub fn limits(&self) -> SummaryHierarchyLimits { self.limits }
    pub fn verify_completed(&self, run: &Int4SummaryHierarchyRun) -> Result<(), Int4SourceMapError> { receipt::verify(self, run) }
}
pub struct PreparedInt4SummaryHierarchy<'s> {
    base: PreparedInt4SummarySynthesis<'s>,
    expected: Int4SummaryHierarchyPreflight,
    mapping: Int4SourceMapLimits,
}
impl SourceTaskPlanner {
    /// Reserve maximum-context work for EVERY permitted reduction pass before
    /// discovery. Dynamic groups are later admitted a whole level at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int4_summary_hierarchy_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: SourceSummarySynthesis, hierarchy: SummaryHierarchyLimits, budget: TaskBudget,
        context: &PlanContext<'_>, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits,
        group_width: usize, control: &mut C) -> Result<PreparedInt4SummaryHierarchy<'s>, Int4SourceMapError> {
        checkpoint(control)?;
        hierarchy.validate()?; request.validate(planning)?;
        let capacity = self.int4_map_capacity_with_control(&SourceMapTask::Summarize(request.synthesis_options),
            budget, context, planning, group_width, control)?;
        let mut reduced = mapping;
        for _ in 1..hierarchy.max_passes { checkpoint(control)?; reduced = reserve(budget, planning, reduced, group_width)?.0; }
        let base = self.plan_int4_summary_synthesis_with_control(source, request, budget, context, planning,
            reduced, group_width, control)?;
        let discovery = base.preflight_metadata();
        let mut reserved_work = discovery.discovery.planned_work();
        for _ in 0..hierarchy.max_passes {
            reserved_work = add_work(reserved_work, discovery.synthesis_reserve).ok_or(Int4SourceMapError::WorkLimit)?;
        }
        let reserved_masks = discovery.synthesis_masks.checked_mul(hierarchy.max_passes as u64)
            .and_then(|n| n.checked_add(discovery.discovery.reserved_mask_visits())).ok_or(Int4SourceMapError::WorkLimit)?;
        if !fits(reserved_work, mapping.max_model_work) || reserved_masks > mapping.max_mask_visits { return Err(Int4SourceMapError::WorkLimit); }
        let expected = Int4SummaryHierarchyPreflight { discovery, limits: hierarchy, capacity, reserved_work, reserved_masks };
        checkpoint(control)?;
        Ok(PreparedInt4SummaryHierarchy { base, expected, mapping })
    }
}
impl PreparedInt4SummaryHierarchy<'_> {
    pub fn preflight_metadata(&self) -> Int4SummaryHierarchyPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.base.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.base.max_result_bytes() }
    pub fn max_kv_bytes(&self) -> u64 { self.base.max_kv_bytes() }
    pub fn group_width(&self) -> usize { self.expected.group_width() }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4SummaryHierarchyRun, Int4SourceMapError> {
        checkpoint(control)?;
        let Self { base, expected, mapping } = self;
        if planner.tokenizer_digest() != base.identity.tokenizer_digest || *planner.template_digest() != base.identity.template_digest {
            return Err(Int4SourceMapError::Admission);
        }
        base.map.preflight(admitted, engine)?;
        let PreparedInt4SummarySynthesis { source, map, identity, budget, planning, request, .. } = base;
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        expected.discovery.discovery.verify_completed(&discovery)?;
        let mut remaining = remaining_verification(request.limits.verification, discovery.verification_used)?;
        let collection = evidence::collect(source, discovery.mapped.root().value(), request.map_options,
            request.limits, &mut remaining, control)?;
        let initial = EvidenceSize::of(&collection);
        let mut driver = execution::Native { planner, identity: &identity, budget, planning,
            options: request.synthesis_options, mapping, engine, vocabulary, group_width: expected.group_width() };
        let reduced = execution::reduce(source, collection, request, expected, mapping.reduction.max_result_bytes as u64,
            &mut remaining, &mut driver, control)?;
        finish(discovery, initial, reduced, remaining, expected, mapping.reduction.max_result_bytes as u64, control)
    }
}
fn finish<C: DecodeStepControl>(discovery: Int4SourceMapRun, initial: EvidenceSize, reduced: execution::Reduced,
    remaining: GroundingBudget, expected: Int4SummaryHierarchyPreflight, cap: u64, control: &mut C)
    -> Result<Int4SummaryHierarchyRun, Int4SourceMapError> {
    checkpoint(control)?;
    let run = Int4SummaryHierarchyRun { schema_version: 1, execution: INT4_SUMMARY_HIERARCHY_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: expected.group_width(), semantics: SEMANTICS, limits: expected.limits,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        untrusted_fields: ["passes", "discovery.mapped.root.value"], initial_evidence: initial,
        status: reduced.status, final_pass: reduced.final_pass, reserved_model_work: expected.reserved_work,
        planned_model_work: add_work(discovery.planned_model_work, reduced.planned).ok_or(Int4SourceMapError::WorkLimit)?,
        model_work: add_work(discovery.model_work, reduced.actual).ok_or(Int4SourceMapError::WorkLimit)?,
        reserved_mask_node_visits: expected.reserved_masks,
        mask_node_visit_charge: discovery.mask_node_visit_charge.checked_add(reduced.masks).ok_or(Int4SourceMapError::WorkLimit)?,
        verification_used: SourceMapVerification::used(expected.discovery.limits.verification, remaining)?,
        tokenizer_work: reduced.tokenizer, levels: reduced.levels, passes: reduced.passes, discovery };
    expected.verify_completed(&run)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EvidenceSize { pub segments: usize, pub bytes: usize }
impl EvidenceSize {
    fn of(collection: &evidence::Collection) -> Self { Self { segments: collection.segment_count(), bytes: collection.text.len() } }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct HierarchyTokenizerWork { pub calls: usize, pub bytes: u64 }
#[derive(Serialize)]
pub struct SummaryHierarchyPass {
    pub level: usize,
    pub group: usize,
    /// Indices into this level's ordered evidence frontier, not source chunks.
    pub input_segments: Range<usize>,
    pub input_bytes: usize,
    pub input_tokens: usize,
    pub planned_model_work: Int4Work,
    /// Native result with coordinates LOCAL to this group's quote document.
    pub native: Int4SourceTaskRun,
    /// All generated bullets with verified ORIGINAL-document coordinates.
    pub bullets: Vec<CitedBullet>,
}
#[derive(Serialize)]
pub struct SummaryHierarchyLevel {
    pub input: EvidenceSize,
    pub passes: Range<usize>,
    /// None marks a final single-group pass. Some marks lossy quote selection.
    pub next_evidence: Option<EvidenceSize>,
}
#[derive(Serialize)]
pub struct Int4SummaryHierarchyRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub semantics: &'static str,
    pub limits: SummaryHierarchyLimits,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 2],
    pub initial_evidence: EvidenceSize,
    pub status: SummarySynthesisStatus,
    /// Final single-group pass index, without duplicating its payload. None
    /// means no initial evidence or an exhausted intermediate quote frontier.
    pub final_pass: Option<usize>,
    pub reserved_model_work: Int4Work,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SourceMapVerification,
    pub tokenizer_work: HierarchyTokenizerWork,
    pub levels: Vec<SummaryHierarchyLevel>,
    pub passes: Vec<SummaryHierarchyPass>,
    pub discovery: Int4SourceMapRun,
}
impl Int4SummaryHierarchyRun {
    pub fn bullets(&self) -> &[CitedBullet] {
        self.final_pass.and_then(|i| self.passes.get(i)).map_or(&[], |pass| pass.bullets.as_slice())
    }
}

#[cfg(test)] mod tests;
