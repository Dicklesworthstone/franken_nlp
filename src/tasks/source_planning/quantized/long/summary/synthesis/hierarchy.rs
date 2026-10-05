//! Explicit, lossy, evidence-only hierarchical summary reduction.
//! Every level consumes its entire frontier. Only independently lifted quotes,
//! never generated assertions, may enter the next level. All passes share one
//! reservation and controller; nonprogress and exhausted limits refuse the run.
use super::*;
use crate::tasks::source_planning::quantized::capacity::Int8SourceMapCapacity;
use std::ops::Range;
mod execution;
mod receipt;
mod transport;

pub const INT8_SUMMARY_HIERARCHY_EXECUTION: &str = "portable-int8-hierarchical-quote-summary-v1";
const SEMANTICS: &str = "lossy-quote-selection-every-level-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SummaryHierarchyLimits {
    /// Includes the final single-group level. Never renewed by a smaller frontier.
    pub max_levels: usize,
    /// Total additional native calls, including the final call, NOT per level.
    pub max_passes: usize,
    /// Additional exact source-token counts for greedy grouping, not compilation.
    pub max_tokenizer_calls: usize,
    pub max_tokenizer_bytes: u64,
}
impl Default for SummaryHierarchyLimits {
    fn default() -> Self {
        Self { max_levels: 8, max_passes: 32, max_tokenizer_calls: 8192, max_tokenizer_bytes: 64 * 1024 * 1024 }
    }
}
impl SummaryHierarchyLimits {
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
pub struct Int8SummaryHierarchyPreflight {
    discovery: Int8SummarySynthesisPreflight,
    limits: SummaryHierarchyLimits,
    capacity: Int8SourceMapCapacity,
    output_tokens: usize,
    reserved_work: Int8Work,
    reserved_masks: u64,
}
impl Int8SummaryHierarchyPreflight {
    pub fn chunk_count(&self) -> usize { self.discovery.chunk_count() }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.discovery.source_span() }
    pub fn reserved_model_work(&self) -> Int8Work { self.reserved_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.reserved_masks }
    pub fn limits(&self) -> SummaryHierarchyLimits { self.limits }
    pub fn verify_completed(&self, run: &Int8SummaryHierarchyRun) -> Result<(), Int8SourceMapError> {
        receipt::verify(self, run)
    }
}

pub struct PreparedInt8SummaryHierarchy<'s> {
    base: PreparedInt8SummarySynthesis<'s>,
    expected: Int8SummaryHierarchyPreflight,
    mapping: Int8SourceMapLimits,
}
impl SourceTaskPlanner {
    /// Reserve MAXIMUM-CONTEXT work for every permitted reduction call before
    /// compiling the actual map. No fabricated document or inference receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int8_summary_hierarchy_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: SourceSummarySynthesis, hierarchy: SummaryHierarchyLimits, budget: TaskBudget,
        context: &PlanContext<'_>, planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<PreparedInt8SummaryHierarchy<'s>, Int8SourceMapError> {
        checkpoint(control)?;
        hierarchy.validate()?;
        request.validate(planning)?;
        let capacity = self.int8_map_capacity_with_control(&SourceMapTask::Summarize(request.synthesis_options),
            budget, context, planning, control)?;
        let mut reduced = mapping;
        for _ in 1..hierarchy.max_passes {
            checkpoint(control)?;
            reduced = reserve(budget, planning, reduced)?.0;
        }
        // The existing planner reserves the remaining one pass, constrains the
        // actual map scaffold and compiles the SAME lossless source partition.
        let base = self.plan_int8_summary_synthesis_with_control(source, request, budget, context, planning, reduced, control)?;
        let discovery = base.preflight_metadata();
        let mut reserved_work = discovery.discovery_work;
        for _ in 0..hierarchy.max_passes {
            reserved_work = add_work(reserved_work, discovery.synthesis_reserve).ok_or(Int8SourceMapError::WorkLimit)?;
        }
        let reserved_masks = discovery.synthesis_masks.checked_mul(hierarchy.max_passes as u64)
            .and_then(|n| n.checked_add(discovery.discovery_masks)).ok_or(Int8SourceMapError::WorkLimit)?;
        if !within(reserved_work, mapping.max_model_work) || reserved_masks > mapping.max_mask_visits {
            return Err(Int8SourceMapError::WorkLimit);
        }
        let expected = Int8SummaryHierarchyPreflight { discovery, limits: hierarchy, capacity,
            output_tokens: budget.max_output_tokens as usize, reserved_work, reserved_masks };
        checkpoint(control)?;
        Ok(PreparedInt8SummaryHierarchy { base, expected, mapping })
    }
}
impl PreparedInt8SummaryHierarchy<'_> {
    pub fn preflight_metadata(&self) -> Int8SummaryHierarchyPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.base.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.base.max_result_bytes() }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8SummaryHierarchyRun, Int8SourceMapError> {
        checkpoint(control)?;
        let Self { base, expected, mapping } = self;
        if planner.tokenizer_digest() != base.identity.tokenizer_digest || *planner.template_digest() != base.identity.template_digest {
            return Err(Int8SourceMapError::Admission);
        }
        base.map.preflight(admitted, engine)?;
        let PreparedInt8SummarySynthesis { source, map, identity, budget, planning, request, .. } = base;
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        let mut remaining = request.limits.verification;
        let collection = evidence::collect(source, discovery.mapped.root().value(), request.map_options,
            request.limits, &mut remaining, control)?;
        let initial = EvidenceSize::of(&collection);
        let mut driver = execution::Native { planner, identity: &identity, budget, planning,
            options: request.synthesis_options, mapping, engine, vocabulary };
        let reduced = execution::reduce(source, collection, request, expected, mapping.reduction.max_result_bytes as u64,
            &mut remaining, &mut driver, control)?;
        checkpoint(control)?;
        let total = request.limits.verification;
        let run = Int8SummaryHierarchyRun { schema_version: 1, execution: INT8_SUMMARY_HIERARCHY_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, semantics: SEMANTICS, limits: expected.limits,
            citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
            untrusted_fields: ["passes", "discovery.mapped.root.value"], initial_evidence: initial,
            status: reduced.status, final_pass: reduced.final_pass,
            reserved_model_work: expected.reserved_work,
            planned_model_work: add_work(discovery.planned_model_work, reduced.planned).ok_or(Int8SourceMapError::WorkLimit)?,
            model_work: add_work(discovery.model_work, reduced.actual).ok_or(Int8SourceMapError::WorkLimit)?,
            reserved_mask_node_visits: expected.reserved_masks,
            mask_node_visit_charge: discovery.mask_node_visit_charge.checked_add(reduced.masks).ok_or(Int8SourceMapError::WorkLimit)?,
            verification_used: SummaryVerificationWork {
                fields: total.max_fields.checked_sub(remaining.max_fields).ok_or_else(invalid)?,
                matches: total.max_matches.checked_sub(remaining.max_matches).ok_or_else(invalid)?,
                scan_steps: total.max_scan_steps.checked_sub(remaining.max_scan_steps).ok_or_else(invalid)? },
            tokenizer_work: reduced.tokenizer, levels: reduced.levels, passes: reduced.passes, discovery };
        expected.verify_completed(&run)?;
        extract_int8::check_size(&run, mapping.reduction.max_result_bytes as u64).map_err(Int8SourceError::from)?;
        checkpoint(control)?;
        Ok(run)
    }
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
    pub planned_model_work: Int8Work,
    /// Unchanged receipt with coordinates LOCAL to this group's evidence text.
    pub native: Int8SourceTaskRun,
    /// All generated bullets with verified ORIGINAL-document coordinates.
    pub bullets: Vec<CitedBullet>,
}
#[derive(Serialize)]
pub struct SummaryHierarchyLevel {
    pub input: EvidenceSize,
    pub passes: Range<usize>,
    /// None means a single final synthesis pass; Some means quote selection.
    pub next_evidence: Option<EvidenceSize>,
}
#[derive(Serialize)]
pub struct Int8SummaryHierarchyRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub semantics: &'static str,
    pub limits: SummaryHierarchyLimits,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 2],
    pub initial_evidence: EvidenceSize,
    pub status: SummarySynthesisStatus,
    /// Index of the final single-group pass. No duplicate native/text payload.
    /// None for no initial evidence or a completely exhausted quote frontier.
    pub final_pass: Option<usize>,
    pub reserved_model_work: Int8Work,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SummaryVerificationWork,
    pub tokenizer_work: HierarchyTokenizerWork,
    pub levels: Vec<SummaryHierarchyLevel>,
    pub passes: Vec<SummaryHierarchyPass>,
    pub discovery: Int8SourceMapRun,
}
impl Int8SummaryHierarchyRun {
    pub fn bullets(&self) -> &[CitedBullet] {
        self.final_pass.and_then(|i| self.passes.get(i)).map_or(&[], |pass| pass.bullets.as_slice())
    }
}
#[cfg(test)] mod tests;
