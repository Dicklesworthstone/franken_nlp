//! Opt-in neural document summary from verified verbatim map evidence only.
//! Generated map bullets are retained, never promoted to source facts. Every
//! quote must fit the final context; this is not full-document equivalence.
use super::*;
use crate::{
    native_engine::portable_int8::ProjectionWork,
    tasks::{ir::ScoreSpace, summarize::{CitedBullet, CitationGuarantee, SourceCitation, SummarySemanticSupport}},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences, FieldGroundingError},
};
mod evidence;
/// Separate opt-in hierarchical quote compression; the single-pass API is unchanged.
pub mod hierarchy;

pub const INT8_SUMMARY_SYNTHESIS_EXECUTION: &str = "portable-int8-verbatim-evidence-summary-synthesis-v1";
const SEMANTICS: &str = "summary-from-collected-quotes-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SummarySynthesisLimits {
    /// Distinct (source chunk, quote) segments; no ranking or silent truncation.
    pub max_evidence_segments: usize,
    /// Complete evidence document INCLUDING code-owned two-newline separators.
    pub max_evidence_bytes: usize,
    /// One nonrenewable independent verification ledger for collection and lift.
    pub verification: GroundingBudget,
}
impl SummarySynthesisLimits {
    pub fn validate(self, planning: SourcePlanningLimits) -> Result<(), Int8SourceMapError> {
        let v = self.verification;
        if !(1..=1024).contains(&self.max_evidence_segments)
            || self.max_evidence_bytes == 0 || self.max_evidence_bytes > planning.max_input_bytes
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || !(1..=1_000_000_000_000).contains(&v.max_scan_steps) {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SourceSummarySynthesis {
    pub map_options: SummaryOptions,
    pub synthesis_options: SummaryOptions,
    pub limits: SummarySynthesisLimits,
}
impl SourceSummarySynthesis {
    pub fn validate(self, planning: SourcePlanningLimits) -> Result<(), Int8SourceMapError> {
        self.map_options.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        self.synthesis_options.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        self.limits.validate(planning)
    }
}

/// Content-free accounting commitment. This is not a native execution receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SummarySynthesisPreflight {
    chunks: usize,
    source_span: VerifiedSourceSpan,
    discovery_work: Int8Work,
    discovery_masks: u64,
    synthesis_reserve: Int8Work,
    synthesis_masks: u64,
    limits: SummarySynthesisLimits,
    options: SummaryOptions,
}
impl Int8SummarySynthesisPreflight {
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn synthesis_reserved_work(&self) -> Int8Work { self.synthesis_reserve }
    pub fn reserved_model_work(&self) -> Result<Int8Work, Int8SourceMapError> {
        add_work(self.discovery_work, self.synthesis_reserve).ok_or(Int8SourceMapError::WorkLimit)
    }
    pub fn reserved_mask_visits(&self) -> Result<u64, Int8SourceMapError> {
        self.discovery_masks.checked_add(self.synthesis_masks).ok_or(Int8SourceMapError::WorkLimit)
    }
    pub fn verify_completed(&self, run: &Int8SummarySynthesisRun) -> Result<(), Int8SourceMapError> {
        let map = &run.discovery; let root = map.mapped.root();
        let actual = add_work(map.model_work, run.synthesis_model_work).ok_or(Int8SourceMapError::WorkLimit)?;
        let masks = map.mask_node_visit_charge.checked_add(run.synthesis_mask_node_visits)
            .ok_or(Int8SourceMapError::WorkLimit)?;
        if run.schema_version != 1 || run.execution != INT8_SUMMARY_SYNTHESIS_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.semantics != SEMANTICS
            || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || run.semantic_support != SummarySemanticSupport::NotAssessed
            || run.untrusted_fields != ["bullets", "synthesis_native", "discovery.mapped.root.value"]
            || map.schema_version != 1 || map.execution != INT8_SOURCE_MAP_EXECUTION
            || map.semantics != "independent-chunks-no-cross-chunk-reasoning-v1"
            || root.chunk_range() != (0..self.chunks) || root.value().len() != self.chunks
            || root.source_span() != self.source_span || map.planned_model_work != self.discovery_work
            || map.reserved_mask_node_visits != self.discovery_masks
            || !within(map.model_work, self.discovery_work) || map.mask_node_visit_charge > self.discovery_masks
            || run.synthesis_reserved_work != self.synthesis_reserve
            || run.reserved_model_work != self.reserved_model_work()? || run.model_work != actual
            || !within(run.synthesis_planned_work, self.synthesis_reserve)
            || !within(run.synthesis_model_work, run.synthesis_planned_work)
            || run.reserved_mask_node_visits != self.reserved_mask_visits()? || run.mask_node_visit_charge != masks
            || run.synthesis_mask_node_visits > self.synthesis_masks
            || run.evidence_segments > self.limits.max_evidence_segments || run.evidence_bytes > self.limits.max_evidence_bytes
            || run.verification_used.fields > self.limits.verification.max_fields
            || run.verification_used.matches > self.limits.verification.max_matches
            || run.verification_used.scan_steps > self.limits.verification.max_scan_steps {
            return Err(invalid());
        }
        match &run.synthesis_native {
            None => {
                if run.status != SummarySynthesisStatus::NoEvidenceCollected || run.evidence_segments != 0
                    || run.evidence_bytes != 0 || !run.bullets.is_empty()
                    || run.synthesis_planned_work != Int8Work::default() || run.synthesis_model_work != Int8Work::default()
                    || run.synthesis_mask_node_visits != 0 || map.mapped.root().value().chunks().any(|c|
                        !matches!(&c.native.result, SourceTaskResult::Summarize(r) if r.bullets.is_empty())) {
                    return Err(invalid());
                }
            }
            Some(native) => {
                let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
                evidence::check_summary(raw, self.options)?;
                if native.schema_version != 1 || native.execution != INT8_SOURCE_EXECUTION
                    || native.model_work != run.synthesis_model_work
                    || raw.forward_positions != native.model_work.forward_positions
                    || raw.projected_logits != native.model_work.projected_logits
                    || raw.mask_node_visit_charge != run.synthesis_mask_node_visits
                    || raw.generated_token_ids.is_empty() || run.evidence_segments == 0 || run.evidence_bytes == 0
                    || native.model_work.forward_positions == 0 || raw.bullets.len() != run.bullets.len()
                    || run.status != if raw.bullets.is_empty() { SummarySynthesisStatus::NoBulletsProduced }
                        else { SummarySynthesisStatus::Synthesized } { return Err(invalid()); }
                for (raw, lifted) in raw.bullets.iter().zip(&run.bullets) {
                    if raw.text != lifted.text || raw.citations.len() != lifted.citations.len() { return Err(invalid()); }
                    for (local, original) in raw.citations.iter().zip(&lifted.citations) {
                        if local.quote != original.quote { return Err(invalid()); }
                        evidence::check_citation(original, self.options.max_quote_scalars)?;
                        if original.spans.iter().any(|s| s.byte_end > self.source_span.byte_end
                            || s.scalar_end > self.source_span.scalar_end) { return Err(invalid()); }
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct PreparedInt8SummarySynthesis<'s> {
    source: &'s str,
    map: PreparedInt8SourceMap<'s>,
    identity: ExecutionIdentity,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int8SourceMapLimits,
    request: SourceSummarySynthesis,
    expected: Int8SummarySynthesisPreflight,
}
impl SourceTaskPlanner {
    /// Compile every real map prompt after reserving a maximum-context final
    /// pass on ALL five work axes and the same whole-invocation mask allowance.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int8_summary_synthesis_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: SourceSummarySynthesis, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<PreparedInt8SummarySynthesis<'s>, Int8SourceMapError> {
        checkpoint(control)?;
        request.validate(planning)?;
        let (discovery_limits, reserve) = reserve(budget, planning, mapping)?;
        // Derive the real summary scaffold, rather than relying on a caller's
        // guessed reserve. The shared chunk planner remains the only partitioner.
        let task = SourceMapTask::Summarize(request.map_options);
        let capacity = self.int8_map_capacity_with_control(&task, budget, context, planning, control)?;
        let mut discovery_limits = discovery_limits;
        discovery_limits.chunks = capacity.constrain_chunks(discovery_limits.chunks)?;
        let map = self.plan_int8_map_with_control(source, &task, budget, context, planning, discovery_limits, control)?;
        let last = map.chunks.chunks().last().ok_or(Int8SourceMapError::EmptySource)?.span();
        let expected = Int8SummarySynthesisPreflight {
            chunks: map.chunk_count(), source_span: VerifiedSourceSpan { byte_start: 0, byte_end: source.len(),
                scalar_start: 0, scalar_end: last.scalar_end }, discovery_work: map.work, discovery_masks: map.masks,
            synthesis_reserve: reserve, synthesis_masks: mapping.mask_visits_per_chunk,
            limits: request.limits, options: request.synthesis_options,
        };
        if !within(expected.reserved_model_work()?, mapping.max_model_work)
            || expected.reserved_mask_visits()? > mapping.max_mask_visits { return Err(Int8SourceMapError::WorkLimit); }
        checkpoint(control)?;
        Ok(PreparedInt8SummarySynthesis { source, map, identity: context.execution_identity().clone(),
            budget, planning, mapping, request, expected })
    }
}
fn reserve(budget: TaskBudget, planning: SourcePlanningLimits, mut mapping: Int8SourceMapLimits)
    -> Result<(Int8SourceMapLimits, Int8Work), Int8SourceMapError> {
    mapping.validate(budget, planning)?;
    let output = budget.max_output_tokens as usize;
    let prompt = planning.max_context_tokens.checked_sub(output).ok_or(Int8SourceMapError::InvalidLimits)?
        .min(budget.max_input_tokens as usize);
    if prompt == 0 { return Err(Int8SourceMapError::InvalidLimits); }
    let work = constrained_int8::planned_work(prompt, output).map_err(|_| Int8SourceMapError::WorkLimit)?;
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int8SourceMapError::WorkLimit);
    let w = mapping.max_model_work;
    mapping.max_model_work = Int8Work { forward_positions: sub(w.forward_positions, work.forward_positions)?,
        projected_logits: sub(w.projected_logits, work.projected_logits)?,
        attention_pairs: sub(w.attention_pairs, work.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(w.projections.dot_products, work.projections.dot_products)?,
            multiply_accumulates: sub(w.projections.multiply_accumulates, work.projections.multiply_accumulates)? } };
    mapping.max_mask_visits = sub(mapping.max_mask_visits, mapping.mask_visits_per_chunk)?;
    Ok((mapping, work))
}
impl PreparedInt8SummarySynthesis<'_> {
    pub fn preflight_metadata(&self) -> Int8SummarySynthesisPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.map.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8SummarySynthesisRun, Int8SourceMapError> {
        checkpoint(control)?;
        if planner.tokenizer_digest() != self.identity.tokenizer_digest || *planner.template_digest() != self.identity.template_digest {
            return Err(Int8SourceMapError::Admission);
        }
        self.map.preflight(admitted, engine)?;
        let Self { source, map, identity, budget, planning, mapping, request, expected } = self;
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        let mut remaining = request.limits.verification;
        let collection = evidence::collect(source, discovery.mapped.root().value(), request.map_options,
            request.limits, &mut remaining, control)?;
        let count = collection.segment_count(); let bytes = collection.text.len();
        let (native, planned, bullets) = if count == 0 { (None, Int8Work::default(), Vec::new()) } else {
            let context = PlanContext::new(&identity, budget).map_err(|_| Int8SourceMapError::Admission)?;
            let final_request = SourceTaskRequest::Summarize { document: copy(&collection.text)?,
                options: request.synthesis_options, budget };
            let plan = planner.plan_int8_with_control(&final_request, &context, planning, control)?;
            let planned = plan.planned_work();
            if !within(planned, expected.synthesis_reserve) { return Err(Int8SourceMapError::WorkLimit); }
            // Dynamic source/prompt/schema identity is admitted against the same
            // real model and full resident KV before any synthesis forward.
            preflight_plans(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), engine)?;
            let mut driver = NativeDriver { engine, vocabulary, control, limits: mapping };
            let native = driver.run(&plan, plan.execution_identity())?;
            check_run(&plan, &native, mapping.mask_visits_per_chunk)?;
            extract_int8::check_size(&native, plan.max_result_bytes()).map_err(Int8SourceError::from)?;
            let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
            let bullets = evidence::lift_summary(source, raw, &collection, request.synthesis_options, &mut remaining, control)?;
            (Some(native), planned, bullets)
        };
        drop(collection);
        checkpoint(control)?;
        let work = native.as_ref().map_or(Int8Work::default(), |n| n.model_work);
        let masks = native.as_ref().map(|n| mask_charge(&n.result)).transpose()?.unwrap_or(0);
        let status = if native.is_none() { SummarySynthesisStatus::NoEvidenceCollected }
            else if bullets.is_empty() { SummarySynthesisStatus::NoBulletsProduced } else { SummarySynthesisStatus::Synthesized };
        let total = request.limits.verification;
        let run = Int8SummarySynthesisRun { schema_version: 1, execution: INT8_SUMMARY_SYNTHESIS_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, semantics: SEMANTICS, status, evidence_segments: count, evidence_bytes: bytes,
            synthesis_reserved_work: expected.synthesis_reserve, synthesis_planned_work: planned, synthesis_model_work: work,
            reserved_model_work: expected.reserved_model_work()?,
            model_work: add_work(discovery.model_work, work).ok_or(Int8SourceMapError::WorkLimit)?,
            reserved_mask_node_visits: expected.reserved_mask_visits()?, synthesis_mask_node_visits: masks,
            mask_node_visit_charge: discovery.mask_node_visit_charge.checked_add(masks).ok_or(Int8SourceMapError::WorkLimit)?,
            verification_used: SummaryVerificationWork {
                fields: total.max_fields.checked_sub(remaining.max_fields).ok_or_else(invalid)?,
                matches: total.max_matches.checked_sub(remaining.max_matches).ok_or_else(invalid)?,
                scan_steps: total.max_scan_steps.checked_sub(remaining.max_scan_steps).ok_or_else(invalid)? },
            citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
            untrusted_fields: ["bullets", "synthesis_native", "discovery.mapped.root.value"],
            bullets, synthesis_native: native, discovery };
        expected.verify_completed(&run)?;
        extract_int8::check_size(&run, mapping.reduction.max_result_bytes as u64).map_err(Int8SourceError::from)?;
        checkpoint(control)?;
        Ok(run)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SummarySynthesisStatus { NoEvidenceCollected, NoBulletsProduced, Synthesized }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SummaryVerificationWork { pub fields: usize, pub matches: usize, pub scan_steps: u64 }
#[derive(Serialize)]
pub struct Int8SummarySynthesisRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub semantics: &'static str,
    pub status: SummarySynthesisStatus,
    pub evidence_segments: usize,
    pub evidence_bytes: usize,
    pub synthesis_reserved_work: Int8Work,
    pub synthesis_planned_work: Int8Work,
    pub synthesis_model_work: Int8Work,
    pub reserved_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub synthesis_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SummaryVerificationWork,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 3],
    /// Synthesized text with evidence-reachable ORIGINAL-document coordinates.
    pub bullets: Vec<CitedBullet>,
    /// Unchanged native receipt; its citations are EVIDENCE-document-local.
    pub synthesis_native: Option<Int8SourceTaskRun>,
    /// All map judgments remain visible, including empty summaries.
    pub discovery: Int8SourceMapRun,
}
fn invalid() -> Int8SourceMapError { Int8SourceError::InvalidResult.into() }
fn vector<T>(n: usize) -> Result<Vec<T>, Int8SourceMapError> {
    let mut v = Vec::new(); v.try_reserve_exact(n).map_err(|_| Int8SourceMapError::Allocation)?; Ok(v)
}
fn copy(s: &str) -> Result<String, Int8SourceMapError> {
    let mut out = String::new(); out.try_reserve_exact(s.len()).map_err(|_| Int8SourceMapError::Allocation)?;
    out.push_str(s); Ok(out)
}
#[cfg(test)] mod tests;
