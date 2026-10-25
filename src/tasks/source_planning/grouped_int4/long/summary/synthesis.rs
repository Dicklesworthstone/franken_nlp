//! Native INT4 summary synthesis from verified verbatim map evidence only.
//! Generated map bullets remain inspectable output, never factual input. The
//! complete evidence set must fit one final context; overflow returns no result.
use super::*;
use crate::{
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::ProjectionWork},
    tasks::summarize::{CitedBullet, SourceCitation},
};
mod evidence;
/// Explicit lossy quote selection across several bounded synthesis levels.
pub mod hierarchy;

pub const INT4_SUMMARY_SYNTHESIS_EXECUTION: &str = "grouped-int4-verbatim-evidence-summary-synthesis-v1";
const SEMANTICS: &str = "summary-from-collected-quotes-not-full-document-equivalence-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SummarySynthesisLimits {
    /// Distinct (source chunk, quote) segments; no ranking or silent truncation.
    pub max_evidence_segments: usize,
    /// Evidence bytes INCLUDING the code-owned two-newline separators.
    pub max_evidence_bytes: usize,
    /// One ceiling for map verification, collection, and final citation lifting.
    pub verification: GroundingBudget,
}
impl SummarySynthesisLimits {
    pub fn validate(self, planning: SourcePlanningLimits) -> Result<(), Int4SourceMapError> {
        let v = self.verification;
        if !(1..=1024).contains(&self.max_evidence_segments)
            || self.max_evidence_bytes == 0 || self.max_evidence_bytes > planning.max_input_bytes
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || !(1..=1_000_000_000_000).contains(&v.max_scan_steps) {
            return Err(Int4SourceMapError::InvalidLimits);
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
    pub fn validate(self, planning: SourcePlanningLimits) -> Result<(), Int4SourceMapError> {
        self.map_options.validate().map_err(|_| Int4SourceMapError::InvalidLimits)?;
        self.synthesis_options.validate().map_err(|_| Int4SourceMapError::InvalidLimits)?;
        self.limits.validate(planning)
    }
}

/// A private, content-free planning commitment, never a native model receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SummarySynthesisPreflight {
    discovery: Int4SourceMapPreflight,
    synthesis_reserve: Int4Work,
    synthesis_masks: u64,
    limits: SummarySynthesisLimits,
    options: SummaryOptions,
    max_output_tokens: usize,
    eos_token_id: u32,
}
impl Int4SummarySynthesisPreflight {
    pub fn discovery(&self) -> Int4SourceMapPreflight { self.discovery }
    pub fn chunk_count(&self) -> usize { self.discovery.chunk_count() }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.discovery.source_span() }
    pub fn group_width(&self) -> usize { self.discovery.group_width() }
    pub fn synthesis_reserved_work(&self) -> Int4Work { self.synthesis_reserve }
    pub fn reserved_model_work(&self) -> Result<Int4Work, Int4SourceMapError> {
        add_work(self.discovery.planned_work(), self.synthesis_reserve).ok_or(Int4SourceMapError::WorkLimit)
    }
    pub fn reserved_mask_visits(&self) -> Result<u64, Int4SourceMapError> {
        self.discovery.reserved_mask_visits().checked_add(self.synthesis_masks).ok_or(Int4SourceMapError::WorkLimit)
    }
    pub fn verify_completed(&self, run: &Int4SummarySynthesisRun) -> Result<(), Int4SourceMapError> {
        self.discovery.verify_completed(&run.discovery)?;
        let map = &run.discovery;
        let actual = add_work(map.model_work, run.synthesis_model_work).ok_or(Int4SourceMapError::WorkLimit)?;
        let masks = map.mask_node_visit_charge.checked_add(run.synthesis_mask_node_visits)
            .ok_or(Int4SourceMapError::WorkLimit)?;
        if self.discovery.task_spec != SUMMARIZE_TASK_VERSION || run.schema_version != 1
            || run.execution != INT4_SUMMARY_SYNTHESIS_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
            || run.group_width != self.group_width() || run.semantics != SEMANTICS
            || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || run.semantic_support != SummarySemanticSupport::NotAssessed
            || run.untrusted_fields != ["bullets", "synthesis_native", "discovery.mapped.root.value"]
            || run.synthesis_reserved_work != self.synthesis_reserve
            || run.reserved_model_work != self.reserved_model_work()? || run.model_work != actual
            || !fits(run.synthesis_planned_work, self.synthesis_reserve)
            || !fits(run.synthesis_model_work, run.synthesis_planned_work)
            || run.reserved_mask_node_visits != self.reserved_mask_visits()? || run.mask_node_visit_charge != masks
            || run.synthesis_mask_node_visits > self.synthesis_masks
            || run.evidence_segments > self.limits.max_evidence_segments || run.evidence_bytes > self.limits.max_evidence_bytes
            || !run.verification_used.fits(self.limits.verification)
            || run.verification_used.fields < map.verification_used.fields
            || run.verification_used.matches < map.verification_used.matches
            || run.verification_used.scan_steps < map.verification_used.scan_steps {
            return Err(invalid());
        }
        match &run.synthesis_native {
            None => {
                if run.status != SummarySynthesisStatus::NoEvidenceCollected || run.evidence_segments != 0
                    || run.evidence_bytes != 0 || !run.bullets.is_empty() || run.synthesis_prompt_tokens != 0
                    || run.synthesis_planned_work != Int4Work::default() || run.synthesis_model_work != Int4Work::default()
                    || run.synthesis_mask_node_visits != 0 || map.mapped.root().value().chunks().any(|c|
                        !matches!(&c.native.result, SourceTaskResult::Summarize(r) if r.bullets.is_empty())) {
                    return Err(invalid());
                }
            }
            Some(native) => {
                let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
                evidence::check_summary(raw, self.options)?;
                let ids = &raw.generated_token_ids;
                if ids.is_empty() || ids.len() > self.max_output_tokens || ids.last() != Some(&self.eos_token_id)
                    || ids[..ids.len() - 1].iter().any(|id| *id == self.eos_token_id || *id as usize >= NANBEIGE_VOCAB_SIZE) {
                    return Err(invalid());
                }
                let planned = constrained_int4::planned_work(self.group_width(), run.synthesis_prompt_tokens, self.max_output_tokens)
                    .map_err(Int4SourceError::from)?;
                let work = constrained_int4::planned_work(self.group_width(), run.synthesis_prompt_tokens, ids.len())
                    .map_err(Int4SourceError::from)?;
                if native.schema_version != 1 || native.execution != INT4_SOURCE_EXECUTION
                    || native.group_width != self.group_width() || planned != run.synthesis_planned_work
                    || work != native.model_work || native.model_work != run.synthesis_model_work
                    || raw.forward_positions != native.model_work.forward_positions
                    || raw.projected_logits != native.model_work.projected_logits
                    || raw.mask_node_visit_charge != run.synthesis_mask_node_visits
                    || run.evidence_segments == 0 || run.evidence_bytes == 0 || raw.bullets.len() != run.bullets.len()
                    || run.status != if raw.bullets.is_empty() { SummarySynthesisStatus::NoBulletsProduced }
                        else { SummarySynthesisStatus::Synthesized } { return Err(invalid()); }
                let source = self.source_span();
                for (raw, lifted) in raw.bullets.iter().zip(&run.bullets) {
                    if raw.text != lifted.text || raw.citations.len() != lifted.citations.len() { return Err(invalid()); }
                    for (local, original) in raw.citations.iter().zip(&lifted.citations) {
                        if local.quote != original.quote { return Err(invalid()); }
                        evidence::check_citation(original, self.options.max_quote_scalars)?;
                        if original.spans.iter().any(|s| s.byte_end > source.byte_end || s.scalar_end > source.scalar_end) {
                            return Err(invalid());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct PreparedInt4SummarySynthesis<'s> {
    source: &'s str,
    map: PreparedInt4SourceMap<'s>,
    identity: ExecutionIdentity,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits,
    request: SourceSummarySynthesis,
    expected: Int4SummarySynthesisPreflight,
}
impl SourceTaskPlanner {
    /// Reserve the final full-context pass before compiling any discovery map.
    /// Work left unused by early EOS is never reassigned to the final pass.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int4_summary_synthesis_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: SourceSummarySynthesis, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, group_width: usize, control: &mut C)
        -> Result<PreparedInt4SummarySynthesis<'s>, Int4SourceMapError> {
        checkpoint(control)?;
        request.validate(planning)?;
        let (mut discovery_limits, synthesis_reserve) = reserve(budget, planning, mapping, group_width)?;
        discovery_limits.verification = meet_verification(discovery_limits.verification, request.limits.verification);
        let task = SourceMapTask::Summarize(request.map_options);
        let map = self.plan_int4_map_with_control(source, &task, budget, context, planning,
            discovery_limits, group_width, control)?;
        let expected = Int4SummarySynthesisPreflight { discovery: map.preflight_metadata(), synthesis_reserve,
            synthesis_masks: mapping.mask_visits_per_chunk, limits: request.limits,
            options: request.synthesis_options, max_output_tokens: budget.max_output_tokens as usize,
            eos_token_id: self.eos };
        if !fits(expected.reserved_model_work()?, mapping.max_model_work)
            || expected.reserved_mask_visits()? > mapping.max_mask_visits { return Err(Int4SourceMapError::WorkLimit); }
        checkpoint(control)?;
        Ok(PreparedInt4SummarySynthesis { source, map, identity: context.execution_identity().clone(),
            budget, planning, mapping, request, expected })
    }
}
fn meet_verification(a: GroundingBudget, b: GroundingBudget) -> GroundingBudget {
    GroundingBudget { max_fields: a.max_fields.min(b.max_fields), max_matches: a.max_matches.min(b.max_matches),
        max_scan_steps: a.max_scan_steps.min(b.max_scan_steps) }
}
fn reserve(budget: TaskBudget, planning: SourcePlanningLimits, mut mapping: Int4SourceMapLimits, group_width: usize)
    -> Result<(Int4SourceMapLimits, Int4Work), Int4SourceMapError> {
    mapping.validate(budget, planning)?;
    let output = budget.max_output_tokens as usize;
    let prompt = planning.max_context_tokens.checked_sub(output).ok_or(Int4SourceMapError::InvalidLimits)?
        .min(budget.max_input_tokens as usize);
    if prompt == 0 { return Err(Int4SourceMapError::InvalidLimits); }
    let work = constrained_int4::planned_work(group_width, prompt, output).map_err(Int4SourceError::from)?;
    mapping.max_model_work = subtract(mapping.max_model_work, work)?;
    mapping.max_mask_visits = mapping.max_mask_visits.checked_sub(mapping.mask_visits_per_chunk)
        .ok_or(Int4SourceMapError::WorkLimit)?;
    Ok((mapping, work))
}
fn subtract(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4SourceMapError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int4SourceMapError::WorkLimit);
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
impl PreparedInt4SummarySynthesis<'_> {
    pub fn preflight_metadata(&self) -> Int4SummarySynthesisPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.map.execution_identities() }
    pub fn max_result_bytes(&self) -> u64 { self.mapping.reduction.max_result_bytes as u64 }
    pub fn max_kv_bytes(&self) -> u64 { self.budget.max_kv_bytes }
    pub fn group_width(&self) -> usize { self.expected.group_width() }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4SummarySynthesisRun, Int4SourceMapError> {
        checkpoint(control)?;
        if planner.tokenizer_digest() != self.identity.tokenizer_digest || *planner.template_digest() != self.identity.template_digest {
            return Err(Int4SourceMapError::Admission);
        }
        self.map.preflight(admitted, engine)?;
        let Self { source, map, identity, budget, planning, mapping, request, expected } = self;
        let discovery = map.execute_with_control(admitted, engine, vocabulary, control)?;
        expected.discovery.verify_completed(&discovery)?;
        let mut remaining = remaining_verification(request.limits.verification, discovery.verification_used)?;
        let collection = evidence::collect(source, discovery.mapped.root().value(), request.map_options,
            request.limits, &mut remaining, control)?;
        let segments = collection.segment_count(); let bytes = collection.text.len();
        let (native, prompt_tokens, planned, bullets) = if segments == 0 { (None, 0, Int4Work::default(), Vec::new()) } else {
            let context = PlanContext::new(&identity, budget).map_err(|_| Int4SourceMapError::Admission)?;
            let final_request = SourceTaskRequest::Summarize { document: copy(&collection.text)?,
                options: request.synthesis_options, budget };
            let plan = planner.plan_int4_with_control(&final_request, &context, planning, expected.group_width(), control)?;
            let planned = plan.planned_work();
            if !fits(planned, expected.synthesis_reserve) { return Err(Int4SourceMapError::WorkLimit); }
            let mut driver = NativeDriver { engine, vocabulary, control, limits: mapping };
            // The dynamic evidence prompt, group partition and entire resident
            // KV are admitted against the same actual model before synthesis.
            preflight(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), &driver)?;
            let native = driver.run(&plan, plan.execution_identity())?;
            check_run(&plan, &native, mapping.mask_visits_per_chunk)?;
            let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
            let bullets = evidence::lift_summary(source, raw, &collection, request.synthesis_options, &mut remaining, control)?;
            (Some(native), plan.prompt_tokens(), planned, bullets)
        };
        drop(collection);
        finish(discovery, native, prompt_tokens, planned, bullets, segments, bytes, remaining, expected,
            mapping.reduction.max_result_bytes as u64, control)
    }
}
fn remaining_verification(total: GroundingBudget, used: SourceMapVerification) -> Result<GroundingBudget, Int4SourceMapError> {
    Ok(GroundingBudget { max_fields: total.max_fields.checked_sub(used.fields).ok_or(Int4SourceMapError::WorkLimit)?,
        max_matches: total.max_matches.checked_sub(used.matches).ok_or(Int4SourceMapError::WorkLimit)?,
        max_scan_steps: total.max_scan_steps.checked_sub(used.scan_steps).ok_or(Int4SourceMapError::WorkLimit)? })
}
#[allow(clippy::too_many_arguments)]
fn finish<C: DecodeStepControl>(discovery: Int4SourceMapRun, native: Option<Int4SourceTaskRun>, prompt_tokens: usize,
    planned: Int4Work, bullets: Vec<CitedBullet>, segments: usize, bytes: usize, remaining: GroundingBudget,
    expected: Int4SummarySynthesisPreflight, cap: u64, control: &mut C) -> Result<Int4SummarySynthesisRun, Int4SourceMapError> {
    checkpoint(control)?;
    let work = native.as_ref().map_or(Int4Work::default(), |n| n.model_work);
    let masks = native.as_ref().map_or(0, |n| mask_charge(&n.result));
    let status = if native.is_none() { SummarySynthesisStatus::NoEvidenceCollected }
        else if bullets.is_empty() { SummarySynthesisStatus::NoBulletsProduced } else { SummarySynthesisStatus::Synthesized };
    let run = Int4SummarySynthesisRun { schema_version: 1, execution: INT4_SUMMARY_SYNTHESIS_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: expected.group_width(), semantics: SEMANTICS, status,
        evidence_segments: segments, evidence_bytes: bytes, synthesis_prompt_tokens: prompt_tokens,
        synthesis_reserved_work: expected.synthesis_reserve, synthesis_planned_work: planned, synthesis_model_work: work,
        reserved_model_work: expected.reserved_model_work()?,
        model_work: add_work(discovery.model_work, work).ok_or(Int4SourceMapError::WorkLimit)?,
        reserved_mask_node_visits: expected.reserved_mask_visits()?, synthesis_mask_node_visits: masks,
        mask_node_visit_charge: discovery.mask_node_visit_charge.checked_add(masks).ok_or(Int4SourceMapError::WorkLimit)?,
        verification_used: SourceMapVerification::used(expected.limits.verification, remaining)?,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        untrusted_fields: ["bullets", "synthesis_native", "discovery.mapped.root.value"],
        bullets, synthesis_native: native, discovery };
    expected.verify_completed(&run)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SummarySynthesisStatus { NoEvidenceCollected, NoBulletsProduced, Synthesized }
#[derive(Serialize)]
pub struct Int4SummarySynthesisRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub semantics: &'static str,
    pub status: SummarySynthesisStatus,
    pub evidence_segments: usize,
    pub evidence_bytes: usize,
    pub synthesis_prompt_tokens: usize,
    pub synthesis_reserved_work: Int4Work,
    pub synthesis_planned_work: Int4Work,
    pub synthesis_model_work: Int4Work,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub synthesis_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SourceMapVerification,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub untrusted_fields: [&'static str; 3],
    /// Synthesized text with evidence-reachable ORIGINAL-document coordinates.
    pub bullets: Vec<CitedBullet>,
    /// Unchanged native result with EVIDENCE-document-local citations.
    pub synthesis_native: Option<Int4SourceTaskRun>,
    /// All map results, including empty summaries and conflicting bullets.
    pub discovery: Int4SourceMapRun,
}
fn invalid() -> Int4SourceMapError { Int4SourceError::InvalidResult.into() }
fn vector<T>(n: usize) -> Result<Vec<T>, Int4SourceMapError> {
    let mut v = Vec::new(); v.try_reserve_exact(n).map_err(|_| Int4SourceMapError::Allocation)?; Ok(v)
}
fn copy(text: &str) -> Result<String, Int4SourceMapError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| Int4SourceMapError::Allocation)?;
    out.push_str(text); Ok(out)
}

#[cfg(test)] mod tests;
