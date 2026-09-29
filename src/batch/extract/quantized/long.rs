//! Exact-schema extraction over a complete original document on one INT8 engine.
//!
//! Reuse the pinned extraction compiler and the common lossless chunk/reduce
//! coordinator. Every real schema/source plan and all five work axes are funded
//! before the first forward. Independent JSON values are NOT merged into a
//! fabricated global object; required fields are not replaced with guessed nulls.
use super::*;
use std::sync::Arc;
use crate::{
    tasks::{extract::{ExtractionGrounding, quantized::check_size},
        mapreduce::{self, ChunkPlan, MapReduceError, MapReduceResult, MapReduceTask, MapOutput,
            ReduceInput, ReductionPolicy, SourceChunk},
        source_planning::quantized::{Int8SourceError, long::{Int8SourceMapError, Int8SourceMapLimits}}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan,
        SourceOccurrence, FieldGroundingError, scan_occurrences},
};
mod execution;

pub const INT8_EXTRACTION_MAP_EXECUTION: &str = "portable-int8-exact-schema-document-map-v1";
const SEMANTICS: &str = "independent-chunk-json-no-global-object-merge-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int8ExtractionMapLimits {
    pub mapping: Int8SourceMapLimits,
    /// Additional nonrenewable original-coordinate verification across ALL
    /// chunks. Native schema/source validation retains its own per-pass caps.
    pub verification: GroundingBudget,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8ExtractionMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    work: Int8Work,
    masks: u64,
    grounding: ExtractionGrounding,
    verification: GroundingBudget,
}
impl Int8ExtractionMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn verify_completed(&self, run: &Int8ExtractionMapRun) -> Result<(), Int8SourceMapError> {
        let root = run.mapped.root();
        let (work, masks, fields, spans) = execution::statistics(root.value())?;
        if run.schema_version != 1 || run.execution != INT8_EXTRACTION_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.semantics != SEMANTICS
            || run.grounding != self.grounding || run.untrusted_fields != ["mapped.root.value"]
            || root.source_span() != self.source_span || root.chunk_range() != (0..self.chunks)
            || root.value().chunks.len() != self.chunks || run.planned_model_work != self.work
            || run.reserved_mask_node_visits != self.masks || run.model_work != work
            || run.mask_node_visit_charge != masks || !fits(work, self.work) || masks > self.masks
            || run.verification_used.fields != fields || run.verification_used.matches != spans
            || fields > self.verification.max_fields || spans > self.verification.max_matches
            || run.verification_used.scan_steps > self.verification.max_scan_steps {
            return Err(Int8SourceError::InvalidResult.into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
pub struct MappedExtractionChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Unchanged native result, including exact JSON STRING and local evidence.
    pub native: Int8ExtractRun,
    /// Same schema field pointers, with ALL occurrences in original coordinates.
    /// Structural-only extraction has no fabricated source evidence.
    pub original_fields: Vec<SourceFieldEvidence>,
}
pub struct ExtractionMapValue { chunks: Vec<Arc<MappedExtractionChunk>> }
impl ExtractionMapValue {
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = &MappedExtractionChunk> { self.chunks.iter().map(Arc::as_ref) }
}
impl Serialize for ExtractionMapValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.chunks())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExtractionMapVerification { pub fields: usize, pub matches: usize, pub scan_steps: u64 }
#[derive(Serialize)]
pub struct Int8ExtractionMapRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub semantics: &'static str,
    pub grounding: ExtractionGrounding,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: ExtractionMapVerification,
    pub untrusted_fields: [&'static str; 1],
    pub mapped: MapReduceResult<ExtractionMapValue>,
}

/// Source-borrowing, consumed commitment. No wire constructor or external
/// native-result input can replace the exact prepared schema/source identities.
pub struct PreparedInt8ExtractionMap<'s> {
    chunks: ChunkPlan<'s>,
    plans: Vec<PreparedInt8BatchExtraction>,
    limits: Int8ExtractionMapLimits,
    expected: Int8ExtractionMapPreflight,
}
impl PreparedInt8ExtractionMap<'_> {
    pub fn preflight_metadata(&self) -> Int8ExtractionMapPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.plans.iter().map(PreparedInt8BatchExtraction::execution_identity)
    }
    pub fn max_result_bytes(&self) -> u64 { self.limits.mapping.reduction.max_result_bytes as u64 }
}
impl Int8ExtractionBatchPlanner {
    /// Compile every actual chunk, not a placeholder document. The caller must
    /// retain a preparation guard for this complete set; CLI preflight drops it
    /// before loading weights and the charged host builds the same set again.
    pub fn plan_document_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        args: &ExtractionBatchArgs, limits: Int8ExtractionMapLimits, control: &mut C)
        -> Result<PreparedInt8ExtractionMap<'s>, Int8SourceMapError> {
        step(control)?;
        if source.is_empty() { return Err(Int8SourceMapError::EmptySource); }
        let b = args.budget; let c = &self.compiler; let m = limits.mapping; let v = limits.verification;
        b.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        if b.max_input_tokens > c.ceiling.max_input_tokens || b.max_output_tokens > c.ceiling.max_output_tokens
            || b.max_output_bytes > c.ceiling.max_output_bytes || b.max_grammar_states > c.ceiling.max_grammar_states
            || b.max_kv_bytes > c.ceiling.max_kv_bytes || args.schema.len() > c.compiler_limits.max_schema_bytes
            || m.chunks.max_chunks > 256 || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
            || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
            || m.reduction.max_result_bytes == 0 || !(1..=1_000_000).contains(&v.max_fields)
            || !(1..=1_000_000).contains(&v.max_matches) || v.max_scan_steps == 0 {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        // Use the factory's exact pinned fragments and source encoder. Schema
        // remains exact declarative bytes, including decimal constants and keys;
        // neither it nor source text passes through trusted template rendering.
        let scaffold = c.fragments.iter().try_fold(0_usize, |n, f| n.checked_add(f.len()))
            .ok_or(Int8SourceMapError::WorkLimit)?;
        let schema = c.encoder.encode(&args.schema, c.compiler_limits.max_schema_bytes, b.max_input_tokens as usize)
            .map_err(|_| Int8SourceMapError::InvalidLimits)?;
        let non_source = scaffold.checked_add(schema.total_token_count()).ok_or(Int8SourceMapError::WorkLimit)?;
        let token_room = (b.max_input_tokens as usize).checked_sub(non_source).filter(|&n| n > 0)
            .ok_or(Int8SourceMapError::InvalidLimits)?;
        let byte_room = (b.max_input_tokens as usize).checked_sub(scaffold)
            .and_then(|n| n.checked_sub(args.schema.len())).filter(|&n| n > 0).ok_or(Int8SourceMapError::InvalidLimits)?;
        let mut chunk_limits = m.chunks;
        chunk_limits.effective_token_limit().map_err(Int8SourceMapError::Chunk)?;
        chunk_limits.reserved_tokens = chunk_limits.reserved_tokens.max(non_source.checked_add(b.max_output_tokens as usize)
            .ok_or(Int8SourceMapError::WorkLimit)?);
        chunk_limits.max_chunk_tokens = chunk_limits.max_chunk_tokens.min(token_room);
        chunk_limits.max_chunk_bytes = chunk_limits.max_chunk_bytes.min(byte_room.max(4));
        let mut cancelled = None;
        let chunks = ChunkPlan::build_with_checkpoints(source, chunk_limits, |text| {
            c.encoder.encode(text, chunk_limits.max_chunk_bytes, chunk_limits.max_chunk_bytes)
                .map(|d| d.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
        }, || match control.prefill_checkpoint(0) {
            Some(cause) => { cancelled = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
        });
        if let Some(cause) = cancelled { return Err(Int8SourceError::Cancelled(cause).into()); }
        let chunks = chunks.map_err(Int8SourceMapError::Chunk)?;
        let first = chunks.chunks().first().ok_or(Int8SourceMapError::EmptySource)?.span();
        let last = chunks.chunks().last().ok_or(Int8SourceMapError::EmptySource)?.span();
        let mut expected = Int8ExtractionMapPreflight {
            source_span: VerifiedSourceSpan { byte_start: first.byte_start, byte_end: last.byte_end,
                scalar_start: first.scalar_start, scalar_end: last.scalar_end }, chunks: chunks.chunks().len(),
            work: Int8Work::default(), masks: 0, verification: v,
            grounding: match args.grounding { ExtractionBatchGrounding::Structural => ExtractionGrounding::NotRequested,
                ExtractionBatchGrounding::SourceMembership => ExtractionGrounding::SourceMembership } };
        let mut plans = Vec::new();
        plans.try_reserve_exact(expected.chunks).map_err(|_| Int8SourceMapError::Allocation)?;
        for chunk in chunks.chunks() {
            step(control)?;
            let plan = self.prepare_with_control(BatchDocument { id: format!("chunk_{}", chunk.id()), text: copy(chunk.text())?,
                task_args: Some(ExtractionBatchArgs { schema: copy(&args.schema)?, grounding: args.grounding, budget: b }) }, control)
                .map_err(batch_error)?;
            if non_source.checked_add(chunk.tokens()) != Some(plan.plan.prompt_tokens())
                || plan.plan.prompt_tokens().checked_add(b.max_output_tokens as usize).is_none_or(|n| n > chunk_limits.context_tokens)
                || plan.source().text() != chunk.text() {
                return Err(Int8SourceMapError::Admission);
            }
            expected.work = add_work(expected.work, plan.planned_work()).filter(|w| fits(*w, m.max_model_work))
                .ok_or(Int8SourceMapError::WorkLimit)?;
            expected.masks = expected.masks.checked_add(m.mask_visits_per_chunk).filter(|&n| n <= m.max_mask_visits)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            plans.push(plan);
        }
        step(control)?;
        Ok(PreparedInt8ExtractionMap { chunks, plans, limits, expected })
    }
}
fn step<C: DecodeStepControl>(control: &mut C) -> Result<(), Int8SourceError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int8SourceError::Cancelled(cause)), None => Ok(()) }
}
fn batch_error(error: BatchItemFailure) -> Int8SourceMapError {
    if let Some(cause) = error.fault.cancellation { return Int8SourceError::Cancelled(cause).into(); }
    match error.fault.code { BatchCode::Allocation => Int8SourceMapError::Allocation,
        BatchCode::WorkLimit => Int8SourceMapError::WorkLimit, _ => Int8SourceMapError::Admission }
}
fn copy(text: &str) -> Result<String, Int8SourceMapError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| Int8SourceMapError::Allocation)?;
    out.push_str(text); Ok(out)
}

#[cfg(test)] mod tests;
