//! Exact-schema extraction over a complete document on one grouped-INT4 engine.
//!
//! Reuse the pinned extraction compiler and the common lossless chunk/reduce
//! coordinator. Every real schema/source plan and all five work axes are funded
//! before the first forward. Independent JSON values are NOT merged into a
//! fabricated global object; required fields are not replaced with guessed nulls.
use super::*;
use super::native::{Driver, NativeDriver, preflight, native_budget, valid_result};
use std::sync::Arc;
use crate::{
    native_engine::{constrained_int4, decode::DecodeCancellationKind,
        strict_int4::{StrictInt4Engine, STRICT_INT4_PROFILE}},
    tasks::{extract::{ExtractionGrounding, quantized::check_size},
        extract::grouped_int4::Int4ExtractRun,
        mapreduce::{self, ChunkLimits, ChunkPlan, ExecutionError, ExecutionLimits, MapReduceError,
            MapReduceResult, MapReduceTask, MapOutput, ReduceInput, ReductionPolicy, SourceChunk}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan,
        SourceOccurrence, FieldGroundingError, scan_occurrences},
};
#[cfg(test)] use crate::{native_engine::constrained_int4::Int4JsonBudget,
    tasks::extract::grouped_int4::INT4_EXTRACT_VERSION};
mod execution;

pub const INT4_EXTRACTION_MAP_EXECUTION: &str = "grouped-int4-exact-schema-document-map-v1";
const SEMANTICS: &str = "independent-chunk-json-no-global-object-merge-v1";

/// Whole-document allowances. Native work depends on the actual group
/// partition; these are not aliases for an INT8 execution ledger.
#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractionMappingLimits {
    pub chunks: ChunkLimits,
    pub reduction: ExecutionLimits,
    pub max_model_work: Int4Work,
    pub mask_limits: MaskWorkLimits,
    pub mask_visits_per_chunk: u64,
    pub max_mask_visits: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractionMapLimits {
    pub mapping: Int4ExtractionMappingLimits,
    /// Additional nonrenewable original-coordinate verification across ALL
    /// chunks. Native schema/source validation retains its own per-pass caps.
    pub verification: GroundingBudget,
}

/// Default diagnostics reveal neither source text nor schema/compiler details.
pub enum Int4ExtractionMapError {
    InvalidLimits, EmptySource, WorkLimit, Admission, Allocation,
    Chunk(MapReduceError), Extraction(Int4ExtractError), Reduction(ExecutionError<Int4ExtractError>),
}
impl fmt::Display for Int4ExtractionMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid INT4 extraction document limits",
            Self::EmptySource => "INT4 extraction document requires a nonempty source",
            Self::WorkLimit => "INT4 extraction whole-document allowance exceeded",
            Self::Admission => "INT4 extraction document identity or resident capacity refused",
            Self::Allocation => "INT4 extraction document allocation refused",
            Self::Chunk(_) => "INT4 extraction document partition refused",
            Self::Extraction(_) => "INT4 extraction document native task refused",
            Self::Reduction(_) => "INT4 extraction document reduction refused",
        })
    }
}
impl fmt::Debug for Int4ExtractionMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4ExtractionMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Extraction(e) => Some(e), Self::Reduction(e) => Some(e), _ => None }
    }
}
impl From<Int4ExtractError> for Int4ExtractionMapError { fn from(e: Int4ExtractError) -> Self { Self::Extraction(e) } }
impl From<ExtractError> for Int4ExtractionMapError { fn from(e: ExtractError) -> Self { Self::Extraction(e.into()) } }
impl Int4ExtractionMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Extraction(e) | Self::Reduction(ExecutionError::Task { source: e, .. }) => e.cancellation(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    work: Int4Work,
    masks: u64,
    group_width: usize,
    grounding: ExtractionGrounding,
    verification: GroundingBudget,
}
impl Int4ExtractionMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn verify_completed(&self, run: &Int4ExtractionMapRun) -> Result<(), Int4ExtractionMapError> {
        let root = run.mapped.root();
        let (work, masks, fields, spans) = execution::statistics(root.value())?;
        if run.schema_version != 1 || run.execution != INT4_EXTRACTION_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.semantics != SEMANTICS || run.group_width != self.group_width
            || run.grounding != self.grounding || run.untrusted_fields != ["mapped.root.value"]
            || root.source_span() != self.source_span || root.chunk_range() != (0..self.chunks)
            || root.value().chunks.len() != self.chunks || run.planned_model_work != self.work
            || run.reserved_mask_node_visits != self.masks || run.model_work != work
            || run.mask_node_visit_charge != masks || !fits(work, self.work) || masks > self.masks
            || run.verification_used.fields != fields || run.verification_used.matches != spans
            || fields > self.verification.max_fields || spans > self.verification.max_matches
            || run.verification_used.scan_steps > self.verification.max_scan_steps {
            return Err(ExtractError::InvalidResult.into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
pub struct MappedExtractionChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Unchanged native result, including exact JSON STRING and local evidence.
    pub native: Int4ExtractRun,
    /// Same schema field pointers, with all chunk-local occurrences lifted to
    /// original coordinates. This does not claim matches in other chunks.
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
pub struct Int4ExtractionMapRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub semantics: &'static str,
    pub grounding: ExtractionGrounding,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: ExtractionMapVerification,
    pub untrusted_fields: [&'static str; 1],
    pub mapped: MapReduceResult<ExtractionMapValue>,
}

/// Source-borrowing, consumed commitment. No wire constructor or external
/// native-result input can replace the exact prepared schema/source identities.
pub struct PreparedInt4ExtractionMap<'s> {
    chunks: ChunkPlan<'s>,
    plans: Vec<PreparedInt4BatchExtraction>,
    limits: Int4ExtractionMapLimits,
    expected: Int4ExtractionMapPreflight,
    kv_ceiling: u64,
}
impl PreparedInt4ExtractionMap<'_> {
    pub fn preflight_metadata(&self) -> Int4ExtractionMapPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.plans.iter().map(PreparedInt4BatchExtraction::execution_identity)
    }
    pub fn max_result_bytes(&self) -> u64 { self.limits.mapping.reduction.max_result_bytes as u64 }
    pub fn group_width(&self) -> usize { self.expected.group_width }
    pub fn max_kv_bytes(&self) -> u64 { self.kv_ceiling }
}
impl Int4ExtractionBatchPlanner {
    /// Compile every actual chunk, not a placeholder document. The caller must
    /// retain a preparation guard for this complete set. Metadata can survive
    /// dropping a preliminary plan; the charged host recompiles the same source.
    pub fn plan_document_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        args: &ExtractionBatchArgs, limits: Int4ExtractionMapLimits, control: &mut C)
        -> Result<PreparedInt4ExtractionMap<'s>, Int4ExtractionMapError> {
        step(control)?;
        if source.is_empty() { return Err(Int4ExtractionMapError::EmptySource); }
        let b = args.budget; let c = &self.compiler; let m = limits.mapping; let v = limits.verification;
        b.validate().map_err(|_| Int4ExtractionMapError::InvalidLimits)?;
        // A stream's sealed shared policy cannot be overridden by a document
        // map. An identical request uses those same immutable defaults.
        if self.shared_defaults_only && c.defaults.as_ref().is_none_or(|shared|
            shared.schema != args.schema || shared.grounding != args.grounding || shared.budget != args.budget) {
            return Err(Int4ExtractionMapError::Admission);
        }
        if b.max_input_tokens > c.ceiling.max_input_tokens || b.max_output_tokens > c.ceiling.max_output_tokens
            || b.max_output_bytes > c.ceiling.max_output_bytes || b.max_grammar_states > c.ceiling.max_grammar_states
            || b.max_kv_bytes > c.ceiling.max_kv_bytes || args.schema.len() > c.compiler_limits.max_schema_bytes
            || !(1..=256).contains(&m.chunks.max_chunks) || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
            || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
            || !valid_reduction(m.reduction) || !positive_work(m.max_model_work) || !(1..=1_000_000).contains(&v.max_fields)
            || !(1..=1_000_000).contains(&v.max_matches) || v.max_scan_steps == 0 {
            return Err(Int4ExtractionMapError::InvalidLimits);
        }
        // Use the factory's exact pinned fragments and source encoder. Schema
        // remains exact declarative bytes, including decimal constants and keys;
        // neither it nor source text passes through trusted template rendering.
        let scaffold = c.fragments.iter().try_fold(0_usize, |n, f| n.checked_add(f.len()))
            .ok_or(Int4ExtractionMapError::WorkLimit)?;
        let schema = c.encoder.encode(&args.schema, c.compiler_limits.max_schema_bytes, b.max_input_tokens as usize)
            .map_err(|_| Int4ExtractionMapError::InvalidLimits)?;
        let non_source = scaffold.checked_add(schema.total_token_count()).ok_or(Int4ExtractionMapError::WorkLimit)?;
        let token_room = (b.max_input_tokens as usize).checked_sub(non_source).filter(|&n| n > 0)
            .ok_or(Int4ExtractionMapError::InvalidLimits)?;
        let byte_room = (b.max_input_tokens as usize).checked_sub(scaffold)
            .and_then(|n| n.checked_sub(args.schema.len())).filter(|&n| n > 0).ok_or(Int4ExtractionMapError::InvalidLimits)?;
        let mut chunk_limits = m.chunks;
        chunk_limits.effective_token_limit().map_err(Int4ExtractionMapError::Chunk)?;
        chunk_limits.reserved_tokens = chunk_limits.reserved_tokens.max(non_source.checked_add(b.max_output_tokens as usize)
            .ok_or(Int4ExtractionMapError::WorkLimit)?);
        chunk_limits.max_chunk_tokens = chunk_limits.max_chunk_tokens.min(token_room);
        chunk_limits.max_chunk_bytes = chunk_limits.max_chunk_bytes.min(byte_room.max(4));
        let mut cancelled = None;
        let chunks = ChunkPlan::build_with_checkpoints(source, chunk_limits, |text| {
            c.encoder.encode(text, chunk_limits.max_chunk_bytes, chunk_limits.max_chunk_bytes)
                .map(|d| d.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
        }, || match control.prefill_checkpoint(0) {
            Some(cause) => { cancelled = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
        });
        if let Some(cause) = cancelled { return Err(cancelled_error(cause).into()); }
        let chunks = chunks.map_err(Int4ExtractionMapError::Chunk)?;
        let first = chunks.chunks().first().ok_or(Int4ExtractionMapError::EmptySource)?.span();
        let last = chunks.chunks().last().ok_or(Int4ExtractionMapError::EmptySource)?.span();
        let mut expected = Int4ExtractionMapPreflight {
            source_span: VerifiedSourceSpan { byte_start: first.byte_start, byte_end: last.byte_end,
                scalar_start: first.scalar_start, scalar_end: last.scalar_end }, chunks: chunks.chunks().len(),
            work: Int4Work::default(), masks: 0, group_width: self.group_width, verification: v,
            grounding: match args.grounding { ExtractionBatchGrounding::Structural => ExtractionGrounding::NotRequested,
                ExtractionBatchGrounding::SourceMembership => ExtractionGrounding::SourceMembership } };
        let mut plans = Vec::new();
        plans.try_reserve_exact(expected.chunks).map_err(|_| Int4ExtractionMapError::Allocation)?;
        for chunk in chunks.chunks() {
            step(control)?;
            let task_args = if self.shared_defaults_only { None } else {
                Some(ExtractionBatchArgs { schema: copy(&args.schema)?, grounding: args.grounding, budget: b })
            };
            let plan = self.prepare_with_control(BatchDocument { id: format!("chunk_{}", chunk.id()), text: copy(chunk.text())?,
                task_args }, control)
                .map_err(batch_error)?;
            if non_source.checked_add(chunk.tokens()) != Some(plan.plan.prompt_tokens())
                || plan.plan.prompt_tokens().checked_add(b.max_output_tokens as usize).is_none_or(|n| n > chunk_limits.context_tokens)
                || plan.source().text() != chunk.text() || plan.plan.group_width() != self.group_width {
                return Err(Int4ExtractionMapError::Admission);
            }
            expected.work = add_work(expected.work, plan.planned_work()).filter(|w| fits(*w, m.max_model_work))
                .ok_or(Int4ExtractionMapError::WorkLimit)?;
            expected.masks = expected.masks.checked_add(m.mask_visits_per_chunk).filter(|&n| n <= m.max_mask_visits)
                .ok_or(Int4ExtractionMapError::WorkLimit)?;
            plans.push(plan);
        }
        step(control)?;
        Ok(PreparedInt4ExtractionMap { chunks, plans, limits, expected, kv_ceiling: b.max_kv_bytes })
    }
}
fn step<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4ExtractError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(cancelled_error(cause)), None => Ok(()) }
}
fn cancelled_error(cause: DecodeCancellationKind) -> Int4ExtractError {
    ExtractError::Decode(JsonDecodeError::Cancelled(cause)).into()
}
fn batch_error(error: BatchItemFailure) -> Int4ExtractionMapError {
    if let Some(cause) = error.fault.cancellation { return cancelled_error(cause).into(); }
    match error.fault.code { BatchCode::Allocation => Int4ExtractionMapError::Allocation,
        BatchCode::WorkLimit => Int4ExtractionMapError::WorkLimit, _ => Int4ExtractionMapError::Admission }
}
fn copy(text: &str) -> Result<String, Int4ExtractionMapError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| Int4ExtractionMapError::Allocation)?;
    out.push_str(text); Ok(out)
}

fn add_work(left: Int4Work, right: Int4Work) -> Option<Int4Work> { left.checked_add(right).ok() }
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn positive_work(work: Int4Work) -> bool {
    work.forward_positions > 0 && work.projected_logits > 0 && work.attention_pairs > 0
        && work.projections.dot_products > 0 && work.projections.multiply_accumulates > 0
}
fn valid_reduction(limits: ExecutionLimits) -> bool {
    (1..=1024).contains(&limits.map_batch_chunks) && (2..=1024).contains(&limits.reduce_fan_in)
        && limits.max_reduction_levels <= 64 && (1..=1_000_000).contains(&limits.max_task_calls)
        && [limits.max_value_bytes, limits.max_live_value_bytes, limits.max_total_value_bytes, limits.max_result_bytes]
            .iter().all(|n| (1..=64 * 1024 * 1024).contains(n))
        && limits.max_value_bytes <= limits.max_live_value_bytes
}

#[cfg(test)] mod tests;
