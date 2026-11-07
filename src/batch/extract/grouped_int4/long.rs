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
    tasks::{extract::{ExtractionGrounding, quantized::check_size,
            array_union::{self, RootArrayUnionLimits, RootArrayUnionResult, RootArrayUnionError, ROOT_ARRAY_UNION_POLICY}},
        extract::grouped_int4::Int4ExtractRun,
        mapreduce::{self, ChunkLimits, ChunkPlan, ExecutionError, ExecutionLimits, MapReduceError,
            MapReduceResult, MapReduceTask, MapOutput, ReduceInput, ReductionPolicy, SourceChunk}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan,
        SourceOccurrence, FieldGroundingError, scan_occurrences},
};
#[cfg(test)] use crate::{native_engine::constrained_int4::Int4JsonBudget,
    tasks::extract::grouped_int4::INT4_EXTRACT_VERSION};
mod execution;
pub mod cohort;
/// Ordered live populations of independently complete source documents.
pub mod stream;

pub const INT4_EXTRACTION_MAP_EXECUTION: &str = "grouped-int4-exact-schema-document-map-v1";
pub const INT4_EXTRACTION_MAP_SELECTED_EXECUTION: &str = "grouped-int4-selected-row-exact-schema-document-map-v1";
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
    /// Explicit exact union of root-array records; None preserves independent
    /// chunk output and its original serialized identity.
    pub root_array_union: Option<RootArrayUnionLimits>,
}
impl Int4ExtractionMapLimits {
    /// Additional simultaneous reducer payload beyond the ordinary map quote.
    /// The complete document cap still covers BOTH chunks and optional union.
    pub fn root_array_workspace_bytes(self) -> Result<u64, Int4ExtractionMapError> {
        let Some(limits) = self.root_array_union else { return Ok(0); };
        if limits.max_result_bytes > self.mapping.reduction.max_result_bytes
            || limits.max_json_bytes > self.mapping.reduction.max_result_bytes {
            return Err(Int4ExtractionMapError::InvalidLimits);
        }
        Ok(limits.requirements()?.workspace_bytes)
    }
}

/// Default diagnostics reveal neither source text nor schema/compiler details.
pub enum Int4ExtractionMapError {
    InvalidLimits, EmptySource, WorkLimit, Admission, Allocation,
    Chunk(MapReduceError), Extraction(Int4ExtractError), Reduction(ExecutionError<Int4ExtractError>),
    ArrayUnion(RootArrayUnionError),
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
            Self::ArrayUnion(_) => "INT4 extraction root-array union refused",
        })
    }
}
impl fmt::Debug for Int4ExtractionMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4ExtractionMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Extraction(e) => Some(e), Self::Reduction(e) => Some(e),
            Self::ArrayUnion(e) => Some(e), _ => None }
    }
}
impl From<Int4ExtractError> for Int4ExtractionMapError { fn from(e: Int4ExtractError) -> Self { Self::Extraction(e) } }
impl From<ExtractError> for Int4ExtractionMapError { fn from(e: ExtractError) -> Self { Self::Extraction(e.into()) } }
impl From<RootArrayUnionError> for Int4ExtractionMapError { fn from(e: RootArrayUnionError) -> Self { Self::ArrayUnion(e) } }
impl Int4ExtractionMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Extraction(e) | Self::Reduction(ExecutionError::Task { source: e, .. }) => e.cancellation(),
            Self::ArrayUnion(RootArrayUnionError::Cancelled(cause)) => Some(*cause),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int4ExtractionMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    work: Int4Work,
    masks: u64,
    group_width: usize,
    grounding: ExtractionGrounding,
    verification: GroundingBudget,
    root_array_binding: Option<Sha256Digest>,
    selected_rows: Option<Int4JsonSparseLimits>,
}
impl fmt::Debug for Int4ExtractionMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4ExtractionMapPreflight")
            .field("source_span", &self.source_span).field("chunks", &self.chunks)
            .field("work", &self.work).field("masks", &self.masks).field("group_width", &self.group_width)
            .field("grounding", &self.grounding).field("verification", &self.verification)
            .field("selected_rows", &self.selected_rows)
            .field("root_array_union", &self.root_array_binding.is_some()).finish()
    }
}
impl Int4ExtractionMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.selected_rows }
    pub fn execution_version(&self) -> &'static str {
        if self.selected_rows.is_some() { INT4_EXTRACTION_MAP_SELECTED_EXECUTION } else { INT4_EXTRACTION_MAP_EXECUTION }
    }
    pub fn verify_completed(&self, run: &Int4ExtractionMapRun) -> Result<(), Int4ExtractionMapError> {
        let root = run.mapped.root();
        let (work, masks, fields, spans) = execution::statistics(root.value(), self.group_width, self.selected_rows)?;
        let same_children = run.root_array_union.as_ref().is_none_or(|union| union.matches_chunks(
            root.value().chunks().map(|chunk| array_union::ArrayUnionChunk {
                chunk_id: chunk.chunk_id, source_span: chunk.source_span,
                json: &chunk.native.result.output.json, source_fields: &chunk.original_fields,
            })));
        if run.schema_version != 1 || run.execution != self.execution_version()
            || run.max_selected_rows != self.selected_rows.map(|limits| limits.max_rows_per_step)
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.semantics != if self.root_array_binding.is_some() { ROOT_ARRAY_UNION_POLICY } else { SEMANTICS }
            || run.root_array_union.as_ref().map(RootArrayUnionResult::binding) != self.root_array_binding
            || !same_children
            || run.grounding != self.grounding || run.untrusted_fields.as_slice() != if self.root_array_binding.is_some() {
                &["mapped.root.value", "root_array_union"][..] } else { &["mapped.root.value"][..] }
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
    /// Complete legal-row ceiling per token selection, including terminal EOS.
    /// Omitted for the unchanged full-vocabulary document contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_selected_rows: Option<usize>,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: ExtractionMapVerification,
    pub untrusted_fields: Vec<&'static str>,
    pub mapped: MapReduceResult<ExtractionMapValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_array_union: Option<RootArrayUnionResult>,
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
    fn reduce_root_array<C: DecodeStepControl>(&self, mapped: &MapReduceResult<ExtractionMapValue>, control: &mut C)
        -> Result<Option<RootArrayUnionResult>, Int4ExtractionMapError> {
        let Some(limits) = self.limits.root_array_union else { return Ok(None); };
        let first = self.plans.first().ok_or(Int4ExtractionMapError::Admission)?;
        let schema = first.plan.semantic_schema(&first.task, &first.source)?;
        let binding = self.expected.root_array_binding.ok_or(Int4ExtractionMapError::Admission)?;
        Ok(Some(array_union::reduce(schema, binding, limits,
            mapped.root().value().chunks().map(|chunk| array_union::ArrayUnionChunk {
                chunk_id: chunk.chunk_id, source_span: chunk.source_span,
                json: &chunk.native.result.output.json, source_fields: &chunk.original_fields,
            }), |value| first.plan.validate_reduced_value(&first.task, &first.source, value)
                .map_err(|_| RootArrayUnionError::Input), control)?))
    }
}
impl Int4ExtractionBatchPlanner {
    /// Compile every actual chunk, not a placeholder document. The caller must
    /// retain a preparation guard for this complete set. Metadata can survive
    /// dropping a preliminary plan; the charged host recompiles the same source.
    pub fn plan_document_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        args: &ExtractionBatchArgs, limits: Int4ExtractionMapLimits, control: &mut C)
        -> Result<PreparedInt4ExtractionMap<'s>, Int4ExtractionMapError> {
        step(control)?;
        limits.root_array_workspace_bytes()?;
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
            work: Int4Work::default(), masks: 0, group_width: self.group_width, verification: v, root_array_binding: None,
            selected_rows: self.selected_rows(),
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
                || plan.source().text() != chunk.text() || plan.plan.group_width() != self.group_width
                || plan.plan.selected_rows() != expected.selected_rows {
                return Err(Int4ExtractionMapError::Admission);
            }
            expected.work = add_work(expected.work, plan.planned_work()).filter(|w| fits(*w, m.max_model_work))
                .ok_or(Int4ExtractionMapError::WorkLimit)?;
            expected.masks = expected.masks.checked_add(m.mask_visits_per_chunk).filter(|&n| n <= m.max_mask_visits)
                .ok_or(Int4ExtractionMapError::WorkLimit)?;
            plans.push(plan);
        }
        if let Some(union) = limits.root_array_union {
            let first = plans.first().ok_or(Int4ExtractionMapError::Admission)?;
            union.validate_schema(first.plan.semantic_schema(&first.task, &first.source)?)?;
            let mut identities = Vec::new();
            identities.try_reserve_exact(plans.len()).map_err(|_| Int4ExtractionMapError::Allocation)?;
            for plan in &plans {
                step(control)?;
                identities.push(plan.execution_identity().receipt_digest().map_err(|_| Int4ExtractionMapError::Admission)?);
            }
            // A sealed aggregate from another source, model, chunk partition,
            // schema or policy cannot be transplanted into this map result.
            let bytes = canonjson::canonical_bytes(&(union.binding(first.execution_identity().schema_digest)?,
                Sha256Digest::of_bytes(source.as_bytes()), identities)).map_err(|_| Int4ExtractionMapError::Admission)?;
            expected.root_array_binding = Some(Sha256Digest::of_bytes(&bytes));
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
