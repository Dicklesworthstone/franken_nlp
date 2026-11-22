//! Exact-schema extraction over a complete original document on one INT8 engine.
//!
//! Reuse the pinned extraction compiler and the common lossless chunk/reduce
//! coordinator. Every real schema/source plan and all five work axes are funded
//! before the first forward. Independent JSON values are NOT merged into a
//! fabricated global object; required fields are not replaced with guessed nulls.
use super::*;
use std::{cell::RefCell, sync::{Arc, Mutex}};
use crate::{
    tasks::{extract::{ExtractionGrounding, quantized::check_size},
        mapreduce::{self, ChunkPlan, MapReduceError, MapReduceResult, MapReduceTask, MapOutput,
            ReduceInput, ReductionPolicy, SourceChunk},
        source_planning::quantized::{Int8SourceError, long::{Int8SourceMapError, Int8SourceMapLimits}}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan,
        SourceOccurrence, FieldGroundingError, scan_occurrences},
};
mod execution;
/// Bounded corpus transport over complete documents and one native driver.
pub mod stream;

pub const INT8_EXTRACTION_MAP_EXECUTION: &str = "portable-int8-exact-schema-document-map-v1";
const SEMANTICS: &str = "independent-chunk-json-no-global-object-merge-v1";

/// The planner owns one nonrenewable operation ledger across sizing, every
/// real schema/source plan, and hosted reconstruction. No source is cached.
pub(super) struct MapEncoding {
    limits: SafeBpeWorkLimits,
    used: Mutex<SafeBpeWork>,
}

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
    /// Configure complete-document compact extraction before deriving any
    /// execution identity. The same planner must survive CLI preflight and
    /// hosted reconstruction; another call never restores spent work.
    pub fn with_map_document_encoding(mut self, policy: UntrustedEncoding,
        limits: SafeBpeWorkLimits) -> Result<Self, BatchFault> {
        if !matches!(policy, UntrustedEncoding::SafeBpeV1(_)) {
            return Err(BatchCode::InvalidLimits.into());
        }
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self = self.with_document_encoding(policy)?;
        // Keep the pinned template binding while extending its semantic
        // identity with ALL declared limits, never decreasing remaining work.
        self.compiler.identity.template_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            "int8-extraction-map-encoding-v1", self.compiler.identity.template_digest,
            self.document_encoding(), limits,
        )).map_err(|_| BatchCode::Admission)?);
        self.map_encoding = Some(MapEncoding { limits, used: Mutex::new(SafeBpeWork::default()) });
        Ok(self)
    }
    pub fn map_document_encoding_limits(&self) -> Option<SafeBpeWorkLimits> {
        self.map_encoding.as_ref().map(|ledger| ledger.limits)
    }
    pub fn map_document_encoding_work(&self) -> Result<Option<SafeBpeWork>, BatchFault> {
        self.map_encoding.as_ref().map(|ledger| ledger.used.try_lock().map(|used| *used)
            .map_err(|_| BatchCode::Admission.into())).transpose()
    }
    /// Extra transient encoder and ledger storage, beside all retained plans.
    /// A record-local policy alone cannot authorize repeated document probes.
    pub fn map_document_encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        match (self.document_encoding(), self.map_document_encoding_limits()) {
            (UntrustedEncoding::ByteFallback, None) => Ok(0),
            (UntrustedEncoding::SafeBpeV1(_), Some(limits)) => {
                limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
                if std::mem::size_of::<MapEncoding>() > 1024 { return Err(BatchCode::InvalidLimits.into()); }
                self.encoding_workspace_bytes()?.checked_add(1024).ok_or_else(|| BatchCode::InvalidLimits.into())
            },
            _ => Err(BatchCode::InvalidLimits.into()),
        }
    }
    /// Compile every actual chunk, not a placeholder document. The caller must
    /// retain a preparation guard for this complete set; CLI preflight drops it
    /// before loading weights and the charged host builds the same set again.
    pub fn plan_document_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        args: &ExtractionBatchArgs, limits: Int8ExtractionMapLimits, control: &mut C)
        -> Result<PreparedInt8ExtractionMap<'s>, Int8SourceMapError> {
        step(control)?;
        self.map_document_encoding_workspace_bytes().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        let total = self.map_document_encoding_limits();
        let mut shared = self.map_encoding.as_ref().map(|ledger| ledger.used.try_lock()
            .map_err(|_| Int8SourceMapError::Admission)).transpose()?;
        let mut byte_work = SafeBpeWork::default();
        let used = shared.as_deref_mut().unwrap_or(&mut byte_work);
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
        let schema = self.encode_map_segment(&args.schema, c.compiler_limits.max_schema_bytes,
            b.max_input_tokens as usize, used, total, control)?;
        let non_source = scaffold.checked_add(schema.total_token_count()).ok_or(Int8SourceMapError::WorkLimit)?;
        let token_room = (b.max_input_tokens as usize).checked_sub(non_source).filter(|&n| n > 0)
            .ok_or(Int8SourceMapError::InvalidLimits)?;
        let byte_room = match self.document_encoding() {
            UntrustedEncoding::ByteFallback => (b.max_input_tokens as usize).checked_sub(scaffold)
                .and_then(|n| n.checked_sub(args.schema.len())),
            UntrustedEncoding::SafeBpeV1(call) => call.max_input_bytes.checked_sub(args.schema.len()),
        }.filter(|&n| n > 0).ok_or(Int8SourceMapError::InvalidLimits)?;
        let mut chunk_limits = m.chunks;
        chunk_limits.effective_token_limit().map_err(Int8SourceMapError::Chunk)?;
        chunk_limits.reserved_tokens = chunk_limits.reserved_tokens.max(non_source.checked_add(b.max_output_tokens as usize)
            .ok_or(Int8SourceMapError::WorkLimit)?);
        chunk_limits.max_chunk_tokens = chunk_limits.max_chunk_tokens.min(token_room);
        chunk_limits.max_chunk_bytes = chunk_limits.max_chunk_bytes.min(byte_room.max(4));
        let mut cancelled = None;
        let mut encoding_error = None;
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints(source, chunk_limits, |text| {
                self.encode_map_segment(text, chunk_limits.max_chunk_bytes, chunk_limits.max_chunk_bytes,
                    used, total, &mut **control.borrow_mut())
                    .map(|document| document.total_token_count()).map_err(|error| {
                        encoding_error = Some(error); MapReduceError::Tokenizer
                    })
            }, || match control.borrow_mut().prefill_checkpoint(0) {
                Some(cause) => { cancelled = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
            })
        };
        if let Some(cause) = cancelled { return Err(Int8SourceError::Cancelled(cause).into()); }
        if let Some(error) = encoding_error { return Err(error); }
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
            let plan = self.prepare_with_invocation_encoding_work(BatchDocument {
                id: format!("chunk_{}", chunk.id()), text: copy(chunk.text())?,
                task_args: Some(ExtractionBatchArgs { schema: copy(&args.schema)?, grounding: args.grounding, budget: b }) },
                used, total, control)
                .map_err(batch_error)?;
            if non_source.checked_add(chunk.tokens()) != Some(plan.plan.prompt_tokens())
                || plan.plan.prompt_tokens().checked_add(b.max_output_tokens as usize).is_none_or(|n| n > chunk_limits.context_tokens)
                || plan.source().text() != chunk.text()
                || plan.source().total_token_count() != chunk.tokens()
                || plan.source().document_encoding() != self.document_encoding() {
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
    fn encode_map_segment<C: DecodeStepControl>(&self, text: &str, max_bytes: usize,
        max_tokens: usize, used: &mut SafeBpeWork, total: Option<SafeBpeWorkLimits>, control: &mut C)
        -> Result<SourceDocument, Int8SourceMapError> {
        step(control)?;
        let effective = match (self.document_encoding(), total) {
            (UntrustedEncoding::ByteFallback, None) if *used == SafeBpeWork::default() =>
                UntrustedEncoding::ByteFallback,
            (UntrustedEncoding::SafeBpeV1(call), Some(total)) => UntrustedEncoding::SafeBpeV1(
                total.remaining_call(*used, *call).map_err(|_| Int8SourceMapError::WorkLimit)?),
            _ => return Err(Int8SourceMapError::InvalidLimits),
        };
        let mut work = SafeBpeWork::default();
        let result = self.compiler.encoder.encode_with_policy_and_work(text, &effective,
            max_bytes, max_tokens, &mut work, &mut || control.prefill_checkpoint(0));
        // Failure and cancellation never refund work already performed.
        *used = used.checked_add(work).ok_or(Int8SourceMapError::WorkLimit)?;
        result.map_err(|error| Int8SourceError::from(error).into())
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
