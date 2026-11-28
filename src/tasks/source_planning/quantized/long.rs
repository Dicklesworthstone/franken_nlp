//! Native INT8 map/merge over an entire, losslessly partitioned document.
//!
//! This preserves independent chunk results, not a fabricated global summary,
//! ranking, entity census or single-context equivalent. The existing map/reduce
//! coordinator owns ordering/lineage/value budgets; the native driver owns all
//! neural calls. Every prompt and the complete work ceiling are prepared before
//! the first forward. There is no alternate tokenizer, runtime, retry or model.

use super::*;
use std::{cell::RefCell, sync::Arc};
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, kv::KV_BYTES_PER_TOKEN,
        strict_int8::{Int8RunBudget, prefill::Int8PrefillLimits}},
    tasks::mapreduce::{self, ChunkLimits, ChunkPlan, ExecutionError, ExecutionLimits,
        MapOutput, MapReduceError, MapReduceResult, MapReduceTask, ReduceInput,
        ReductionPolicy, SourceChunk},
    validation::grounded_fields::VerifiedSourceSpan,
};

/// Opt-in complete cited-summary reduction; the default ordered map is unchanged.
pub mod summary;
/// Exact document-wide keyphrase ranking over all native map candidates.
pub mod keyphrase_reduction;
/// Exact typed-surface union with all evidence from proposing chunks.
pub mod entity_reduction;
/// Question-aware passage planning and evidence-preserving answer collection.
pub mod question;

pub const INT8_SOURCE_MAP_EXECUTION: &str = "portable-int8-source-map-ordered-merge-v1";
const MAX_SOURCE_CHUNKS: usize = 256;

/// Options apply independently to EACH chunk. Question-aware QA has a separate
/// SourceQuestion planner; arbitrary source-only maps cannot acquire QA semantics.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "task", content = "options", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceMapTask { Ner(NerOptions), Keyphrases(KeyphraseOptions), Summarize(SummaryOptions) }
impl SourceMapTask {
    fn kind(&self) -> BuiltInTask {
        match self { Self::Ner(_) => BuiltInTask::Ner, Self::Keyphrases(_) => BuiltInTask::Keyphrases,
            Self::Summarize(_) => BuiltInTask::Summarize }
    }
    fn request(&self, document: String, budget: TaskBudget) -> SourceTaskRequest {
        match self {
            Self::Ner(options) => SourceTaskRequest::Ner { document, options: options.clone(), budget },
            Self::Keyphrases(options) => SourceTaskRequest::Keyphrases { document, options: *options, budget },
            Self::Summarize(options) => SourceTaskRequest::Summarize { document, options: *options, budget },
        }
    }
}

/// Whole invocation ceilings, never renewed per chunk or reduction level.
/// The host separately prices retained prepared grammars/prompts, vocabulary,
/// allocator overhead and reduction metadata. Serialized sizes are not RSS.
#[derive(Clone, Copy, Debug)]
pub struct Int8SourceMapLimits {
    pub chunks: ChunkLimits,
    pub reduction: ExecutionLimits,
    pub max_model_work: Int8Work,
    pub mask_limits: MaskWorkLimits,
    pub mask_visits_per_chunk: u64,
    pub max_mask_visits: u64,
}
impl Int8SourceMapLimits {
    pub(crate) fn validate(self, budget: TaskBudget, planning: SourcePlanningLimits) -> Result<(), Int8SourceMapError> {
        self.chunks.effective_token_limit().map_err(Int8SourceMapError::Chunk)?;
        budget.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        if self.chunks.max_chunks > MAX_SOURCE_CHUNKS
            || self.chunks.max_chunk_bytes > planning.max_input_bytes
            || self.chunks.context_tokens > planning.max_context_tokens
            || self.chunks.reserved_tokens < budget.max_output_tokens as usize
            || self.mask_visits_per_chunk == 0 || self.max_mask_visits < self.mask_visits_per_chunk
            || self.mask_limits.max_trie_node_visits == 0 || self.mask_limits.checkpoint_interval_nodes == 0
            || self.reduction.max_result_bytes == 0 {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}

/// No private document/compiler diagnostics in default formatting.
pub enum Int8SourceMapError {
    InvalidLimits, EmptySource, WorkLimit, Admission, Allocation,
    Chunk(MapReduceError), Source(Int8SourceError), Reduction(ExecutionError<Int8SourceError>),
}
impl fmt::Display for Int8SourceMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid int8 source map limits",
            Self::EmptySource => "int8 source map requires a nonempty document",
            Self::WorkLimit => "int8 source map whole-document work exceeded",
            Self::Admission => "int8 source map identity or resident capacity refused",
            Self::Allocation => "int8 source map allocation refused",
            Self::Chunk(_) => "int8 source map partition refused",
            Self::Source(_) => "int8 source map native task refused",
            Self::Reduction(_) => "int8 source map reduction refused",
        })
    }
}
impl fmt::Debug for Int8SourceMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8SourceMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Source(e) => Some(e), Self::Reduction(e) => Some(e), _ => None }
    }
}
impl From<Int8SourceError> for Int8SourceMapError { fn from(e: Int8SourceError) -> Self { Self::Source(e) } }
impl Int8SourceMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Source(e) | Self::Reduction(ExecutionError::Task { source: e, .. }) => e.cancellation(),
            _ => None,
        }
    }
}

/// Typed location in the UNCHANGED chunk-local native result.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceMapField { Entity { index: usize }, Keyphrase { index: usize }, SummaryCitation { bullet: usize, citation: usize } }
#[derive(Serialize)]
pub struct OriginalSourceSpans { pub field: SourceMapField, pub spans: Vec<VerifiedSourceSpan> }

/// Native result coordinates remain CHUNK-LOCAL. original_spans supplies exact
/// original-document coordinates for every reported occurrence, without
/// relabeling the native result or claiming occurrences in unexamined chunks.
#[derive(Serialize)]
pub struct MappedSourceChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub native: Int8SourceTaskRun,
    pub original_spans: Vec<OriginalSourceSpans>,
}

/// Reductions share immutable native results; they never deep-clone model text
/// or token buffers. No serde `rc` feature or exposed admission token is needed.
pub struct SourceMapValue { chunks: Vec<Arc<MappedSourceChunk>> }
impl SourceMapValue {
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = &MappedSourceChunk> { self.chunks.iter().map(Arc::as_ref) }
    pub fn len(&self) -> usize { self.chunks.len() }
    pub fn is_empty(&self) -> bool { self.chunks.is_empty() }
}
impl Serialize for SourceMapValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.chunks())
    }
}

#[derive(Serialize)]
pub struct Int8SourceMapRun {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Independent chunk judgments. No global reranking or cross-chunk reasoning.
    pub semantics: &'static str,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub mapped: MapReduceResult<SourceMapValue>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}

/// The exact admitted source, task, policy and complete partition, not a native
/// execution receipt. Private commitments never enter diagnostics or output.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int8SourceMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    task: BuiltInTask,
    work: Int8Work,
    masks: u64,
    max_value_bytes: usize,
    max_result_bytes: u64,
    request_binding: [u8; 32],
}
impl fmt::Debug for Int8SourceMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8SourceMapPreflight").field("source_span", &self.source_span)
            .field("chunks", &self.chunks).field("task", &self.task)
            .field("work", &self.work).field("masks", &self.masks).finish_non_exhaustive()
    }
}
impl Int8SourceMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn verify_completed(&self, run: &Int8SourceMapRun) -> Result<(), Int8SourceMapError> {
        self.verify_completed_with_checkpoints(run, &mut || Ok(()))
    }
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8SourceMapRun,
        control: &mut C) -> Result<(), Int8SourceMapError> {
        self.verify_completed_with_checkpoints(run,
            &mut || checkpoint(control).map_err(Int8SourceMapError::from))
    }
    fn verify_completed_with_checkpoints<F>(&self, run: &Int8SourceMapRun, check: &mut F)
        -> Result<(), Int8SourceMapError> where F: FnMut() -> Result<(), Int8SourceMapError> {
        check()?;
        if run.request_binding != self.request_binding || run.result_binding != source_map_result_binding(run,
            self.max_result_bytes, check)? { return Err(Int8SourceError::InvalidResult.into()); }
        let root = run.mapped.root();
        if run.schema_version != 1 || run.execution != INT8_SOURCE_MAP_EXECUTION
            || run.semantics != "independent-chunks-no-cross-chunk-reasoning-v1"
            || run.planned_model_work != self.work || run.reserved_mask_node_visits != self.masks
            || !within(run.model_work, self.work) || run.mask_node_visit_charge > self.masks
            || root.source_span() != self.source_span || root.chunk_range() != (0..self.chunks)
            || root.value().len() != self.chunks { return Err(Int8SourceError::InvalidResult.into()); }
        let mut byte_end = 0; let mut scalar_end = 0;
        let mut work = Int8Work::default(); let mut masks = 0_u64;
        for (index, chunk) in root.value().chunks().enumerate() {
            check()?;
            let span = chunk.source_span;
            let kind = match &chunk.native.result { SourceTaskResult::Ner(_) => BuiltInTask::Ner,
                SourceTaskResult::Keyphrases(_) => BuiltInTask::Keyphrases,
                SourceTaskResult::Summarize(_) => BuiltInTask::Summarize,
                SourceTaskResult::Answer(_) => return Err(Int8SourceError::InvalidResult.into()) };
            if chunk.chunk_id != index || kind != self.task
                || span.byte_start != byte_end || span.scalar_start != scalar_end
                || span.byte_end <= span.byte_start || span.scalar_end <= span.scalar_start {
                return Err(Int8SourceError::InvalidResult.into());
            }
            byte_end = span.byte_end; scalar_end = span.scalar_end;
            work = add_work(work, chunk.native.model_work).ok_or(Int8SourceMapError::WorkLimit)?;
            masks = masks.checked_add(mask_charge(&chunk.native.result)?).ok_or(Int8SourceMapError::WorkLimit)?;
        }
        if byte_end != self.source_span.byte_end || scalar_end != self.source_span.scalar_end
            || work != run.model_work || masks != run.mask_node_visit_charge {
            return Err(Int8SourceError::InvalidResult.into());
        }
        check()?;
        Ok(())
    }
}

fn source_map_result_binding<F>(run: &Int8SourceMapRun, cap: u64, check: &mut F)
    -> Result<[u8; 32], Int8SourceMapError> where F: FnMut() -> Result<(), Int8SourceMapError> {
    question::binding::composed_result_with_checkpoints(b"fnlp-int8-source-map-result-v1",
        run.request_binding, &[], run, cap, check)
}

/// Borrows the unchanged original text. Not Clone/Deserialize: execution
/// consumes this complete commitment rather than renewing its work ledger.
pub struct PreparedInt8SourceMap<'s> {
    chunks: ChunkPlan<'s>,
    plans: Vec<PreparedInt8SourceTask>,
    limits: Int8SourceMapLimits,
    work: Int8Work,
    masks: u64,
    expected: Int8SourceMapPreflight,
}
impl SourceTaskPlanner {
    /// Size and bind the actual document without retaining native grammars.
    /// The same partitioner and operation-owned encoding ledger are used by
    /// later full planning. No sizing work is refunded or renewed.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int8_map_with_control<C: DecodeStepControl>(&self, source: &str,
        task: &SourceMapTask, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, limits: Int8SourceMapLimits, control: &mut C)
        -> Result<Int8SourceMapPreflight, Int8SourceMapError> {
        limits.validate(budget, planning)?;
        let capacity = self.int8_map_capacity_with_control(task, budget, context, planning, control)?;
        let chunks = self.partition_int8_map_with_control(source, limits, control)?;
        let mut work = Int8Work::default();
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let prompt = capacity.scaffold_tokens().checked_add(chunk.tokens()).ok_or(Int8SourceMapError::WorkLimit)?;
            if prompt > budget.max_input_tokens as usize
                || prompt.checked_add(budget.max_output_tokens as usize)
                    .is_none_or(|n| n > planning.max_context_tokens) { return Err(Int8SourceMapError::WorkLimit); }
            let part = constrained_int8::planned_work(prompt, budget.max_output_tokens as usize)
                .map_err(|_| Int8SourceMapError::WorkLimit)?;
            work = add_work(work, part).filter(|&n| within(n, limits.max_model_work))
                .ok_or(Int8SourceMapError::WorkLimit)?;
        }
        let masks = limits.mask_visits_per_chunk.checked_mul(chunks.chunks().len() as u64)
            .filter(|&n| n <= limits.max_mask_visits).ok_or(Int8SourceMapError::WorkLimit)?;
        self.int8_map_expectation(source, task, budget, context, planning, limits, &chunks, work, masks, control)
    }

    /// Compile every chunk with the SAME admitted task/profile and source
    /// encoder. Complete prompts are independently checked by the task planner;
    /// an underestimated scaffold reserve fails BEFORE any model execution.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int8_map_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        task: &SourceMapTask, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, limits: Int8SourceMapLimits, control: &mut C)
        -> Result<PreparedInt8SourceMap<'s>, Int8SourceMapError> {
        limits.validate(budget, planning)?;
        checkpoint(control)?;
        // Compact sizing probes and final plans must share one nonrenewable
        // document ledger. A one-request encoder alone cannot authorize a map.
        self.map_document_encoding_workspace_bytes().map_err(Int8SourceError::from)?;
        constrained_int8::check_profile(context.execution_identity()).map_err(Int8SourceError::from)?;
        if source.is_empty() { return Err(Int8SourceMapError::EmptySource); }
        let chunks = self.partition_int8_map_with_control(source, limits, control)?;
        let masks = limits.mask_visits_per_chunk.checked_mul(chunks.chunks().len() as u64)
            .filter(|&n| n <= limits.max_mask_visits).ok_or(Int8SourceMapError::WorkLimit)?;
        let mut plans = Vec::new();
        plans.try_reserve_exact(chunks.chunks().len()).map_err(|_| Int8SourceMapError::Allocation)?;
        let mut work = Int8Work::default();
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let request = task.request(chunk.text().to_owned(), budget);
            let plan = self.plan_int8_with_control(&request, context, planning, control)?;
            work = add_work(work, plan.planned_work()).ok_or(Int8SourceMapError::WorkLimit)?;
            if !within(work, limits.max_model_work) { return Err(Int8SourceMapError::WorkLimit); }
            plans.push(plan);
        }
        let expected = self.int8_map_expectation(source, task, budget, context, planning, limits,
            &chunks, work, masks, control)?;
        checkpoint(control)?;
        Ok(PreparedInt8SourceMap { chunks, plans, limits, work, masks, expected })
    }

    fn partition_int8_map_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        limits: Int8SourceMapLimits, control: &mut C) -> Result<ChunkPlan<'s>, Int8SourceMapError> {
        checkpoint(control)?;
        self.map_document_encoding_workspace_bytes().map_err(Int8SourceError::from)?;
        if source.is_empty() { return Err(Int8SourceMapError::EmptySource); }
        let mut cancelled = None;
        let mut encoding_error = None;
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints(source, limits.chunks, |text| {
                // Count the complete candidate before the lossless partitioner
                // shrinks it. Rejected probes remain charged to the same ledger.
                self.encoder.encode_with_control(text, limits.chunks.max_chunk_bytes,
                    limits.chunks.max_chunk_bytes, &mut || control.borrow_mut().prefill_checkpoint(0))
                    .map(|document| document.total_token_count()).map_err(|error| {
                        encoding_error = Some(error); MapReduceError::Tokenizer
                    })
            }, || {
                if let Some(cause) = control.borrow_mut().prefill_checkpoint(0) {
                    cancelled = Some(cause); Err(MapReduceError::Cancelled)
                } else { Ok(()) }
            })
        };
        if let Some(cause) = cancelled { return Err(Int8SourceError::Cancelled(cause).into()); }
        if let Some(error) = encoding_error { return Err(Int8SourceError::from(error).into()); }
        chunks.map_err(Int8SourceMapError::Chunk)
    }

    #[allow(clippy::too_many_arguments)]
    fn int8_map_expectation<C: DecodeStepControl>(&self, source: &str, task: &SourceMapTask,
        budget: TaskBudget, context: &PlanContext<'_>, planning: SourcePlanningLimits,
        limits: Int8SourceMapLimits, chunks: &ChunkPlan<'_>, work: Int8Work, masks: u64, control: &mut C)
        -> Result<Int8SourceMapPreflight, Int8SourceMapError> {
        let last = chunks.chunks().last().ok_or(Int8SourceMapError::EmptySource)?.span();
        let request_binding = question::binding::document_request(b"fnlp-int8-source-map-request-v1",
            source, task, budget, context, planning, limits, self, chunks, control)?;
        Ok(Int8SourceMapPreflight { source_span: VerifiedSourceSpan { byte_start: 0, byte_end: source.len(),
            scalar_start: 0, scalar_end: last.scalar_end }, chunks: chunks.chunks().len(), task: task.kind(),
            work, masks, max_value_bytes: limits.reduction.max_value_bytes,
            max_result_bytes: limits.reduction.max_result_bytes as u64, request_binding })
    }
}
impl PreparedInt8SourceMap<'_> {
    pub fn preflight_metadata(&self) -> Int8SourceMapPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.plans.iter().map(PreparedInt8SourceTask::execution_identity)
    }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn chunk_count(&self) -> usize { self.plans.len() }
    pub fn max_result_bytes(&self) -> u64 { self.limits.reduction.max_result_bytes as u64 }

    /// Host must admit EVERY actual prepared identity before execution. The
    /// last chunk is checked before the first forward, including full resident
    /// KV pricing. The caller retains its process/output guards through delivery.
    pub fn preflight(&self, admitted: &[ExecutionIdentity], engine: &StrictInt8Engine<'_>) -> Result<(), Int8SourceMapError> {
        preflight_plans(&self.plans, admitted, engine)
    }

    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8SourceMapRun, Int8SourceMapError> {
        checkpoint(control)?;
        self.preflight(admitted, engine)?;
        let driver = NativeDriver { engine, vocabulary, control, limits: self.limits, prefill: None };
        self.execute_with_driver(admitted, driver)
    }

    /// Reuse bounded layer-major prompt storage across every original chunk.
    /// The same engine may carry one persistent projection team. Prepared
    /// identities, per-row arithmetic, source coordinates and the complete
    /// model/mask allowances remain unchanged.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
        prefill: Int8PrefillLimits, control: &mut C) -> Result<Int8SourceMapRun, Int8SourceMapError> {
        checkpoint(control)?;
        check_prefill(prefill)?;
        self.preflight(admitted, engine)?;
        let driver = NativeDriver { engine, vocabulary, control, limits: self.limits, prefill: Some(prefill) };
        self.execute_with_driver(admitted, driver)
    }

    fn execute_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity], driver: D)
        -> Result<Int8SourceMapRun, Int8SourceMapError> {
        verify_identities(&self.plans, admitted)?;
        let mut task = SourceMapExecutor { chunks: &self.chunks, plans: &self.plans, admitted,
            driver, next_chunk: 0, work: Int8Work::default(), masks: 0, mask_cap: self.limits.mask_visits_per_chunk };
        // The SAME control is borrowed by the native driver and every map/
        // merge checkpoint. No second no-op controller enters native execution.
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || Ok(()));
        task.driver.checkpoint()?;
        let mapped = mapped.map_err(Int8SourceMapError::Reduction)?;
        if task.next_chunk != self.plans.len() || mapped.root().value().len() != self.plans.len()
            || !within(task.work, self.work) || task.masks > self.masks { return Err(Int8SourceError::InvalidResult.into()); }
        let mut result = Int8SourceMapRun { schema_version: 1, execution: INT8_SOURCE_MAP_EXECUTION,
            semantics: "independent-chunks-no-cross-chunk-reasoning-v1", planned_model_work: self.work,
            model_work: task.work, reserved_mask_node_visits: self.masks,
            mask_node_visit_charge: task.masks, mapped,
            request_binding: self.expected.request_binding, result_binding: [0; 32] };
        let mut check = || task.driver.checkpoint().map_err(Int8SourceMapError::from);
        result.result_binding = source_map_result_binding(&result, self.max_result_bytes(), &mut check)?;
        self.expected.verify_completed_with_checkpoints(&result, &mut check)?;
        Ok(result)
    }
}
fn preflight_plans(plans: &[PreparedInt8SourceTask], admitted: &[ExecutionIdentity],
    engine: &StrictInt8Engine<'_>) -> Result<(), Int8SourceMapError> {
    verify_identities(plans, admitted)?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int8SourceMapError::Admission); }
    let capacity = engine.kv_cache().capacity_positions() as u64;
    let resident = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(Int8SourceMapError::Admission)?;
    for (plan, identity) in plans.iter().zip(admitted) {
        constrained_int8::check_profile(identity).map_err(Int8SourceError::from)?;
        let model = engine.artifact_identity();
        if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
            || model.recipe_id != identity.quant_recipe
            || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
            return Err(Int8SourceMapError::Admission);
        }
        if plan.planned_work().forward_positions > capacity || resident > plan.task_budget().max_kv_bytes {
            return Err(Int8SourceMapError::Admission);
        }
    }
    Ok(())
}
fn verify_identities(plans: &[PreparedInt8SourceTask], admitted: &[ExecutionIdentity]) -> Result<(), Int8SourceMapError> {
    if plans.is_empty() || plans.len() != admitted.len() { return Err(Int8SourceMapError::Admission); }
    for (plan, identity) in plans.iter().zip(admitted) { plan.verify_identity(identity)?; }
    Ok(())
}

trait SourceDriver {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError>;
    fn run(&mut self, plan: &PreparedInt8SourceTask, admitted: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError>;
}
struct NativeDriver<'e, 'w, 'v, 'c, C> {
    engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary,
    control: &'c mut C, limits: Int8SourceMapLimits,
    prefill: Option<Int8PrefillLimits>,
}
impl<C: DecodeStepControl> SourceDriver for NativeDriver<'_, '_, '_, '_, C> {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { checkpoint(self.control) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, admitted: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let work = plan.planned_work();
        let budget = Int8JsonBudget {
            native: Int8RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: plan.task_budget().max_kv_bytes,
                max_total_mask_node_visits: self.limits.mask_visits_per_chunk, mask_limits: self.limits.mask_limits,
            },
        };
        let result = match self.prefill {
            None => plan.execute_with_control(admitted, self.engine, self.vocabulary, budget, self.control),
            Some(prefill) => plan.execute_layer_major_with_control(admitted, self.engine,
                self.vocabulary, budget, prefill, self.control),
        }?;
        if self.engine.is_poisoned() || !self.engine.kv_cache().all_slots_have_len(0)
            || mask_charge(&result.result)? > self.limits.mask_visits_per_chunk { return Err(Int8SourceError::InvalidResult); }
        Ok(result)
    }
}
fn check_prefill(prefill: Int8PrefillLimits) -> Result<(), Int8SourceMapError> {
    prefill.validate().map(|_| ()).map_err(|_| Int8SourceMapError::InvalidLimits)
}

struct SourceMapExecutor<'a, 's, D> {
    chunks: &'a ChunkPlan<'s>, plans: &'a [PreparedInt8SourceTask], admitted: &'a [ExecutionIdentity],
    driver: D, next_chunk: usize, work: Int8Work, masks: u64, mask_cap: u64,
}
impl<D: SourceDriver> MapReduceTask for SourceMapExecutor<'_, '_, D> {
    type Value = SourceMapValue;
    type Error = Int8SourceError;
    fn policy(&self) -> ReductionPolicy {
        ReductionPolicy { id: "lossless-source-order-chunk-results-v1", may_discard_information: false }
    }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<SourceMapValue>>, Int8SourceError> {
        let mut results = Vec::new();
        results.try_reserve_exact(chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for chunk in chunks {
            self.driver.checkpoint()?;
            if chunk.id() != self.next_chunk { return Err(Int8SourceError::InvalidResult); }
            let plan = self.plans.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
            let identity = self.admitted.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
            let native = self.driver.run(plan, identity)?;
            check_run(plan, &native, self.mask_cap)?;
            self.work = add_work(self.work, native.model_work).ok_or(Int8SourceError::InvalidResult)?;
            self.masks = self.masks.checked_add(mask_charge(&native.result)?).ok_or(Int8SourceError::InvalidResult)?;
            let original_spans = original_spans(self.chunks, chunk, &native.result, &mut || self.driver.checkpoint())?;
            let value = SourceMapValue { chunks: vec![Arc::new(MappedSourceChunk {
                chunk_id: chunk.id(), source_span: chunk.span(), native, original_spans,
            })] };
            results.push(MapOutput { chunk_id: chunk.id(), value });
            self.next_chunk += 1;
            self.driver.checkpoint()?;
        }
        Ok(results)
    }
    fn reduce(&mut self, input: ReduceInput<'_, SourceMapValue>) -> Result<SourceMapValue, Int8SourceError> {
        self.driver.checkpoint()?;
        let count = input.children.iter().try_fold(0_usize, |n, child| n.checked_add(child.value().len()))
            .ok_or(Int8SourceError::InvalidResult)?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(count).map_err(|_| SourcePlanningError::AllocationRefused)?;
        let mut next = input.children.first().ok_or(Int8SourceError::InvalidResult)?.chunk_range().start;
        for child in input.children {
            if child.chunk_range().start != next { return Err(Int8SourceError::InvalidResult); }
            for chunk in &child.value().chunks {
                self.driver.checkpoint()?;
                if chunk.chunk_id != next { return Err(Int8SourceError::InvalidResult); }
                chunks.push(Arc::clone(chunk)); next += 1;
            }
            if child.chunk_range().end != next { return Err(Int8SourceError::InvalidResult); }
        }
        self.driver.checkpoint()?;
        Ok(SourceMapValue { chunks })
    }
}
fn check_run(plan: &PreparedInt8SourceTask, run: &Int8SourceTaskRun, mask_cap: u64) -> Result<(), Int8SourceError> {
    let (spec, profile, ids, positions, logits) = match &run.result {
        SourceTaskResult::Ner(r) => (&r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits),
        SourceTaskResult::Keyphrases(r) => (&r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits),
        SourceTaskResult::Summarize(r) => (&r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits),
        SourceTaskResult::Answer(r) => (&r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits),
    };
    let expected = constrained_int8::planned_work(plan.prompt_tokens(), ids.len()).map_err(|_| Int8SourceError::InvalidResult)?;
    if run.schema_version != 1 || run.execution != INT8_SOURCE_EXECUTION
        || spec != &plan.execution_identity().task_spec || profile != STRICT_INT8_PROFILE
        || ids.is_empty() || ids.len() > plan.task_budget().max_output_tokens as usize
        || expected != run.model_work || positions != expected.forward_positions || logits != expected.projected_logits
        || !within(run.model_work, plan.planned_work()) || mask_charge(&run.result)? > mask_cap {
        return Err(Int8SourceError::InvalidResult);
    }
    Ok(())
}
fn mask_charge(result: &SourceTaskResult) -> Result<u64, Int8SourceError> {
    match result { SourceTaskResult::Ner(r) => Ok(r.mask_node_visit_charge),
        SourceTaskResult::Keyphrases(r) => Ok(r.mask_node_visit_charge),
        SourceTaskResult::Summarize(r) => Ok(r.mask_node_visit_charge),
        SourceTaskResult::Answer(r) => Ok(r.mask_node_visit_charge) }
}
fn original_spans<F: FnMut() -> Result<(), Int8SourceError>>(plan: &ChunkPlan<'_>, chunk: &SourceChunk<'_>,
    result: &SourceTaskResult, checkpoint: &mut F) -> Result<Vec<OriginalSourceSpans>, Int8SourceError> {
    let mut fields = Vec::new();
    let mut field = |kind, text: &str, spans: &[VerifiedSourceSpan]| -> Result<(), Int8SourceError> {
        checkpoint()?;
        fields.try_reserve(1).map_err(|_| SourcePlanningError::AllocationRefused)?;
        fields.push(OriginalSourceSpans { field: kind, spans: lift_spans(plan, chunk, text, spans, checkpoint)? }); Ok(())
    };
    match result {
        SourceTaskResult::Ner(r) => for (index, entity) in r.entities.iter().enumerate() {
            field(SourceMapField::Entity { index }, &entity.text, &entity.spans)?;
        },
        SourceTaskResult::Keyphrases(r) => for (index, phrase) in r.phrases.iter().enumerate() {
            field(SourceMapField::Keyphrase { index }, &phrase.text, &phrase.spans)?;
        },
        SourceTaskResult::Summarize(r) => for (bullet, value) in r.bullets.iter().enumerate() {
            for (citation, quote) in value.citations.iter().enumerate() {
                field(SourceMapField::SummaryCitation { bullet, citation }, &quote.quote, &quote.spans)?;
            }
        },
        SourceTaskResult::Answer(_) => return Err(Int8SourceError::InvalidResult),
    }
    Ok(fields)
}
fn lift_spans<F: FnMut() -> Result<(), Int8SourceError>>(plan: &ChunkPlan<'_>, chunk: &SourceChunk<'_>,
    text: &str, spans: &[VerifiedSourceSpan], checkpoint: &mut F) -> Result<Vec<VerifiedSourceSpan>, Int8SourceError> {
    if text.is_empty() || spans.is_empty() { return Err(Int8SourceError::InvalidResult); }
    let mut lifted = Vec::new();
    lifted.try_reserve_exact(spans.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
    for span in spans {
        checkpoint()?;
        if chunk.text().get(span.byte_start..span.byte_end) != Some(text) {
            return Err(Int8SourceError::InvalidResult);
        }
        // The planner owns the original coordinates. Scan its exact prefix
        // and selected range once, with the same request control throughout.
        // Preserve a typed native stop instead of laundering it into InvalidSpan.
        let mut stopped = None;
        let original = plan.lift_span_with_checkpoints(chunk.id(), span.byte_start..span.byte_end, || {
            match checkpoint() {
                Ok(()) => Ok(()),
                Err(error) => { stopped = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = stopped { return Err(error); }
        let original = original.map_err(|_| Int8SourceError::InvalidResult)?;
        if original.scalar_start.checked_sub(chunk.span().scalar_start) != Some(span.scalar_start)
            || original.scalar_end.checked_sub(chunk.span().scalar_start) != Some(span.scalar_end) {
            return Err(Int8SourceError::InvalidResult);
        }
        lifted.push(original);
    }
    Ok(lifted)
}
fn within(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.dot_products <= ceiling.projections.dot_products
        && work.projections.multiply_accumulates <= ceiling.projections.multiply_accumulates
}
fn add_work(a: Int8Work, b: Int8Work) -> Option<Int8Work> {
    Some(Int8Work { forward_positions: a.forward_positions.checked_add(b.forward_positions)?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits)?,
        attention_pairs: a.attention_pairs.checked_add(b.attention_pairs)?,
        projections: a.projections.checked_add(b.projections).ok()? })
}

#[cfg(test)] mod tests;
