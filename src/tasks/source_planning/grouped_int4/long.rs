//! Complete-document NER, keyphrases and cited summaries on grouped INT4.
//! Every actual chunk is prepared before the first forward. Independent native
//! results retain original-document lineage; ordered reduction never invents a
//! global entity census, phrase ranking or neural summary synthesis.
use super::*;
use std::{cell::RefCell, sync::Arc};
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, strict_int4::{Int4MemoryRequirement, Int4RunBudget}},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace,
        mapreduce::{self, ChunkLimits, ChunkPlan, ExecutionError, ExecutionLimits, MapOutput,
            MapReduceError, MapReduceResult, MapReduceTask, ReduceInput, ReductionPolicy, SourceChunk}},
    validation::grounded_fields::{FieldGroundingError, GroundingBudget, SourceOccurrence,
        VerifiedSourceSpan, scan_occurrences},
};
mod execution;
/// Shared native decoder layers over independently planned source chunks.
pub mod cohort;
pub mod summary;
/// Complete document-wide exact keyphrase union and deterministic final ranking.
pub mod keyphrases;
/// Question-aware passage planning and complete original-source answer collection.
pub mod question;

pub const INT4_SOURCE_MAP_EXECUTION: &str = "grouped-int4-source-document-map-v1";
const SEMANTICS: &str = "independent-chunks-no-cross-chunk-reasoning-v1";

/// Options apply to every chunk. Passage QA has a different question-aware
/// capacity and evidence contract and is deliberately not a source-only map.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "task", content = "options", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceMapTask { Ner(NerOptions), Keyphrases(KeyphraseOptions), Summarize(SummaryOptions) }
impl SourceMapTask {
    fn task_spec(&self) -> &'static str {
        match self { Self::Ner(_) => NER_TASK_VERSION, Self::Keyphrases(_) => KEYPHRASES_TASK_VERSION,
            Self::Summarize(_) => SUMMARIZE_TASK_VERSION }
    }
    fn request(&self, text: String, budget: TaskBudget) -> SourceTaskRequest {
        match self {
            Self::Ner(options) => SourceTaskRequest::Ner { document: text, options: options.clone(), budget },
            Self::Keyphrases(options) => SourceTaskRequest::Keyphrases { document: text, options: *options, budget },
            Self::Summarize(options) => SourceTaskRequest::Summarize { document: text, options: *options, budget },
        }
    }
}

/// Whole-document allowances, never renewed per chunk, merge or native call.
#[derive(Clone, Copy, Debug)]
pub struct Int4SourceMapLimits {
    pub chunks: ChunkLimits,
    pub reduction: ExecutionLimits,
    pub max_model_work: Int4Work,
    pub mask_limits: MaskWorkLimits,
    pub mask_visits_per_chunk: u64,
    pub max_mask_visits: u64,
    /// Independent original-coordinate verification across the whole source.
    pub verification: GroundingBudget,
}
impl Int4SourceMapLimits {
    fn validate(self, budget: TaskBudget, planning: SourcePlanningLimits) -> Result<(), Int4SourceMapError> {
        self.chunks.effective_token_limit().map_err(Int4SourceMapError::Chunk)?;
        budget.validate().map_err(|_| Int4SourceMapError::InvalidLimits)?;
        let v = self.verification;
        if !(1..=256).contains(&self.chunks.max_chunks)
            || self.chunks.max_chunk_bytes > planning.max_input_bytes
            || self.chunks.context_tokens > planning.max_context_tokens
            || self.chunks.reserved_tokens < budget.max_output_tokens as usize
            || self.mask_visits_per_chunk == 0 || self.max_mask_visits < self.mask_visits_per_chunk
            || self.mask_limits.max_trie_node_visits == 0 || self.mask_limits.checkpoint_interval_nodes == 0
            || !positive_work(self.max_model_work) || !valid_reduction(self.reduction)
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || v.max_scan_steps == 0 {
            return Err(Int4SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}

pub enum Int4SourceMapError {
    InvalidLimits, EmptySource, WorkLimit, Admission, Allocation,
    Chunk(MapReduceError), Source(Int4SourceError), Reduction(ExecutionError<Int4SourceError>),
}
impl fmt::Display for Int4SourceMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid INT4 source document limits",
            Self::EmptySource => "INT4 source document requires nonempty input",
            Self::WorkLimit => "INT4 source whole-document allowance exceeded",
            Self::Admission => "INT4 source document identity or resident capacity refused",
            Self::Allocation => "INT4 source document allocation refused",
            Self::Chunk(_) => "INT4 source document partition refused",
            Self::Source(_) => "INT4 source document native task refused",
            Self::Reduction(_) => "INT4 source document reduction refused",
        })
    }
}
impl fmt::Debug for Int4SourceMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4SourceMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Source(e) => Some(e), Self::Reduction(e) => Some(e), _ => None }
    }
}
impl From<Int4SourceError> for Int4SourceMapError { fn from(e: Int4SourceError) -> Self { Self::Source(e) } }
impl Int4SourceMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Source(e) | Self::Reduction(ExecutionError::Task { source: e, .. }) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    group_width: usize,
    task_spec: &'static str,
    work: Int4Work,
    masks: u64,
    verification: GroundingBudget,
}
impl Int4SourceMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn verify_completed(&self, run: &Int4SourceMapRun) -> Result<(), Int4SourceMapError> {
        let root = run.mapped.root();
        let (work, masks, fields, matches) = execution::statistics(root.value())?;
        if run.schema_version != 1 || run.execution != INT4_SOURCE_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.task_spec != self.task_spec || run.semantics != SEMANTICS
            || run.untrusted_fields != ["mapped.root.value"] || root.source_span() != self.source_span
            || root.chunk_range() != (0..self.chunks) || root.value().len() != self.chunks
            || run.planned_model_work != self.work || run.model_work != work || !fits(work, self.work)
            || run.reserved_mask_node_visits != self.masks || run.mask_node_visit_charge != masks || masks > self.masks
            || run.verification_used.fields != fields || run.verification_used.matches != matches
            || !run.verification_used.fits(self.verification) {
            return Err(Int4SourceError::InvalidResult.into());
        }
        let mut next = VerifiedSourceSpan { byte_start: 0, byte_end: 0, scalar_start: 0, scalar_end: 0 };
        for (index, chunk) in root.value().chunks().enumerate() {
            if chunk.chunk_id != index || chunk.source_span.byte_start != next.byte_end
                || chunk.source_span.scalar_start != next.scalar_end || chunk.source_span.byte_end <= chunk.source_span.byte_start
                || chunk.source_span.scalar_end <= chunk.source_span.scalar_start
                || chunk.native.group_width != self.group_width {
                return Err(Int4SourceError::InvalidResult.into());
            }
            next = chunk.source_span;
        }
        if next.byte_end != self.source_span.byte_end || next.scalar_end != self.source_span.scalar_end {
            return Err(Int4SourceError::InvalidResult.into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceMapField { Entity { index: usize }, Keyphrase { index: usize }, SummaryCitation { bullet: usize, citation: usize } }
#[derive(Serialize)]
pub struct OriginalSourceSpans { pub field: SourceMapField, pub spans: Vec<VerifiedSourceSpan> }
#[derive(Serialize)]
pub struct MappedSourceChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Unchanged native result, with CHUNK-LOCAL source coordinates.
    pub native: Int4SourceTaskRun,
    /// All compatible occurrences within this producing chunk, lifted into
    /// the original document. No matches in other chunks are inferred.
    pub original_spans: Vec<OriginalSourceSpans>,
}
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SourceMapVerification { pub fields: usize, pub matches: usize, pub scan_steps: u64 }
impl SourceMapVerification {
    fn used(total: GroundingBudget, remaining: GroundingBudget) -> Result<Self, Int4SourceError> {
        Ok(Self { fields: total.max_fields.checked_sub(remaining.max_fields).ok_or(Int4SourceError::InvalidResult)?,
            matches: total.max_matches.checked_sub(remaining.max_matches).ok_or(Int4SourceError::InvalidResult)?,
            scan_steps: total.max_scan_steps.checked_sub(remaining.max_scan_steps).ok_or(Int4SourceError::InvalidResult)? })
    }
    fn fits(self, limit: GroundingBudget) -> bool {
        self.fields <= limit.max_fields && self.matches <= limit.max_matches && self.scan_steps <= limit.max_scan_steps
    }
}
#[derive(Serialize)]
pub struct Int4SourceMapRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub task_spec: &'static str,
    pub semantics: &'static str,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: SourceMapVerification,
    pub untrusted_fields: [&'static str; 1],
    pub mapped: MapReduceResult<SourceMapValue>,
}

pub struct PreparedInt4SourceMap<'s> {
    chunks: ChunkPlan<'s>,
    plans: Vec<PreparedInt4SourceTask>,
    limits: Int4SourceMapLimits,
    expected: Int4SourceMapPreflight,
    kv_ceiling: u64,
}
impl PreparedInt4SourceMap<'_> {
    pub fn preflight_metadata(&self) -> Int4SourceMapPreflight { self.expected }
    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.plans.iter().map(PreparedInt4SourceTask::execution_identity)
    }
    pub fn planned_work(&self) -> Int4Work { self.expected.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.expected.masks }
    pub fn chunk_count(&self) -> usize { self.plans.len() }
    pub fn group_width(&self) -> usize { self.expected.group_width }
    pub fn max_kv_bytes(&self) -> u64 { self.kv_ceiling }
    pub fn max_result_bytes(&self) -> u64 { self.limits.reduction.max_result_bytes as u64 }
}
impl SourceTaskPlanner {
    /// Bind every real prompt/schema/source plan before execution. Automatic
    /// capacity uses the exact trusted scaffold and pinned source tokenizer.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int4_map_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        task: &SourceMapTask, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, limits: Int4SourceMapLimits, group_width: usize, control: &mut C)
        -> Result<PreparedInt4SourceMap<'s>, Int4SourceMapError> {
        checkpoint(control)?;
        limits.validate(budget, planning)?;
        if source.is_empty() { return Err(Int4SourceMapError::EmptySource); }
        let capacity = self.int4_map_capacity_with_control(task, budget, context, planning, group_width, control)?;
        let chunk_limits = capacity.constrain_chunks(limits.chunks)?;
        let mut cancellation = None;
        let chunks = ChunkPlan::build_with_checkpoints(source, chunk_limits, |text| {
            self.encoder.encode(text, chunk_limits.max_chunk_bytes, chunk_limits.max_chunk_bytes)
                .map(|document| document.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
        }, || match control.prefill_checkpoint(0) {
            Some(cause) => { cancellation = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
        });
        if let Some(cause) = cancellation { return Err(Int4SourceError::Cancelled(cause).into()); }
        let chunks = chunks.map_err(Int4SourceMapError::Chunk)?;
        let first = chunks.chunks().first().ok_or(Int4SourceMapError::EmptySource)?.span();
        let last = chunks.chunks().last().ok_or(Int4SourceMapError::EmptySource)?.span();
        let mut expected = Int4SourceMapPreflight {
            source_span: VerifiedSourceSpan { byte_start: first.byte_start, byte_end: last.byte_end,
                scalar_start: first.scalar_start, scalar_end: last.scalar_end },
            chunks: chunks.chunks().len(), group_width, task_spec: task.task_spec(),
            work: Int4Work::default(), masks: 0, verification: limits.verification,
        };
        let mut plans = Vec::new();
        plans.try_reserve_exact(expected.chunks).map_err(|_| Int4SourceMapError::Allocation)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let mut text = String::new();
            text.try_reserve_exact(chunk.text().len()).map_err(|_| Int4SourceMapError::Allocation)?;
            text.push_str(chunk.text());
            let request = task.request(text, budget);
            let plan = self.plan_int4_with_control(&request, context, planning, group_width, control)?;
            if plan.execution_identity().task_spec != expected.task_spec || plan.group_width() != group_width {
                return Err(Int4SourceMapError::Admission);
            }
            expected.work = add_work(expected.work, plan.planned_work()).filter(|work| fits(*work, limits.max_model_work))
                .ok_or(Int4SourceMapError::WorkLimit)?;
            expected.masks = expected.masks.checked_add(limits.mask_visits_per_chunk)
                .filter(|&masks| masks <= limits.max_mask_visits).ok_or(Int4SourceMapError::WorkLimit)?;
            plans.push(plan);
        }
        checkpoint(control)?;
        Ok(PreparedInt4SourceMap { chunks, plans, limits, expected, kv_ceiling: budget.max_kv_bytes })
    }
}
fn add_work(a: Int4Work, b: Int4Work) -> Option<Int4Work> { a.checked_add(b).ok() }
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
