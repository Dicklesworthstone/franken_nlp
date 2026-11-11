//! Complete native sentiment maps with a named flat document estimator.
//!
//! Every source chunk is prepared and admitted before the first forward.
//! Original per-axis finalizers remain authoritative. Complete chunk results
//! are retained in source order and aggregated once without hierarchical
//! copying, cross-chunk context, calibrated confidence or hidden truncation.
use super::*;
use std::cell::RefCell;
use sha2::{Digest, Sha256};
use crate::{native_engine::{int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights,
            cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
            scoring::cohort::{Int4ScoringCohortBudget, required_task_workspace_bytes}}},
    tasks::{mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError},
        sentiment::{SentimentAggregate, SentimentAggregationPolicy,
            aggregation::{SentimentAggregationError, SentimentChunkView, aggregate_with_control}}},
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding},
    validation::grounded_fields::VerifiedSourceSpan};

mod execution;
mod binding;
pub mod memory;
pub const INT4_SENTIMENT_MAP_EXECUTION: &str = "grouped-int4-complete-sentiment-map-v1";
pub const INT4_SENTIMENT_MAP_COHORT_EXECUTION: &str = "grouped-int4-sentiment-chunk-cohorts-v1";
pub const MAX_SENTIMENT_MAP_CHUNKS: usize = 256;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const SEMANTICS: &str = "independent-chunks-flat-source-scalar-mixture-v1";
const COHORT_WRAPPER_BYTES: u64 = 4096;

#[derive(Clone, Copy, Debug)]
pub struct Int4SentimentMapLimits {
    pub chunks: ChunkLimits,
    /// All five native counters cover the complete document, never one chunk.
    pub max_model_work: Int4Work,
    /// Cumulative canonical bytes of retained complete chunk records.
    pub max_retained_result_bytes: u64,
    /// Complete final map envelope, including chunk evidence and aggregate.
    pub max_result_bytes: u64,
    pub aggregation: SentimentAggregationPolicy,
}
impl Int4SentimentMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int4SentimentMapError> {
        self.chunks.effective_token_limit().map_err(Int4SentimentMapError::Chunk)?;
        task.validate().map_err(|_| Int4SentimentMapError::Limits)?;
        self.aggregation.validate().map_err(|_| Int4SentimentMapError::Limits)?;
        if !(1..=MAX_SENTIMENT_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || !positive_work(self.max_model_work) { return Err(Int4SentimentMapError::Limits); }
        Ok(())
    }
}

pub enum Int4SentimentMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization,
    Chunk(MapReduceError), Sentiment(Int4SentimentError), Aggregation,
}
impl fmt::Display for Int4SentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "invalid native sentiment document limits",
            Self::EmptySource => "native sentiment document requires nonempty source",
            Self::Work => "native sentiment whole-document work allowance exceeded",
            Self::Admission => "native sentiment document identity or capacity differs",
            Self::Allocation => "native sentiment document allocation refused",
            Self::Output => "complete native sentiment document output exceeds its allowance",
            Self::Serialization => "native sentiment document serialization refused",
            Self::Chunk(_) => "native sentiment document partition refused",
            Self::Sentiment(_) => "native sentiment document task execution refused",
            Self::Aggregation => "native sentiment complete document aggregation refused",
        })
    }
}
impl fmt::Debug for Int4SentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4SentimentMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(error) => Some(error), Self::Sentiment(error) => Some(error), _ => None }
    }
}
impl From<Int4SentimentError> for Int4SentimentMapError { fn from(error: Int4SentimentError) -> Self { Self::Sentiment(error) } }
impl From<SentimentPlanningError> for Int4SentimentMapError {
    fn from(error: SentimentPlanningError) -> Self { Self::Sentiment(error.into()) }
}
impl From<StrictInt4Error> for Int4SentimentMapError { fn from(error: StrictInt4Error) -> Self { Self::Sentiment(error.into()) } }
impl From<Int4ScoringError> for Int4SentimentMapError { fn from(error: Int4ScoringError) -> Self { Self::Sentiment(error.into()) } }
impl From<SentimentAggregationError> for Int4SentimentMapError {
    fn from(error: SentimentAggregationError) -> Self {
        match error {
            SentimentAggregationError::Cancelled(cause) => Self::Sentiment(Int4SentimentError::Cancelled(cause)),
            SentimentAggregationError::Allocation => Self::Allocation,
            _ => Self::Aggregation,
        }
    }
}
impl Int4SentimentMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Sentiment(error) => error.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int4SentimentMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Exact document tokens under the admitted encoding, excluding the
    /// fixed trusted prompt and five-bin candidate continuations.
    pub source_tokens: usize,
    pub run: Int4SentimentRun,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Int4SentimentMapEncoding {
    pub policy: UntrustedEncoding,
    /// Every attempted chunk and actual chunk plan shares this ledger. The
    /// encoded document is reused across its independent sentiment axes.
    pub work: SafeBpeWork,
}
/// Complete native output only; no caller-supplied result constructor enters
/// the executable path. The aggregate never replaces the original chunks.
#[derive(Clone, PartialEq, Serialize)]
pub struct Int4SentimentMapRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub group_width: usize,
    pub chunk_profile: String,
    pub semantics: String,
    pub single_context_equivalence: bool,
    pub source_span: VerifiedSourceSpan,
    pub chunk_limits: ChunkLimits,
    pub tokenizer_checks: usize,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub retained_result_bytes: u64,
    pub chunks: Vec<Int4SentimentMapChunk>,
    pub aggregate: SentimentAggregate,
    /// Absent for the unchanged byte-fallback contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int4SentimentMapEncoding>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int4SentimentMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4SentimentMapRun")
            .field("schema_version", &self.schema_version).field("execution", &self.execution)
            .field("numerics_profile", &self.numerics_profile).field("group_width", &self.group_width)
            .field("chunk_profile", &self.chunk_profile).field("semantics", &self.semantics)
            .field("single_context_equivalence", &self.single_context_equivalence)
            .field("source_span", &self.source_span).field("chunk_limits", &self.chunk_limits)
            .field("tokenizer_checks", &self.tokenizer_checks).field("planned_model_work", &self.planned_model_work)
            .field("model_work", &self.model_work).field("retained_result_bytes", &self.retained_result_bytes)
            .field("chunks", &self.chunks).field("aggregate", &self.aggregate)
            .field("input_encoding", &self.input_encoding).finish()
    }
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Int4SentimentMapCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub chunk_cohorts: usize,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub output: Int4SentimentMapRun,
    #[serde(skip)]
    executed_cohort_chunks: usize,
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int4SentimentMapCohortRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4SentimentMapCohortRun")
            .field("schema_version", &self.schema_version).field("execution", &self.execution)
            .field("chunk_cohorts", &self.chunk_cohorts).field("group_steps", &self.group_steps)
            .field("projection_groups", &self.projection_groups).field("output", &self.output).finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int4SentimentMapCohortBudget {
    pub native: Int4RunBudget,
    pub memory: Int4MemoryBudget,
    pub max_task_workspace_bytes: u64,
    pub max_output_bytes: u64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int4SentimentMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    heads: usize,
    group_width: usize,
    work: Int4Work,
    max_retained_bytes: u64,
    max_result_bytes: u64,
    aggregation: SentimentAggregationPolicy,
    chunk_limits: ChunkLimits,
    tokenizer_checks: usize,
    source_geometry: [u8; 32],
    encoding: Option<SafeBpeLimits>,
    encoding_work: SafeBpeWork,
    request_binding: [u8; 32],
}
impl fmt::Debug for Int4SentimentMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4SentimentMapPreflight")
            .field("source_span", &self.source_span).field("chunks", &self.chunks)
            .field("heads", &self.heads).field("group_width", &self.group_width).field("work", &self.work)
            .field("max_retained_bytes", &self.max_retained_bytes).field("max_result_bytes", &self.max_result_bytes)
            .field("aggregation", &self.aggregation).field("chunk_limits", &self.chunk_limits)
            .field("tokenizer_checks", &self.tokenizer_checks).field("encoding", &self.encoding)
            .field("encoding_work", &self.encoding_work).finish_non_exhaustive()
    }
}
impl Int4SentimentMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.max_result_bytes }
    fn encoding_record(&self) -> Option<Int4SentimentMapEncoding> {
        self.encoding.map(|limits| Int4SentimentMapEncoding {
            policy: UntrustedEncoding::SafeBpeV1(limits), work: self.encoding_work })
    }
    /// Verify private request association and the unchanged native payload.
    /// Neither a geometric match nor external JSON can mint these seals.
    pub fn verify_completed(&self, run: &Int4SentimentMapRun) -> Result<(), Int4SentimentMapError> {
        self.verify_completed_with_control(run, &mut binding::Continue)
    }
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int4SentimentMapRun,
        control: &mut C) -> Result<(), Int4SentimentMapError> {
        checkpoint(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != binding::result_with_control(run, self.max_result_bytes, control)? {
            return Err(Int4SentimentMapError::Admission);
        }
        if run.schema_version != 1 || run.execution != INT4_SENTIMENT_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.chunk_profile != CHUNK_PROFILE || run.semantics != SEMANTICS || run.single_context_equivalence
            || run.source_span != self.source_span || run.chunks.len() != self.chunks
            || run.planned_model_work != self.work || run.model_work != self.work
            || run.aggregate.policy != self.aggregation || run.aggregate.chunk_count != self.chunks
            || run.aggregate.source_bytes != self.source_span.byte_end
            || run.aggregate.source_scalars != self.source_span.scalar_end
            || run.aggregate.dimensions.len() != self.heads || run.retained_result_bytes > self.max_retained_bytes
            // serde_json maps nonfinite floats to null, just like None.
            // The private byte seal therefore needs this typed check too.
            || run.aggregate.dimensions.iter().any(|axis| axis.estimate.is_some_and(|value| !value.is_finite())) {
            return Err(Int4SentimentMapError::Admission);
        }
        if run.chunk_limits != self.chunk_limits || run.tokenizer_checks != self.tokenizer_checks
            || run.input_encoding != self.encoding_record()
            || geometry(run.chunks.iter().map(|chunk| (chunk.chunk_id, chunk.source_span, chunk.source_tokens))) != self.source_geometry {
            return Err(Int4SentimentMapError::Admission);
        }
        let mut byte = 0; let mut scalar = 0; let mut work = Int4Work::default(); let mut retained = 0_u64;
        for (index, chunk) in run.chunks.iter().enumerate() {
            checkpoint(control)?;
            let span = chunk.source_span;
            if chunk.chunk_id != index || span.byte_start != byte || span.scalar_start != scalar
                || span.byte_end <= span.byte_start || span.scalar_end <= span.scalar_start
                || chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != span.byte_end - span.byte_start)
                || chunk.run.schema_version != 1 || chunk.run.execution != INT4_SENTIMENT_EXECUTION
                || chunk.run.numerics_profile != STRICT_INT4_PROFILE || chunk.run.group_width != self.group_width
                || chunk.run.head_count != self.heads || chunk.run.result.dimensions.len() != self.heads
                || chunk.run.result.dimensions.iter().any(|axis| axis.estimate.is_some_and(|value| !value.is_finite())) {
                return Err(Int4SentimentMapError::Admission);
            }
            byte = span.byte_end; scalar = span.scalar_end;
            work = work.checked_add(chunk.run.model_work)?;
            retained = retained.checked_add(execution::value_bytes(chunk, self.max_retained_bytes)?)
                .filter(|&bytes| bytes <= self.max_retained_bytes).ok_or(Int4SentimentMapError::Output)?;
        }
        if byte != self.source_span.byte_end || scalar != self.source_span.scalar_end
            || work != self.work || retained != run.retained_result_bytes { return Err(Int4SentimentMapError::Admission); }
        execution::value_bytes(run, self.max_result_bytes)?; checkpoint(control)
    }
    /// Cohort diagnostics belong to this exact completed request as well as
    /// its semantic output; valid-looking replacement counters are refused.
    pub fn verify_completed_cohorts(&self, run: &Int4SentimentMapCohortRun, cohort_chunks: usize)
        -> Result<(), Int4SentimentMapError> {
        self.verify_completed_cohorts_with_control(run, cohort_chunks, &mut binding::Continue)
    }
    pub fn verify_completed_cohorts_with_control<C: DecodeStepControl>(&self, run: &Int4SentimentMapCohortRun,
        cohort_chunks: usize, control: &mut C) -> Result<(), Int4SentimentMapError> {
        checkpoint(control)?;
        let cap = self.max_result_bytes.checked_add(COHORT_WRAPPER_BYTES).ok_or(Int4SentimentMapError::Limits)?;
        if run.result_binding != binding::cohort_with_control(run, cap, control)?
            || cohort_chunks == 0 || cohort_chunks > self.chunk_limits.max_chunks || self.heads == 0
            || cohort_chunks.checked_mul(self.heads).is_none_or(|heads| heads > MAX_INT4_SEQUENCES)
            || run.executed_cohort_chunks != cohort_chunks.min(self.chunks)
            || run.schema_version != 1 || run.execution != INT4_SENTIMENT_MAP_COHORT_EXECUTION
            || run.chunk_cohorts != self.chunks.div_ceil(cohort_chunks)
            || run.group_steps == 0 || run.group_steps > self.work.forward_positions
            || run.projection_groups == 0 || run.projection_groups > self.work.projected_logits {
            return Err(Int4SentimentMapError::Admission);
        }
        self.verify_completed_with_control(&run.output, control)?;
        execution::value_bytes(run, cap)?;
        checkpoint(control)
    }
}

/// Borrows the immutable original source and retains every actual chunk plan.
/// No wire constructor, supplied scorer or public driver can fabricate a map.
pub struct PreparedInt4SentimentMap<'a> {
    chunks: ChunkPlan<'a>,
    plans: Vec<PreparedInt4Sentiment>,
    limits: Int4SentimentMapLimits,
    expected: Int4SentimentMapPreflight,
    task: TaskBudget,
}
impl Int4SentimentPlanner {
    pub fn plan_document_with_control<'a, C: DecodeStepControl>(&self, request: &'a SentimentRequest,
        limits: Int4SentimentMapLimits, control: &mut C) -> Result<PreparedInt4SentimentMap<'a>, Int4SentimentMapError> {
        checkpoint(control)?; limits.validate(request.budget)?;
        if request.document.is_empty() { return Err(Int4SentimentMapError::EmptySource); }
        let capacity = self.source_token_capacity(&request.axes, request.budget, limits.chunks.context_tokens)?;
        PreparedInt4SentimentMap::compile_encoded(request, limits, capacity, self.group_width(),
            binding::Authority { ceiling: self.task_budget(), planning: self.planning_limits() },
            self.document_encoding(), SafeBpeWork::default(), control,
            |source, control, work| self.source_tokens_with_work(source, work, control),
            |request, control, work| self.plan_with_encoding_work(request, control, work))
    }
}
impl<'a> PreparedInt4SentimentMap<'a> {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn compile<C: DecodeStepControl, F>(request: &'a SentimentRequest, limits: Int4SentimentMapLimits,
        capacity: usize, width: usize, authority: binding::Authority,
        control: &mut C, mut prepare: F) -> Result<Self, Int4SentimentMapError>
    where F: FnMut(&SentimentRequest, &mut C) -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        Self::compile_encoded(request, limits, capacity, width, authority,
            &UntrustedEncoding::ByteFallback, SafeBpeWork::default(), control,
            |source, _, _| Ok(source.len()), |request, control, _| prepare(request, control))
    }
    #[allow(clippy::too_many_arguments)]
    fn compile_encoded<C: DecodeStepControl, Count, Prepare>(request: &'a SentimentRequest,
        limits: Int4SentimentMapLimits, capacity: usize, width: usize, authority: binding::Authority,
        encoding: &UntrustedEncoding, mut work: SafeBpeWork, control: &mut C, mut count: Count, mut prepare: Prepare)
        -> Result<Self, Int4SentimentMapError>
    where Count: FnMut(&str, &mut C, &mut SafeBpeWork) -> Result<usize, SentimentPlanningError>,
        Prepare: FnMut(&SentimentRequest, &mut C, &mut SafeBpeWork) -> Result<PreparedInt4Sentiment, Int4SentimentError> {
        encoding.validate().map_err(|_| Int4SentimentMapError::Limits)?;
        limits.validate(request.budget)?;
        if request.document.is_empty() { return Err(Int4SentimentMapError::EmptySource); }
        if capacity == 0 { return Err(Int4SentimentMapError::Limits); }
        let reservation = limits.chunks.context_tokens.checked_sub(capacity).ok_or(Int4SentimentMapError::Limits)?;
        let chunks_limits = ChunkLimits { max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reservation), ..limits.chunks };
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints(&request.document, chunks_limits,
                |text| count(text, &mut **control.borrow_mut(), &mut work).map_err(|cause| {
                    *error.borrow_mut() = Some(cause); MapReduceError::Tokenizer
                }),
                || match control.borrow_mut().prefill_checkpoint(0) {
                    Some(cause) => {
                        *error.borrow_mut() = Some(SentimentPlanningError::Cancelled(cause));
                        Err(MapReduceError::Cancelled)
                    }, None => Ok(()),
                })
        };
        if let Some(error) = error.into_inner() { return Err(error.into()); }
        let chunks = chunks.map_err(Int4SentimentMapError::Chunk)?;
        let end = chunks.chunks().last().ok_or(Int4SentimentMapError::EmptySource)?.span();
        let mut expected = Int4SentimentMapPreflight { source_span: VerifiedSourceSpan { byte_start: 0,
            byte_end: end.byte_end, scalar_start: 0, scalar_end: end.scalar_end }, chunks: chunks.chunks().len(),
            heads: request.axes.len(), group_width: width, work: Int4Work::default(),
            max_retained_bytes: limits.max_retained_result_bytes, max_result_bytes: limits.max_result_bytes,
            aggregation: limits.aggregation, chunk_limits: chunks_limits, tokenizer_checks: chunks.tokenizer_calls(),
            source_geometry: geometry(chunks.chunks().iter().map(|chunk| (chunk.id(), chunk.span(), chunk.tokens()))),
            encoding: match encoding { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) }, encoding_work: work,
            request_binding: [0; 32] };
        let mut plans = reserved(expected.chunks)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let mut text = String::new(); text.try_reserve_exact(chunk.text().len()).map_err(|_| Int4SentimentMapError::Allocation)?;
            text.push_str(chunk.text());
            let row = SentimentRequest { document: text, axes: request.axes.clone(), budget: request.budget };
            let plan = prepare(&row, control, &mut work)?;
            if plan.group_width() != width || plan.head_count() != expected.heads || plan.task_budget() != request.budget
                || plan.source_token_count()? != chunk.tokens()
                || plan.required_context() > chunks_limits.context_tokens { return Err(Int4SentimentMapError::Admission); }
            expected.work = expected.work.checked_add(plan.planned_work())?;
            if !fits(expected.work, limits.max_model_work) { return Err(Int4SentimentMapError::Work); }
            plans.push(plan);
        }
        expected.encoding_work = work;
        expected.request_binding = binding::request(request, limits, authority, encoding, work, &chunks, &plans, control)?;
        checkpoint(control)?;
        Ok(Self { chunks, plans, limits, expected, task: request.budget })
    }
    pub fn preflight_metadata(&self) -> Int4SentimentMapPreflight { self.expected }
    pub fn identities(&self) -> Result<Vec<ExecutionIdentity>, Int4SentimentMapError> {
        let mut identities = reserved(self.plans.len())?;
        identities.extend(self.plans.iter().map(|plan| plan.execution_identity().clone())); Ok(identities)
    }
    pub fn planned_work(&self) -> Int4Work { self.expected.work }
    pub fn group_width(&self) -> usize { self.expected.group_width }
    /// Number of independent axes in each chunk, not the whole map head sum.
    pub fn head_count(&self) -> usize { self.expected.heads }
    pub fn chunk_count(&self) -> usize { self.expected.chunks }
    pub fn required_context(&self) -> usize { self.plans.iter().map(PreparedInt4Sentiment::required_context).max().unwrap_or(0) }
    pub fn max_result_bytes(&self) -> u64 { self.limits.max_result_bytes }
    pub fn task_budget(&self) -> TaskBudget { self.task }
}

fn geometry(rows: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"sentiment-map-source-token-geometry-v1");
    for (id, span, tokens) in rows {
        for value in [id, span.byte_start, span.byte_end, span.scalar_start, span.scalar_end, tokens] {
            hash.update((value as u64).to_le_bytes());
        }
    }
    hash.finalize().into()
}

fn positive_work(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

#[cfg(test)] mod tests;
