//! Complete-document classification using the original exact native heads.
//! Serial execution reuses one cache; explicit cohorts keep independent
//! chunk/head contexts. Every plan and all five work axes are admitted before
//! any native forward, and both schedules use the same complete finalizer.
use super::*;
use std::cell::RefCell;
use sha2::{Digest, Sha256};
use crate::{tasks::{classify::{ClassificationMode, ClassificationPolicy, ClassificationAggregate,
        ClassificationAggregateResult, ClassificationAggregationPolicy, ClassificationAggregationError,
        aggregation::{ClassificationChunkView, aggregate_with_control}},
    mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError}},
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding},
    validation::grounded_fields::VerifiedSourceSpan};

mod execution;
pub mod memory;
mod cohort;
pub use cohort::{Int4ClassificationMapCohortBudget, Int4ClassificationMapCohortRun,
    INT4_CLASSIFICATION_MAP_COHORT_EXECUTION};
pub const INT4_CLASSIFICATION_MAP_EXECUTION: &str = "grouped-int4-complete-classification-map-v1";
pub const MAX_CLASSIFICATION_MAP_CHUNKS: usize = 256;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const SEMANTICS: &str = "independent-chunks-flat-source-scalar-mixture-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int4ClassificationMapLimits {
    pub chunks: ChunkLimits,
    /// Complete-document allowance, never replenished per chunk or label.
    pub max_model_work: Int4Work,
    pub max_retained_result_bytes: u64,
    pub max_result_bytes: u64,
    pub aggregation: ClassificationAggregationPolicy,
}
impl Int4ClassificationMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int4ClassificationMapError> {
        self.chunks.effective_token_limit().map_err(Int4ClassificationMapError::Chunk)?;
        task.validate().map_err(|_| Int4ClassificationMapError::Limits)?;
        self.aggregation.validate()?;
        if !(1..=MAX_CLASSIFICATION_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || self.max_model_work.forward_positions == 0 || self.max_model_work.projected_logits == 0
            || self.max_model_work.attention_pairs == 0 || self.max_model_work.projections.dot_products == 0
            || self.max_model_work.projections.multiply_accumulates == 0 {
            return Err(Int4ClassificationMapError::Limits);
        }
        Ok(())
    }
}

pub enum Int4ClassificationMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization,
    Chunk(MapReduceError), Classification(Int4ClassificationError), Aggregation(ClassificationAggregationError),
}
impl fmt::Display for Int4ClassificationMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "invalid native classification document limits",
            Self::EmptySource => "native classification document requires nonempty source",
            Self::Work => "native classification whole-document work allowance exceeded",
            Self::Admission => "native classification document identity or capacity differs",
            Self::Allocation => "native classification document allocation refused",
            Self::Output => "complete native classification document exceeds its output allowance",
            Self::Serialization => "native classification document serialization refused",
            Self::Chunk(_) => "native classification document partition refused",
            Self::Classification(_) => "native classification document task refused",
            Self::Aggregation(_) => "native classification complete document aggregation refused",
        })
    }
}
impl fmt::Debug for Int4ClassificationMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4ClassificationMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Classification(e) => Some(e),
            Self::Aggregation(e) => Some(e), _ => None }
    }
}
impl From<Int4ClassificationError> for Int4ClassificationMapError {
    fn from(e: Int4ClassificationError) -> Self { Self::Classification(e) }
}
impl From<ClassificationPlanningError> for Int4ClassificationMapError {
    fn from(e: ClassificationPlanningError) -> Self { Self::Classification(e.into()) }
}
impl From<StrictInt4Error> for Int4ClassificationMapError {
    fn from(e: StrictInt4Error) -> Self { Self::Classification(e.into()) }
}
impl From<Int4ScoringError> for Int4ClassificationMapError {
    fn from(e: Int4ScoringError) -> Self { Self::Classification(e.into()) }
}
impl From<ClassificationAggregationError> for Int4ClassificationMapError {
    fn from(e: ClassificationAggregationError) -> Self { Self::Aggregation(e) }
}
impl Int4ClassificationMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Classification(e) => e.cancellation(),
            Self::Aggregation(ClassificationAggregationError::Cancelled(cause)) => Some(*cause), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int4ClassificationMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Exact source-token count under the admitted encoding, excluding the
    /// trusted prompt and the separately encoded label data.
    pub source_tokens: usize,
    pub run: Int4ClassificationRun,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Int4ClassificationMapEncoding {
    pub policy: UntrustedEncoding,
    /// Includes label-capacity preparation, every attempted source chunk and
    /// every actual chunk/head plan. No probe, chunk or label renews it.
    pub work: SafeBpeWork,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int4ClassificationMapRun {
    pub schema_version: u32, pub execution: String, pub numerics_profile: String, pub group_width: usize,
    pub chunk_profile: String, pub semantics: String, pub single_context_equivalence: bool,
    pub source_span: VerifiedSourceSpan, pub chunk_limits: ChunkLimits, pub tokenizer_checks: usize,
    pub planned_model_work: Int4Work, pub model_work: Int4Work, pub retained_result_bytes: u64,
    pub chunks: Vec<Int4ClassificationMapChunk>,
    pub aggregate: ClassificationAggregate,
    /// Absent for the unchanged byte-fallback contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int4ClassificationMapEncoding>,
}

/// Content-free geometry from the sealed complete plan. It is not an
/// authenticator for external JSON or a substitute for native execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ClassificationMapPreflight {
    source: VerifiedSourceSpan, chunks: usize, heads: usize, labels: usize,
    mode: ClassificationMode, policy: ClassificationPolicy,
    group_width: usize, work: Int4Work, retained_bytes: u64, result_bytes: u64,
    aggregation: ClassificationAggregationPolicy, chunk_limits: ChunkLimits, tokenizer_checks: usize,
    source_geometry: [u8; 32], encoding: Option<SafeBpeLimits>, encoding_work: SafeBpeWork,
}
impl Int4ClassificationMapPreflight {
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source }
    fn encoding_record(&self) -> Option<Int4ClassificationMapEncoding> {
        self.encoding.map(|limits| Int4ClassificationMapEncoding {
            policy: UntrustedEncoding::SafeBpeV1(limits), work: self.encoding_work })
    }
    pub fn verify_completed_cohorts(&self, run: &Int4ClassificationMapCohortRun, cohort_chunks: usize)
        -> Result<(), Int4ClassificationMapError> {
        if cohort_chunks == 0 || cohort_chunks > self.chunk_limits.max_chunks
            || self.heads == 0 || cohort_chunks.checked_mul(self.heads)
                .is_none_or(|heads| heads > crate::native_engine::strict_int4::cohort::MAX_INT4_SEQUENCES)
            || run.schema_version != 1 || run.execution != INT4_CLASSIFICATION_MAP_COHORT_EXECUTION
            || run.chunk_cohorts != self.chunks.div_ceil(cohort_chunks)
            || run.group_steps == 0 || run.group_steps > self.work.forward_positions
            || run.projection_groups == 0 || run.projection_groups > self.work.projected_logits {
            return Err(Int4ClassificationMapError::Admission);
        }
        self.verify_completed(&run.output)?;
        value_bytes(run, self.result_bytes.checked_add(cohort::COHORT_WRAPPER_BYTES)
            .ok_or(Int4ClassificationMapError::Limits)?)?;
        Ok(())
    }
    pub fn verify_completed(&self, run: &Int4ClassificationMapRun) -> Result<(), Int4ClassificationMapError> {
        if run.schema_version != 1 || run.execution != INT4_CLASSIFICATION_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.chunk_profile != CHUNK_PROFILE || run.semantics != SEMANTICS || run.single_context_equivalence
            || run.source_span != self.source || run.chunks.len() != self.chunks
            || run.chunk_limits != self.chunk_limits || run.tokenizer_checks != self.tokenizer_checks
            || run.planned_model_work != self.work || run.model_work != self.work
            || run.aggregate.schema_version != 1
            || run.aggregate.profile != crate::tasks::classify::CLASSIFICATION_AGGREGATION_PROFILE
            || run.aggregate.policy != self.aggregation || run.aggregate.classification_policy != self.policy
            || run.aggregate.chunk_count != self.chunks || run.aggregate.source_bytes != self.source.byte_end
            || run.aggregate.source_scalars != self.source.scalar_end || run.retained_result_bytes > self.retained_bytes
            || run.input_encoding != self.encoding_record()
            || geometry(run.chunks.iter().map(|chunk| (chunk.chunk_id, chunk.source_span, chunk.source_tokens))) != self.source_geometry {
            return Err(Int4ClassificationMapError::Admission);
        }
        match (self.mode, &run.aggregate.result) {
            (ClassificationMode::Exclusive, ClassificationAggregateResult::Exclusive(result))
                if result.candidate_weights.len() == self.labels => {},
            (ClassificationMode::MultiLabel, ClassificationAggregateResult::MultiLabel(result))
                if result.labels.len() == self.labels => {},
            _ => return Err(Int4ClassificationMapError::Admission),
        }
        let mut byte = 0; let mut scalar = 0; let mut work = Int4Work::default(); let mut bytes = 0_u64;
        for (index, chunk) in run.chunks.iter().enumerate() {
            let span = chunk.source_span;
            if chunk.chunk_id != index || span.byte_start != byte || span.scalar_start != scalar
                || span.byte_end <= byte || span.scalar_end <= scalar
                || chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != span.byte_end - byte)
                || chunk.run.schema_version != 1
                || chunk.run.execution != INT4_CLASSIFICATION_EXECUTION || chunk.run.numerics_profile != STRICT_INT4_PROFILE
                || chunk.run.group_width != self.group_width || chunk.run.head_count != self.heads {
                return Err(Int4ClassificationMapError::Admission);
            }
            byte = span.byte_end; scalar = span.scalar_end; work = work.checked_add(chunk.run.model_work)?;
            bytes = bytes.checked_add(value_bytes(chunk, self.retained_bytes)?)
                .filter(|&n| n <= self.retained_bytes).ok_or(Int4ClassificationMapError::Output)?;
        }
        if byte != self.source.byte_end || scalar != self.source.scalar_end || work != self.work
            || bytes != run.retained_result_bytes { return Err(Int4ClassificationMapError::Admission); }
        value_bytes(run, self.result_bytes)?; Ok(())
    }
}

/// Borrows the original source and retains every compiled per-chunk task.
/// No public scorer, wire constructor or externally supplied scores exist.
pub struct PreparedInt4ClassificationMap<'a> {
    chunks: ChunkPlan<'a>, plans: Vec<PreparedInt4Classification>,
    limits: Int4ClassificationMapLimits, expected: Int4ClassificationMapPreflight, task: TaskBudget,
}
impl Int4ClassificationPlanner {
    pub fn plan_document_with_control<'a, C: DecodeStepControl>(&self, request: &'a ClassificationRequest,
        limits: Int4ClassificationMapLimits, control: &mut C) -> Result<PreparedInt4ClassificationMap<'a>, Int4ClassificationMapError> {
        planning::checkpoint(control)?; limits.validate(request.budget)?;
        if request.document.is_empty() { return Err(Int4ClassificationMapError::EmptySource); }
        let mut work = SafeBpeWork::default();
        let capacity = self.source_token_capacity_with_work(request, limits.chunks.context_tokens, &mut work, control)?;
        PreparedInt4ClassificationMap::compile_encoded(request, limits, capacity, self.group_width(),
            self.document_encoding(), work, control,
            |source, control, work| self.source_tokens_with_work(source, work, control),
            |request, control, work| self.plan_with_encoding_work(request, control, work))
    }
}
impl<'a> PreparedInt4ClassificationMap<'a> {
    #[cfg(test)]
    fn compile<C: DecodeStepControl, F>(request: &'a ClassificationRequest, limits: Int4ClassificationMapLimits,
        capacity: usize, width: usize, control: &mut C, mut prepare: F) -> Result<Self, Int4ClassificationMapError>
    where F: FnMut(&ClassificationRequest, &mut C) -> Result<PreparedInt4Classification, Int4ClassificationError> {
        Self::compile_encoded(request, limits, capacity, width, &UntrustedEncoding::ByteFallback,
            SafeBpeWork::default(), control, |source, _, _| Ok(source.len()),
            |request, control, _| prepare(request, control))
    }
    #[allow(clippy::too_many_arguments)]
    fn compile_encoded<C: DecodeStepControl, Count, Prepare>(request: &'a ClassificationRequest,
        limits: Int4ClassificationMapLimits, capacity: usize, width: usize, encoding: &UntrustedEncoding,
        mut work: SafeBpeWork, control: &mut C, mut count: Count, mut prepare: Prepare)
        -> Result<Self, Int4ClassificationMapError>
    where Count: FnMut(&str, &mut C, &mut SafeBpeWork) -> Result<usize, ClassificationPlanningError>,
        Prepare: FnMut(&ClassificationRequest, &mut C, &mut SafeBpeWork)
            -> Result<PreparedInt4Classification, Int4ClassificationError> {
        encoding.validate().map_err(|_| Int4ClassificationMapError::Limits)?;
        limits.validate(request.budget)?;
        if request.document.is_empty() { return Err(Int4ClassificationMapError::EmptySource); }
        if capacity == 0 { return Err(Int4ClassificationMapError::Limits); }
        let reserved = limits.chunks.context_tokens.checked_sub(capacity).ok_or(Int4ClassificationMapError::Limits)?;
        let chunk_limits = ChunkLimits { max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reserved), ..limits.chunks };
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints(&request.document, chunk_limits,
                |text| count(text, &mut **control.borrow_mut(), &mut work).map_err(|cause| {
                    *error.borrow_mut() = Some(cause); MapReduceError::Tokenizer
                }),
                || match control.borrow_mut().prefill_checkpoint(0) { None => Ok(()), Some(cause) => {
                    *error.borrow_mut() = Some(ClassificationPlanningError::Cancelled(cause));
                    Err(MapReduceError::Cancelled)
                } })
        };
        if let Some(error) = error.into_inner() { return Err(error.into()); }
        let chunks = chunks.map_err(Int4ClassificationMapError::Chunk)?;
        let last = chunks.chunks().last().ok_or(Int4ClassificationMapError::EmptySource)?.span();
        let heads = if request.mode == ClassificationMode::Exclusive { 1 } else { request.labels.len() };
        let mut expected = Int4ClassificationMapPreflight { source: VerifiedSourceSpan { byte_start: 0,
            byte_end: last.byte_end, scalar_start: 0, scalar_end: last.scalar_end }, chunks: chunks.chunks().len(), heads,
            labels: request.labels.len(), mode: request.mode, policy: request.policy, group_width: width,
            work: Int4Work::default(), retained_bytes: limits.max_retained_result_bytes, result_bytes: limits.max_result_bytes,
            aggregation: limits.aggregation, chunk_limits, tokenizer_checks: chunks.tokenizer_calls(),
            source_geometry: geometry(chunks.chunks().iter().map(|chunk| (chunk.id(), chunk.span(), chunk.tokens()))),
            encoding: match encoding { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) }, encoding_work: work };
        let mut plans = planning::reserved(expected.chunks)?;
        for chunk in chunks.chunks() {
            planning::checkpoint(control)?;
            let mut document = String::new(); document.try_reserve_exact(chunk.text().len())
                .map_err(|_| Int4ClassificationMapError::Allocation)?;
            document.push_str(chunk.text());
            let row = ClassificationRequest { document, labels: request.labels.clone(), mode: request.mode,
                policy: request.policy, budget: request.budget };
            let prepared = prepare(&row, control, &mut work)?;
            if prepared.group_width() != width || prepared.head_count() != heads || prepared.task_budget() != request.budget
                || prepared.source_token_count()? != chunk.tokens()
                || prepared.required_context() > chunk_limits.context_tokens { return Err(Int4ClassificationMapError::Admission); }
            expected.work = expected.work.checked_add(prepared.planned_work())?;
            if !fits(expected.work, limits.max_model_work) { return Err(Int4ClassificationMapError::Work); }
            plans.push(prepared);
        }
        expected.encoding_work = work;
        planning::checkpoint(control)?;
        Ok(Self { chunks, plans, limits, expected, task: request.budget })
    }
    pub fn preflight_metadata(&self) -> Int4ClassificationMapPreflight { self.expected }
    pub fn chunk_count(&self) -> usize { self.plans.len() }
    pub fn head_count(&self) -> usize { self.expected.heads }
    pub fn group_width(&self) -> usize { self.expected.group_width }
    pub fn planned_work(&self) -> Int4Work { self.expected.work }
    pub fn task_budget(&self) -> TaskBudget { self.task }
    pub fn required_context(&self) -> usize { self.plans.iter().map(PreparedInt4Classification::required_context).max().unwrap_or(0) }
    pub fn max_result_bytes(&self) -> u64 { self.limits.max_result_bytes }
    pub fn identities(&self) -> Result<Vec<ExecutionIdentity>, Int4ClassificationMapError> {
        let mut identities = planning::reserved(self.plans.len())?;
        identities.extend(self.plans.iter().map(|plan| plan.execution_identity().clone())); Ok(identities)
    }
    /// Check every future chunk and head against the actual resident engine
    /// without executing a forward. A corpus can freeze the complete admission
    /// before transferring its owned request to the execution phase.
    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<(), Int4ClassificationMapError> {
        planning::checkpoint(control)?;
        if admitted.len() != self.plans.len() || self.plans.is_empty() { return Err(Int4ClassificationMapError::Admission); }
        check_work_ceiling(self.planned_work(), budget.native)?;
        if !fits(self.planned_work(), self.limits.max_model_work) { return Err(Int4ClassificationMapError::Work); }
        // Every future chunk and head is checked against the actual engine
        // before the first native call, not lazily after previous successes.
        for (plan, identity) in self.plans.iter().zip(admitted) {
            planning::checkpoint(control)?;
            plan.preflight(identity, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes })?;
        }
        planning::checkpoint(control)?; Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4ClassificationMapRun, Int4ClassificationMapError> {
        self.preflight_with_control(admitted, engine, budget, control)?;
        self.collect_with_control(control, |index, plan, control| {
            let run = plan.execute_with_control(&admitted[index], engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int4ClassificationMapError::Admission); }
            Ok(run)
        })
    }
    fn collect_with_control<C: DecodeStepControl, F>(&self, control: &mut C, mut execute: F)
        -> Result<Int4ClassificationMapRun, Int4ClassificationMapError>
    where F: FnMut(usize, &PreparedInt4Classification, &mut C) -> Result<Int4ClassificationRun, Int4ClassificationMapError> {
        let mut chunks = planning::reserved(self.plans.len())?;
        let mut work = Int4Work::default(); let mut bytes = 0_u64;
        for (index, (plan, source)) in self.plans.iter().zip(self.chunks.chunks()).enumerate() {
            planning::checkpoint(control)?;
            let run = execute(index, plan, control)?;
            planning::checkpoint(control)?;
            if run.schema_version != 1 || run.execution != INT4_CLASSIFICATION_EXECUTION
                || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width()
                || run.head_count != self.head_count() || run.model_work != plan.planned_work() {
                return Err(Int4ClassificationMapError::Admission);
            }
            let chunk = Int4ClassificationMapChunk { chunk_id: index, source_span: source.span(), source_tokens: source.tokens(), run };
            let room = self.limits.max_retained_result_bytes.checked_sub(bytes).ok_or(Int4ClassificationMapError::Output)?;
            bytes = bytes.checked_add(value_bytes(&chunk, room)?).ok_or(Int4ClassificationMapError::Output)?;
            work = work.checked_add(chunk.run.model_work)?; chunks.push(chunk);
        }
        if chunks.len() != self.expected.chunks || work != self.planned_work() { return Err(Int4ClassificationMapError::Admission); }
        let span = self.expected.source;
        let aggregate = aggregate_with_control(chunks.iter().map(|chunk| ClassificationChunkView {
            chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }), chunks.len(),
            span.byte_end, span.scalar_end, self.expected.mode, self.expected.policy, self.limits.aggregation, control)?;
        let run = Int4ClassificationMapRun { schema_version: 1, execution: INT4_CLASSIFICATION_MAP_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.group_width(),
            chunk_profile: CHUNK_PROFILE.to_owned(), semantics: SEMANTICS.to_owned(), single_context_equivalence: false,
            source_span: span, chunk_limits: self.chunks.limits(), tokenizer_checks: self.chunks.tokenizer_calls(),
            planned_model_work: self.planned_work(), model_work: work, retained_result_bytes: bytes, chunks, aggregate,
            input_encoding: self.expected.encoding_record() };
        value_bytes(&run, self.limits.max_result_bytes)?;
        planning::checkpoint(control)?; Ok(run)
    }
}
fn geometry(rows: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"classification-map-source-token-geometry-v1");
    for (id, span, tokens) in rows {
        for value in [id, span.byte_start, span.byte_end, span.scalar_start, span.scalar_end, tokens] {
            hash.update((value as u64).to_le_bytes());
        }
    }
    hash.finalize().into()
}
pub(super) fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn value_bytes(value: &impl Serialize, cap: u64) -> Result<u64, Int4ClassificationMapError> {
    planning::check_output(value, cap).map_err(|error| if error == ClassificationPlanningError::OutputBudget {
        Int4ClassificationMapError::Output
    } else { Int4ClassificationMapError::Serialization })?;
    let bytes = canonjson::canonical_bytes(value).map_err(|_| Int4ClassificationMapError::Serialization)?;
    if bytes.len() as u64 > cap { return Err(Int4ClassificationMapError::Output); }
    Ok(bytes.len() as u64)
}

#[cfg(test)] mod tests;
