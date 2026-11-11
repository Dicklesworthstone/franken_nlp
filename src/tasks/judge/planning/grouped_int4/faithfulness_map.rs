//! Complete-source faithfulness from lossless chunks and the original sealed
//! E/C/U judge heads. Every chunk is planned before any native forward. The
//! document reducer names its separate evidence semantics explicitly.
use super::*;
use crate::{tasks::{judge::{FaithfulnessAggregate, FaithfulnessAggregationPolicy,
        FaithfulnessAggregationError, aggregation::{FaithfulnessChunkView, aggregate_with_control}},
    mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError}},
    validation::grounded_fields::VerifiedSourceSpan};

pub mod memory;
mod cohort;
mod binding;
pub use cohort::{Int4FaithfulnessMapCohortBudget, Int4FaithfulnessMapCohortRun,
    INT4_FAITHFULNESS_MAP_COHORT_EXECUTION};

pub const INT4_FAITHFULNESS_MAP_EXECUTION: &str = "grouped-int4-complete-faithfulness-map-v1";
pub const MAX_FAITHFULNESS_MAP_CHUNKS: usize = 256;
pub const MAX_FAITHFULNESS_MAP_HEADS: usize = 32;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const SEMANTICS: &str = "independent-source-chunks-complete-evidence-reduction-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int4FaithfulnessMapLimits {
    /// `max_input_bytes` bounds source plus claim; chunk geometry covers the
    /// original source only. The unchanged claim is repeated in every head.
    pub chunks: ChunkLimits,
    /// One complete-document allowance, never replenished per chunk/window.
    pub max_model_work: Int4Work,
    pub max_retained_result_bytes: u64,
    pub max_result_bytes: u64,
    pub aggregation: FaithfulnessAggregationPolicy,
}
impl Int4FaithfulnessMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int4FaithfulnessMapError> {
        self.chunks.effective_token_limit().map_err(Int4FaithfulnessMapError::Chunk)?;
        task.validate().map_err(|_| Int4FaithfulnessMapError::Limits)?;
        self.aggregation.validate()?;
        if !(1..=MAX_FAITHFULNESS_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || self.max_model_work.forward_positions == 0 || self.max_model_work.projected_logits == 0
            || self.max_model_work.attention_pairs == 0 || self.max_model_work.projections.dot_products == 0
            || self.max_model_work.projections.multiply_accumulates == 0 {
            return Err(Int4FaithfulnessMapError::Limits);
        }
        Ok(())
    }
}

pub enum Int4FaithfulnessMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization,
    Chunk(MapReduceError), Judge(Int4JudgeError), Aggregation(FaithfulnessAggregationError),
}
impl fmt::Display for Int4FaithfulnessMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "invalid native complete-source faithfulness limits",
            Self::EmptySource => "native complete-source faithfulness requires nonempty source and claim",
            Self::Work => "native faithfulness whole-document work allowance exceeded",
            Self::Admission => "native faithfulness document identity or capacity differs",
            Self::Allocation => "native faithfulness document allocation refused",
            Self::Output => "complete native faithfulness document exceeds its output allowance",
            Self::Serialization => "native faithfulness document serialization refused",
            Self::Chunk(_) => "native faithfulness document partition refused",
            Self::Judge(_) => "native faithfulness document task refused",
            Self::Aggregation(_) => "native faithfulness document evidence reduction refused",
        })
    }
}
impl fmt::Debug for Int4FaithfulnessMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4FaithfulnessMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Judge(e) => Some(e),
            Self::Aggregation(e) => Some(e), _ => None }
    }
}
impl From<Int4JudgeError> for Int4FaithfulnessMapError {
    fn from(e: Int4JudgeError) -> Self { Self::Judge(e) }
}
impl From<JudgeError> for Int4FaithfulnessMapError {
    fn from(e: JudgeError) -> Self { Self::Judge(e.into()) }
}
impl From<StrictInt4Error> for Int4FaithfulnessMapError {
    fn from(e: StrictInt4Error) -> Self { Self::Judge(e.into()) }
}
impl From<Int4ScoringError> for Int4FaithfulnessMapError {
    fn from(e: Int4ScoringError) -> Self { Self::Judge(e.into()) }
}
impl From<FaithfulnessAggregationError> for Int4FaithfulnessMapError {
    fn from(e: FaithfulnessAggregationError) -> Self { Self::Aggregation(e) }
}
impl Int4FaithfulnessMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Judge(e) => e.cancellation(),
            Self::Aggregation(FaithfulnessAggregationError::Cancelled(cause)) => Some(*cause), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int4FaithfulnessMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub source_tokens: usize,
    /// The original exact result keeps chunk-local evidence coordinates.
    /// Aggregate witnesses separately carry original-source coordinates.
    pub run: Int4JudgeRun,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Int4FaithfulnessMapRun {
    pub schema_version: u32, pub execution: String, pub numerics_profile: String, pub group_width: usize,
    pub chunk_profile: String, pub semantics: String, pub single_context_equivalence: bool,
    pub source_span: VerifiedSourceSpan, pub chunk_limits: ChunkLimits, pub tokenizer_checks: usize,
    pub planned_model_work: Int4Work, pub model_work: Int4Work, pub retained_result_bytes: u64,
    pub chunks: Vec<Int4FaithfulnessMapChunk>, pub aggregate: FaithfulnessAggregate,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int4FaithfulnessMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4FaithfulnessMapRun")
            .field("schema_version", &self.schema_version).field("execution", &self.execution)
            .field("numerics_profile", &self.numerics_profile).field("group_width", &self.group_width)
            .field("chunk_profile", &self.chunk_profile).field("semantics", &self.semantics)
            .field("single_context_equivalence", &self.single_context_equivalence)
            .field("source_span", &self.source_span).field("chunk_limits", &self.chunk_limits)
            .field("tokenizer_checks", &self.tokenizer_checks).field("planned_model_work", &self.planned_model_work)
            .field("model_work", &self.model_work).field("retained_result_bytes", &self.retained_result_bytes)
            .field("chunks", &self.chunks).field("aggregate", &self.aggregate).finish()
    }
}

/// Private request authority and content-free completion geometry. Only the
/// native collector seals completed results; external JSON has no authority.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int4FaithfulnessMapPreflight {
    source: VerifiedSourceSpan, chunks: usize, heads: usize, max_heads: usize,
    policy: FaithfulnessPolicy, group_width: usize, work: Int4Work,
    retained_bytes: u64, result_bytes: u64, aggregation: FaithfulnessAggregationPolicy,
    chunk_limits: ChunkLimits, tokenizer_checks: usize,
    request_binding: [u8; 32],
}
impl fmt::Debug for Int4FaithfulnessMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int4FaithfulnessMapPreflight")
            .field("source", &self.source).field("chunks", &self.chunks)
            .field("heads", &self.heads).field("max_heads", &self.max_heads)
            .field("policy", &self.policy).field("group_width", &self.group_width).field("work", &self.work)
            .field("retained_bytes", &self.retained_bytes).field("result_bytes", &self.result_bytes)
            .field("aggregation", &self.aggregation).field("chunk_limits", &self.chunk_limits)
            .field("tokenizer_checks", &self.tokenizer_checks).finish()
    }
}
impl Int4FaithfulnessMapPreflight {
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn head_count(&self) -> usize { self.heads }
    pub fn max_head_count(&self) -> usize { self.max_heads }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source }
    pub fn verify_completed_cohorts(&self, run: &Int4FaithfulnessMapCohortRun, cohort_chunks: usize)
        -> Result<(), Int4FaithfulnessMapError> {
        self.verify_completed_cohorts_with_control(run, cohort_chunks, &mut Continue)
    }
    /// Verify the physical schedule and nested result without leaving the
    /// caller's deadline and checkpoint allowance during private hashing.
    pub fn verify_completed_cohorts_with_control<C: DecodeStepControl>(&self,
        run: &Int4FaithfulnessMapCohortRun, cohort_chunks: usize, control: &mut C)
        -> Result<(), Int4FaithfulnessMapError> {
        checkpoint(control)?;
        let cap = self.result_bytes.checked_add(cohort::COHORT_WRAPPER_BYTES)
            .ok_or(Int4FaithfulnessMapError::Limits)?;
        let vocabulary = NANBEIGE_VOCAB_SIZE as u64;
        if !run.binding_matches_with_control(cap, control)? || cohort_chunks == 0
            || cohort_chunks > self.chunk_limits.max_chunks
            || cohort_chunks > crate::native_engine::strict_int4::cohort::MAX_INT4_SEQUENCES
            || run.effective_cohort_chunks() != cohort_chunks.min(self.chunks)
            || run.schema_version != 1 || run.execution != INT4_FAITHFULNESS_MAP_COHORT_EXECUTION
            || run.chunk_cohorts != self.chunks.div_ceil(cohort_chunks)
            || run.group_steps == 0 || run.group_steps > self.work.forward_positions
            || self.work.projected_logits % vocabulary != 0
            || run.projection_groups == 0 || run.projection_groups > self.work.projected_logits / vocabulary {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        self.verify_completed_with_control(&run.output, control)?;
        value_bytes(run, cap)?;
        checkpoint(control)?; Ok(())
    }
    pub fn verify_completed(&self, run: &Int4FaithfulnessMapRun) -> Result<(), Int4FaithfulnessMapError> {
        self.verify_completed_with_control(run, &mut Continue)
    }
    /// Preserve cancellation through result hashing and complete validation.
    /// The compatibility wrapper above uses the same checks without a stop source.
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self,
        run: &Int4FaithfulnessMapRun, control: &mut C) -> Result<(), Int4FaithfulnessMapError> {
        checkpoint(control)?;
        // Equal source length, claim length and work do not identify the
        // admitted source/claim. Public fields remain editable for consumers,
        // but no edited result can pass the original completion authority.
        if run.request_binding != self.request_binding
            || run.result_binding != binding::result_with_control(run, self.result_bytes, control)? {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        if run.schema_version != 1 || run.execution != INT4_FAITHFULNESS_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.chunk_profile != CHUNK_PROFILE || run.semantics != SEMANTICS || run.single_context_equivalence
            || run.source_span != self.source || run.chunks.len() != self.chunks
            || run.chunk_limits != self.chunk_limits || run.tokenizer_checks != self.tokenizer_checks
            || run.planned_model_work != self.work || run.model_work != self.work
            || run.retained_result_bytes > self.retained_bytes {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        run.aggregate.verify_metadata(self.aggregation, self.policy, self.chunks,
            self.source.byte_end, self.source.scalar_end)?;
        let mut byte = 0; let mut scalar = 0; let mut heads = 0_usize; let mut max_heads = 0;
        let mut work = Int4Work::default(); let mut bytes = 0_u64;
        for (index, chunk) in run.chunks.iter().enumerate() {
            checkpoint(control)?;
            let span = chunk.source_span;
            let JudgeResult::Faithfulness(result) = &chunk.run.result else { return Err(Int4FaithfulnessMapError::Admission); };
            if chunk.chunk_id != index || span.byte_start != byte || span.scalar_start != scalar
                || span.byte_end <= byte || span.scalar_end <= scalar
                || chunk.source_tokens != span.byte_end - byte || chunk.run.schema_version != 1
                || chunk.run.execution != INT4_JUDGE_EXECUTION || chunk.run.numerics_profile != STRICT_INT4_PROFILE
                || chunk.run.group_width != self.group_width || result.policy != self.policy
                || chunk.run.head_count == 0 || chunk.run.head_count > MAX_FAITHFULNESS_MAP_HEADS
                || chunk.run.head_count != result.heads.len() {
                return Err(Int4FaithfulnessMapError::Admission);
            }
            byte = span.byte_end; scalar = span.scalar_end;
            heads = heads.checked_add(chunk.run.head_count).ok_or(Int4FaithfulnessMapError::Admission)?;
            max_heads = max_heads.max(chunk.run.head_count);
            work = work.checked_add(chunk.run.model_work)?;
            bytes = bytes.checked_add(value_bytes(chunk, self.retained_bytes)?)
                .filter(|&n| n <= self.retained_bytes).ok_or(Int4FaithfulnessMapError::Output)?;
        }
        if byte != self.source.byte_end || scalar != self.source.scalar_end || work != self.work
            || heads != self.heads || max_heads != self.max_heads || bytes != run.retained_result_bytes {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        value_bytes(run, self.result_bytes)?; checkpoint(control)?; Ok(())
    }
}

/// Source and claim remain private. Every original chunk retains a sealed
/// native judgment with its exact complete evidence partition and identity.
pub struct PreparedInt4FaithfulnessMap<'a> {
    source: &'a str, chunks: ChunkPlan<'a>, plans: Vec<PreparedInt4Judge>,
    limits: Int4FaithfulnessMapLimits, expected: Int4FaithfulnessMapPreflight, task: TaskBudget,
}
impl Int4JudgePlanner {
    pub fn plan_faithfulness_document_with_control<'a, C: DecodeStepControl>(&self, request: &'a JudgeRequest,
        limits: Int4FaithfulnessMapLimits, control: &mut C)
        -> Result<PreparedInt4FaithfulnessMap<'a>, Int4FaithfulnessMapError> {
        checkpoint(control)?;
        let capacity = self.max_faithfulness_source_bytes(request, limits.chunks.context_tokens)?;
        PreparedInt4FaithfulnessMap::compile_bound(request, limits, capacity, self.group_width(),
            self.semantic_configuration(), control,
            |request, control| self.plan_with_control(request, control))
    }
}
impl<'a> PreparedInt4FaithfulnessMap<'a> {
    #[cfg(test)]
    fn compile<C: DecodeStepControl, F>(request: &'a JudgeRequest, limits: Int4FaithfulnessMapLimits,
        capacity: usize, width: usize, control: &mut C, mut prepare: F)
        -> Result<Self, Int4FaithfulnessMapError>
    where F: FnMut(&JudgeRequest, &mut C) -> Result<PreparedInt4Judge, Int4JudgeError> {
        Self::compile_bound(request, limits, capacity, width, (request.budget(), JudgeLimits::default()),
            control, |request, control| prepare(request, control))
    }
    #[allow(clippy::too_many_arguments)]
    fn compile_bound<C: DecodeStepControl, F>(request: &'a JudgeRequest, limits: Int4FaithfulnessMapLimits,
        capacity: usize, width: usize, authority: (TaskBudget, JudgeLimits), control: &mut C, mut prepare: F)
        -> Result<Self, Int4FaithfulnessMapError>
    where F: FnMut(&JudgeRequest, &mut C) -> Result<PreparedInt4Judge, Int4JudgeError> {
        checkpoint(control)?;
        let JudgeRequest::Faithfulness { source, claim, policy, budget } = request else {
            return Err(Int4FaithfulnessMapError::Limits);
        };
        limits.validate(*budget)?; policy.validate()?;
        if source.is_empty() || claim.is_empty() { return Err(Int4FaithfulnessMapError::EmptySource); }
        if capacity == 0 || source.len().checked_add(claim.len()).is_none_or(|n| n > limits.chunks.max_input_bytes) {
            return Err(Int4FaithfulnessMapError::Limits);
        }
        let reserved_tokens = limits.chunks.context_tokens.checked_sub(capacity).ok_or(Int4FaithfulnessMapError::Limits)?;
        let chunk_limits = ChunkLimits { max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reserved_tokens), ..limits.chunks };
        let mut stopped = None;
        let chunks = ChunkPlan::build_with_checkpoints(source, chunk_limits, |text| Ok(text.len()),
            || match control.prefill_checkpoint(0) { None => Ok(()), Some(cause) => {
                stopped = Some(cause); Err(MapReduceError::Cancelled) } });
        if let Some(cause) = stopped { return Err(Int4JudgeError::Cancelled(cause).into()); }
        let chunks = chunks.map_err(Int4FaithfulnessMapError::Chunk)?;
        let last = chunks.chunks().last().ok_or(Int4FaithfulnessMapError::EmptySource)?.span();
        let mut expected = Int4FaithfulnessMapPreflight {
            source: VerifiedSourceSpan { byte_start: 0, byte_end: last.byte_end, scalar_start: 0, scalar_end: last.scalar_end },
            chunks: chunks.chunks().len(), heads: 0, max_heads: 0, policy: *policy, group_width: width,
            work: Int4Work::default(), retained_bytes: limits.max_retained_result_bytes, result_bytes: limits.max_result_bytes,
            aggregation: limits.aggregation, chunk_limits, tokenizer_checks: chunks.tokenizer_calls(), request_binding: [0; 32] };
        let mut plans = reserved(expected.chunks)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let row = JudgeRequest::Faithfulness { source: copy_text(chunk.text())?, claim: copy_text(claim)?,
                policy: *policy, budget: *budget };
            let prepared = prepare(&row, control)?;
            if prepared.group_width() != width || prepared.head_count() == 0
                || prepared.head_count() > MAX_FAITHFULNESS_MAP_HEADS || prepared.task_budget() != *budget
                || prepared.required_context() > chunk_limits.context_tokens {
                return Err(Int4FaithfulnessMapError::Admission);
            }
            expected.work = expected.work.checked_add(prepared.planned_work())?;
            if !fits_work(expected.work, limits.max_model_work) { return Err(Int4FaithfulnessMapError::Work); }
            expected.heads = expected.heads.checked_add(prepared.head_count()).ok_or(Int4FaithfulnessMapError::Limits)?;
            expected.max_heads = expected.max_heads.max(prepared.head_count());
            plans.push(prepared);
        }
        expected.request_binding = binding::request(request, limits, authority, &chunks, &plans, control)?;
        checkpoint(control)?;
        Ok(Self { source, chunks, plans, limits, expected, task: *budget })
    }
    pub fn preflight_metadata(&self) -> Int4FaithfulnessMapPreflight { self.expected }
    pub fn chunk_count(&self) -> usize { self.plans.len() }
    /// Total heads across the whole source, including every evidence window.
    pub fn head_count(&self) -> usize { self.expected.heads }
    pub fn max_head_count(&self) -> usize { self.expected.max_heads }
    pub fn group_width(&self) -> usize { self.expected.group_width }
    pub fn planned_work(&self) -> Int4Work { self.expected.work }
    pub fn task_budget(&self) -> TaskBudget { self.task }
    pub fn required_context(&self) -> usize { self.plans.iter().map(PreparedInt4Judge::required_context).max().unwrap_or(0) }
    pub fn max_result_bytes(&self) -> u64 { self.limits.max_result_bytes }
    pub fn identities(&self) -> Result<Vec<ExecutionIdentity>, Int4FaithfulnessMapError> {
        let mut identities = reserved(self.plans.len())?;
        identities.extend(self.plans.iter().map(|plan| plan.execution_identity().clone())); Ok(identities)
    }
    /// Admit every original chunk against the actual resident engine without
    /// forwarding or changing native state. Corpus and durable callers use
    /// this before charging an attempt; execution repeats the same checks.
    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<(), Int4FaithfulnessMapError> {
        checkpoint(control)?;
        if admitted.len() != self.plans.len() || self.plans.is_empty() { return Err(Int4FaithfulnessMapError::Admission); }
        check_work(self.planned_work(), budget.native)?;
        if !fits_work(self.planned_work(), self.limits.max_model_work) { return Err(Int4FaithfulnessMapError::Work); }
        // Admit every future complete judgment against the actual engine
        // before spending any of the document's nonrenewable allowance.
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.preflight(identity, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes })?;
        }
        checkpoint(control)?;
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4FaithfulnessMapRun, Int4FaithfulnessMapError> {
        self.preflight_with_control(admitted, engine, budget, control)?;
        self.collect_with_control(control, |index, plan, control| {
            let run = plan.execute_with_control(&admitted[index], engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int4FaithfulnessMapError::Admission); }
            Ok(run)
        })
    }
    fn collect_with_control<C: DecodeStepControl, F>(&self, control: &mut C, mut execute: F)
        -> Result<Int4FaithfulnessMapRun, Int4FaithfulnessMapError>
    where F: FnMut(usize, &PreparedInt4Judge, &mut C) -> Result<Int4JudgeRun, Int4FaithfulnessMapError> {
        let mut chunks = reserved(self.plans.len())?;
        let mut work = Int4Work::default(); let mut bytes = 0_u64;
        for (index, (plan, source)) in self.plans.iter().zip(self.chunks.chunks()).enumerate() {
            checkpoint(control)?;
            let run = execute(index, plan, control)?;
            checkpoint(control)?;
            if run.schema_version != 1 || run.execution != INT4_JUDGE_EXECUTION
                || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width()
                || run.head_count != plan.head_count() || run.model_work != plan.planned_work()
                || !matches!(&run.result, JudgeResult::Faithfulness(result) if result.policy == self.expected.policy) {
                return Err(Int4FaithfulnessMapError::Admission);
            }
            let chunk = Int4FaithfulnessMapChunk { chunk_id: index, source_span: source.span(), source_tokens: source.tokens(), run };
            let room = self.limits.max_retained_result_bytes.checked_sub(bytes).ok_or(Int4FaithfulnessMapError::Output)?;
            bytes = bytes.checked_add(value_bytes(&chunk, room)?).ok_or(Int4FaithfulnessMapError::Output)?;
            work = work.checked_add(chunk.run.model_work)?; chunks.push(chunk);
        }
        if chunks.len() != self.expected.chunks || work != self.planned_work() { return Err(Int4FaithfulnessMapError::Admission); }
        let mut views = reserved(chunks.len())?;
        for chunk in &chunks {
            let JudgeResult::Faithfulness(result) = &chunk.run.result else { return Err(Int4FaithfulnessMapError::Admission); };
            views.push(FaithfulnessChunkView { chunk_id: chunk.chunk_id, span: chunk.source_span, result });
        }
        let aggregate = aggregate_with_control(views, chunks.len(), self.source,
            self.expected.policy, self.limits.aggregation, control)?;
        let mut run = Int4FaithfulnessMapRun { schema_version: 1, execution: INT4_FAITHFULNESS_MAP_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.group_width(),
            chunk_profile: CHUNK_PROFILE.to_owned(), semantics: SEMANTICS.to_owned(), single_context_equivalence: false,
            source_span: self.expected.source, chunk_limits: self.chunks.limits(), tokenizer_checks: self.chunks.tokenizer_calls(),
            planned_model_work: self.planned_work(), model_work: work, retained_result_bytes: bytes, chunks, aggregate,
            request_binding: self.expected.request_binding, result_binding: [0; 32] };
        run.result_binding = binding::result_with_control(&run, self.limits.max_result_bytes, control)?;
        value_bytes(&run, self.limits.max_result_bytes)?;
        checkpoint(control)?; Ok(run)
    }
}
fn copy_text(text: &str) -> Result<String, Int4FaithfulnessMapError> {
    let mut copy = String::new(); copy.try_reserve_exact(text.len()).map_err(|_| Int4FaithfulnessMapError::Allocation)?;
    copy.push_str(text); Ok(copy)
}
fn fits_work(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn value_bytes(value: &impl Serialize, cap: u64) -> Result<u64, Int4FaithfulnessMapError> {
    struct Count { remaining: u64, exceeded: bool }
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            match self.remaining.checked_sub(bytes.len() as u64) {
                Some(remaining) => { self.remaining = remaining; Ok(bytes.len()) }
                None => { self.exceeded = true; Err(std::io::Error::other("bounded faithfulness map output")) }
            }
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut count = Count { remaining: cap, exceeded: false };
    serde_json::to_writer(&mut count, value).map_err(|_| if count.exceeded {
        Int4FaithfulnessMapError::Output
    } else { Int4FaithfulnessMapError::Serialization })?;
    let bytes = canonjson::canonical_bytes(value).map_err(|_| Int4FaithfulnessMapError::Serialization)?;
    if bytes.len() as u64 > cap { return Err(Int4FaithfulnessMapError::Output); }
    Ok(bytes.len() as u64)
}

#[cfg(test)] mod tests;
