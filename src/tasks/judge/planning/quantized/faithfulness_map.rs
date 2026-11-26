//! Complete-source INT8 faithfulness over lossless original-source chunks.
//! The unchanged claim is judged against every chunk and all its evidence
//! windows. The shared reducer preserves support, counterevidence and explicit
//! coverage abstention; it never claims inference across separate contexts.
use super::*;
use std::cell::RefCell;
use sha2::{Digest, Sha256};
use crate::{
    tasks::{judge::{FaithfulnessAggregate, FaithfulnessAggregationPolicy,
        FaithfulnessAggregationError, aggregation::{FaithfulnessChunkView, aggregate_with_control}},
        mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError}},
    tokenizer::untrusted::SafeBpeLimits,
    validation::grounded_fields::VerifiedSourceSpan,
};

mod binding;

pub const INT8_FAITHFULNESS_MAP_EXECUTION: &str = "portable-int8-complete-faithfulness-map-v1";
pub const MAX_FAITHFULNESS_MAP_CHUNKS: usize = 256;
pub const MAX_FAITHFULNESS_MAP_HEADS: usize = 32;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const SEMANTICS: &str = "independent-source-chunks-complete-evidence-reduction-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int8FaithfulnessMapLimits {
    /// Source plus claim share the input-byte ceiling. Chunk coordinates
    /// cover only the source; the exact claim repeats in every native head.
    pub chunks: ChunkLimits,
    /// One complete-document allowance, never renewed per chunk or window.
    pub max_model_work: Int8Work,
    pub max_retained_result_bytes: u64,
    pub max_result_bytes: u64,
    pub aggregation: FaithfulnessAggregationPolicy,
}
impl Int8FaithfulnessMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int8FaithfulnessMapError> {
        self.chunks.effective_token_limit().map_err(Int8FaithfulnessMapError::Chunk)?;
        task.validate().map_err(|_| Int8FaithfulnessMapError::Limits)?;
        self.aggregation.validate()?;
        if !(1..=MAX_FAITHFULNESS_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || self.max_model_work.forward_positions == 0 || self.max_model_work.projected_logits == 0
            || self.max_model_work.attention_pairs == 0 || self.max_model_work.projections.dot_products == 0
            || self.max_model_work.projections.multiply_accumulates == 0 {
            return Err(Int8FaithfulnessMapError::Limits);
        }
        Ok(())
    }
}

pub enum Int8FaithfulnessMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization,
    Chunk(MapReduceError), Judge(Int8JudgeError), Aggregation(FaithfulnessAggregationError),
}
impl fmt::Display for Int8FaithfulnessMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "invalid INT8 complete-source faithfulness limits",
            Self::EmptySource => "INT8 complete-source faithfulness requires nonempty source and claim",
            Self::Work => "INT8 faithfulness whole-document work allowance exceeded",
            Self::Admission => "INT8 faithfulness document identity or capacity differs",
            Self::Allocation => "INT8 faithfulness document allocation refused",
            Self::Output => "complete INT8 faithfulness document exceeds its output allowance",
            Self::Serialization => "INT8 faithfulness document serialization refused",
            Self::Chunk(_) => "INT8 faithfulness document partition refused",
            Self::Judge(_) => "INT8 faithfulness document task refused",
            Self::Aggregation(_) => "INT8 faithfulness complete-document evidence reduction refused",
        })
    }
}
impl fmt::Debug for Int8FaithfulnessMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8FaithfulnessMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(error) => Some(error), Self::Judge(error) => Some(error),
            Self::Aggregation(error) => Some(error), _ => None }
    }
}
impl From<Int8JudgeError> for Int8FaithfulnessMapError {
    fn from(error: Int8JudgeError) -> Self { Self::Judge(error) }
}
impl From<JudgeError> for Int8FaithfulnessMapError {
    fn from(error: JudgeError) -> Self { Self::Judge(error.into()) }
}
impl From<StrictInt8Error> for Int8FaithfulnessMapError {
    fn from(error: StrictInt8Error) -> Self { Self::Judge(error.into()) }
}
impl From<FaithfulnessAggregationError> for Int8FaithfulnessMapError {
    fn from(error: FaithfulnessAggregationError) -> Self { Self::Aggregation(error) }
}
impl Int8FaithfulnessMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Judge(error) => error.cancellation(),
            Self::Aggregation(FaithfulnessAggregationError::Cancelled(cause)) => Some(*cause), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int8FaithfulnessMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Exact source token count, excluding the repeated claim and scaffold.
    pub source_tokens: usize,
    /// Every unchanged chunk-local judgment and evidence head is retained.
    /// Aggregate witnesses separately carry original-source coordinates.
    pub run: Int8JudgeRun,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Int8FaithfulnessMapEncoding {
    pub policy: UntrustedEncoding,
    /// Includes claim capacity, all source probes and every final chunk/head.
    pub work: SafeBpeWork,
}

/// Native completion authority stays private, has no Deserialize constructor,
/// and is absent from serialized output and diagnostic formatting.
#[derive(Clone, PartialEq, Serialize)]
pub struct Int8FaithfulnessMapRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub chunk_profile: String,
    pub semantics: String,
    pub single_context_equivalence: bool,
    pub source_span: VerifiedSourceSpan,
    pub chunk_limits: ChunkLimits,
    pub tokenizer_checks: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub retained_result_bytes: u64,
    pub chunks: Vec<Int8FaithfulnessMapChunk>,
    pub aggregate: FaithfulnessAggregate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int8FaithfulnessMapEncoding>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int8FaithfulnessMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8FaithfulnessMapRun")
            .field("schema_version", &self.schema_version).field("execution", &self.execution)
            .field("numerics_profile", &self.numerics_profile).field("source_span", &self.source_span)
            .field("chunks", &self.chunks.len()).field("model_work", &self.model_work)
            .field("retained_result_bytes", &self.retained_result_bytes).finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int8FaithfulnessMapPreflight {
    source: VerifiedSourceSpan,
    chunks: usize,
    heads: usize,
    max_heads: usize,
    policy: FaithfulnessPolicy,
    work: Int8Work,
    retained_bytes: u64,
    result_bytes: u64,
    aggregation: FaithfulnessAggregationPolicy,
    chunk_limits: ChunkLimits,
    tokenizer_checks: usize,
    source_geometry: [u8; 32],
    encoding: Option<SafeBpeLimits>,
    encoding_work: SafeBpeWork,
    request_binding: [u8; 32],
}
impl fmt::Debug for Int8FaithfulnessMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8FaithfulnessMapPreflight")
            .field("source", &self.source).field("chunks", &self.chunks)
            .field("heads", &self.heads).field("max_heads", &self.max_heads)
            .field("work", &self.work).field("retained_bytes", &self.retained_bytes)
            .field("result_bytes", &self.result_bytes).field("chunk_limits", &self.chunk_limits)
            .field("tokenizer_checks", &self.tokenizer_checks).finish_non_exhaustive()
    }
}
impl Int8FaithfulnessMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source }
    pub fn chunk_count(&self) -> usize { self.chunks }
    /// Total heads over all chunks, including every local evidence window.
    pub fn head_count(&self) -> usize { self.heads }
    pub fn max_head_count(&self) -> usize { self.max_heads }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.result_bytes }
    fn encoding_record(&self) -> Option<Int8FaithfulnessMapEncoding> {
        self.encoding.map(|limits| Int8FaithfulnessMapEncoding {
            policy: UntrustedEncoding::SafeBpeV1(limits), work: self.encoding_work })
    }
    pub fn verify_completed(&self, run: &Int8FaithfulnessMapRun) -> Result<(), Int8FaithfulnessMapError> {
        self.verify_completed_with_control(run, &mut Continue)
    }
    /// Validate every public field against the private complete native result
    /// under the original cancellation/deadline allowance.
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8FaithfulnessMapRun,
        control: &mut C) -> Result<(), Int8FaithfulnessMapError> {
        checkpoint(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != binding::result(run, self.result_bytes, control)?.0 {
            return Err(Int8FaithfulnessMapError::Admission);
        }
        if run.schema_version != 1 || run.execution != INT8_FAITHFULNESS_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.chunk_profile != CHUNK_PROFILE
            || run.semantics != SEMANTICS || run.single_context_equivalence || run.source_span != self.source
            || run.chunks.len() != self.chunks || run.chunk_limits != self.chunk_limits
            || run.tokenizer_checks != self.tokenizer_checks
            || run.planned_model_work != self.work || run.model_work != self.work
            || run.retained_result_bytes > self.retained_bytes || run.input_encoding != self.encoding_record()
            || geometry(run.chunks.iter().map(|chunk| (chunk.chunk_id, chunk.source_span, chunk.source_tokens))) != self.source_geometry {
            return Err(Int8FaithfulnessMapError::Admission);
        }
        run.aggregate.verify_metadata(self.aggregation, self.policy, self.chunks,
            self.source.byte_end, self.source.scalar_end)?;
        let mut byte = 0; let mut scalar = 0; let mut heads = 0_usize; let mut max_heads = 0;
        let mut work = Int8Work::default(); let mut bytes = 0_u64;
        for (index, chunk) in run.chunks.iter().enumerate() {
            checkpoint(control)?;
            let span = chunk.source_span;
            let JudgeResult::Faithfulness(result) = &chunk.run.result else { return Err(Int8FaithfulnessMapError::Admission); };
            if chunk.chunk_id != index || span.byte_start != byte || span.scalar_start != scalar
                || span.byte_end <= byte || span.scalar_end <= scalar || chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != span.byte_end - byte)
                || chunk.run.schema_version != 1 || chunk.run.execution != INT8_JUDGE_EXECUTION
                || chunk.run.numerics_profile != STRICT_INT8_PROFILE || result.policy != self.policy
                || chunk.run.head_count == 0 || chunk.run.head_count > MAX_FAITHFULNESS_MAP_HEADS
                || chunk.run.head_count != result.heads.len()
                || result.work.projected_logits != chunk.run.model_work.projected_logits
                || !finite_result(result) {
                return Err(Int8FaithfulnessMapError::Admission);
            }
            byte = span.byte_end; scalar = span.scalar_end;
            heads = heads.checked_add(chunk.run.head_count).ok_or(Int8FaithfulnessMapError::Admission)?;
            max_heads = max_heads.max(chunk.run.head_count);
            work = work.checked_add(chunk.run.model_work)?;
            let room = self.retained_bytes.checked_sub(bytes).ok_or(Int8FaithfulnessMapError::Output)?;
            bytes = bytes.checked_add(binding::value_bytes(chunk, room, control)?)
                .ok_or(Int8FaithfulnessMapError::Output)?;
        }
        if byte != self.source.byte_end || scalar != self.source.scalar_end || work != self.work
            || heads != self.heads || max_heads != self.max_heads || bytes != run.retained_result_bytes {
            return Err(Int8FaithfulnessMapError::Admission);
        }
        checkpoint(control)?; Ok(())
    }
}

/// Borrows the exact original source and owns every sealed native chunk plan.
/// No public synthetic scorer or caller-provided result can complete this map.
pub struct PreparedInt8FaithfulnessMap<'a> {
    source: &'a str,
    chunks: ChunkPlan<'a>,
    plans: Vec<PreparedInt8Judge>,
    limits: Int8FaithfulnessMapLimits,
    expected: Int8FaithfulnessMapPreflight,
    task: TaskBudget,
}
impl JudgePlanner {
    pub fn plan_int8_faithfulness_document_with_control<'a, C: DecodeStepControl>(&self,
        request: &'a JudgeRequest, context: &PlanContext<'_>, planning: JudgeLimits,
        limits: Int8FaithfulnessMapLimits, control: &mut C)
        -> Result<PreparedInt8FaithfulnessMap<'a>, Int8FaithfulnessMapError> {
        self.plan_int8_faithfulness_document_with_encoding_work(request, context, planning,
            limits, control, &mut SafeBpeWork::default())
    }

    /// Corpus preparation and exact reconstruction share one document-local
    /// allowance. The caller retains every completed encoding debit on error;
    /// successful metadata describes only this pass, preserving plan identity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_int8_faithfulness_document_with_encoding_work<'a, C: DecodeStepControl>(&self,
        request: &'a JudgeRequest, context: &PlanContext<'_>, planning: JudgeLimits,
        limits: Int8FaithfulnessMapLimits, control: &mut C, encoding_work: &mut SafeBpeWork)
        -> Result<PreparedInt8FaithfulnessMap<'a>, Int8FaithfulnessMapError> {
        checkpoint(control)?;
        check_profile(context.execution_identity())?;
        self.check_context(context, NumericsProfile::StrictQuantized { version: 1 })?;
        let JudgeRequest::Faithfulness { source, claim, policy, budget } = request else {
            return Err(Int8FaithfulnessMapError::Limits);
        };
        limits.validate(*budget)?; policy.validate()?;
        self.document_encoding().validate().map_err(|_| Int8FaithfulnessMapError::Limits)?;
        if source.is_empty() || claim.is_empty() { return Err(Int8FaithfulnessMapError::EmptySource); }
        if source.len().checked_add(claim.len()).is_none_or(|n| n > limits.chunks.max_input_bytes) {
            return Err(Int8FaithfulnessMapError::Limits);
        }
        let before = *encoding_work;
        let capacity = self.faithfulness.source_token_capacity_with_work(claim, *policy, *budget,
            *context.budget_ceiling(), planning, limits.chunks.context_tokens, self.document_encoding(),
            encoding_work, control)?;
        let byte_capacity = FaithfulnessCompiler::source_byte_limit(*policy)?;
        let reserved_tokens = limits.chunks.context_tokens.checked_sub(capacity)
            .ok_or(Int8FaithfulnessMapError::Limits)?;
        let chunk_limits = ChunkLimits {
            max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            max_chunk_bytes: limits.chunks.max_chunk_bytes.min(byte_capacity.max(4)),
            reserved_tokens: limits.chunks.reserved_tokens.max(reserved_tokens), ..limits.chunks };
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints_and_byte_cap(source, chunk_limits,
                byte_capacity.min(chunk_limits.max_chunk_bytes),
                |text| self.faithfulness.source_tokens_with_work(text, self.document_encoding(),
                    encoding_work, &mut **control.borrow_mut()).map_err(|cause| {
                        *error.borrow_mut() = Some(cause); MapReduceError::Tokenizer
                    }),
                || match control.borrow_mut().prefill_checkpoint(0) {
                    None => Ok(()),
                    Some(cause) => {
                        *error.borrow_mut() = Some(JudgeError::Cancelled(cause));
                        Err(MapReduceError::Cancelled)
                    }
                })
        };
        if let Some(error) = error.into_inner() { return Err(error.into()); }
        let chunks = chunks.map_err(Int8FaithfulnessMapError::Chunk)?;
        let last = chunks.chunks().last().ok_or(Int8FaithfulnessMapError::EmptySource)?.span();
        let mut plans = reserved(chunks.chunks().len())?;
        let mut work = Int8Work::default(); let mut heads = 0_usize; let mut max_heads = 0;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let row = JudgeRequest::Faithfulness { source: copy_text(chunk.text())?, claim: copy_text(claim)?,
                policy: *policy, budget: *budget };
            let plan = self.plan_int8_with_encoding_work(&row, context, planning, control, encoding_work)?;
            if plan.head_count() == 0 || plan.head_count() > MAX_FAITHFULNESS_MAP_HEADS
                || plan.task_budget() != *budget || source_token_count(&plan) != Some(chunk.tokens())
                || plan.required_context() > chunk_limits.context_tokens {
                return Err(Int8FaithfulnessMapError::Admission);
            }
            work = work.checked_add(plan.planned_work())?;
            if !fits_work(work, limits.max_model_work) { return Err(Int8FaithfulnessMapError::Work); }
            heads = heads.checked_add(plan.head_count()).ok_or(Int8FaithfulnessMapError::Limits)?;
            max_heads = max_heads.max(plan.head_count());
            plans.push(plan);
        }
        let pass_encoding_work = encoding_work.checked_sub(before)
            .ok_or(Int8FaithfulnessMapError::Admission)?;
        let request_binding = binding::request(request, context, planning, limits,
            self.document_encoding(), pass_encoding_work, &chunks, &plans, control)?;
        let expected = Int8FaithfulnessMapPreflight {
            source: VerifiedSourceSpan { byte_start: 0, byte_end: last.byte_end,
                scalar_start: 0, scalar_end: last.scalar_end },
            chunks: chunks.chunks().len(), heads, max_heads, policy: *policy, work,
            retained_bytes: limits.max_retained_result_bytes, result_bytes: limits.max_result_bytes,
            aggregation: limits.aggregation, chunk_limits, tokenizer_checks: chunks.tokenizer_calls(),
            source_geometry: geometry(chunks.chunks().iter().map(|chunk| (chunk.id(), chunk.span(), chunk.tokens()))),
            encoding: match self.document_encoding() { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) },
            encoding_work: pass_encoding_work, request_binding,
        };
        checkpoint(control)?;
        Ok(PreparedInt8FaithfulnessMap { source, chunks, plans, limits, expected, task: *budget })
    }
}
impl PreparedInt8FaithfulnessMap<'_> {
    pub fn preflight_metadata(&self) -> Int8FaithfulnessMapPreflight { self.expected }
    pub fn chunk_count(&self) -> usize { self.plans.len() }
    /// Total native heads over every original chunk and evidence window.
    pub fn head_count(&self) -> usize { self.expected.heads }
    pub fn max_head_count(&self) -> usize { self.expected.max_heads }
    pub fn planned_work(&self) -> Int8Work { self.expected.work }
    pub fn task_budget(&self) -> TaskBudget { self.task }
    pub fn required_context(&self) -> usize {
        self.plans.iter().map(PreparedInt8Judge::required_context).max().unwrap_or(0)
    }
    pub fn max_result_bytes(&self) -> u64 { self.limits.max_result_bytes }
    pub fn identities(&self) -> Result<Vec<ExecutionIdentity>, Int8FaithfulnessMapError> {
        let mut identities = reserved(self.plans.len())?;
        identities.extend(self.plans.iter().map(|plan| plan.execution_identity().clone()));
        Ok(identities)
    }
    /// Check every future chunk against the actual native engine before the
    /// first forward. The whole allowance cannot renew for another chunk.
    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<(), Int8FaithfulnessMapError> {
        checkpoint(control)?;
        self.check_identities(admitted, budget.native, control)?;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.preflight(identity, engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes })?;
        }
        checkpoint(control)?; Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8FaithfulnessMapRun, Int8FaithfulnessMapError> {
        self.execute_with_prefill(admitted, engine, budget, None, control)
    }
    /// Reuse bounded prompt morsels across every chunk and evidence head.
    /// Candidate continuations, full-vocabulary denominators and the complete
    /// document work allowance retain their sequential execution contract.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
        prefill: Int8PrefillLimits, control: &mut C)
        -> Result<Int8FaithfulnessMapRun, Int8FaithfulnessMapError> {
        self.execute_with_prefill(admitted, engine, budget, Some(prefill), control)
    }
    fn execute_with_prefill<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
        prefill: Option<Int8PrefillLimits>, control: &mut C)
        -> Result<Int8FaithfulnessMapRun, Int8FaithfulnessMapError> {
        if let Some(limits) = prefill { limits.validate()?; }
        self.preflight_with_control(admitted, engine, budget, control)?;
        self.collect_with_control(control, |index, plan, control| {
            let local = Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes };
            let run = match prefill {
                Some(limits) => plan.execute_layer_major_with_control(
                    &admitted[index], engine, local, limits, control),
                None => plan.execute_with_control(&admitted[index], engine, local, control),
            }?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8FaithfulnessMapError::Admission);
            }
            Ok(run)
        })
    }
    fn check_identities<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        budget: Int8RunBudget, control: &mut C) -> Result<(), Int8FaithfulnessMapError> {
        if self.plans.is_empty() || self.plans.len() != self.chunks.chunks().len()
            || admitted.len() != self.plans.len() || self.plans.len() != self.expected.chunks {
            return Err(Int8FaithfulnessMapError::Admission);
        }
        if !fits_work(self.planned_work(), self.limits.max_model_work) { return Err(Int8FaithfulnessMapError::Work); }
        check_work(self.planned_work(), budget)?;
        let mut work = Int8Work::default(); let mut heads = 0_usize; let mut max_heads = 0;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.verify_identity(identity)?;
            if plan.task_budget() != self.task || plan.head_count() == 0
                || plan.head_count() > MAX_FAITHFULNESS_MAP_HEADS
                || plan.required_context() > self.expected.chunk_limits.context_tokens {
                return Err(Int8FaithfulnessMapError::Admission);
            }
            work = work.checked_add(plan.planned_work())?;
            heads = heads.checked_add(plan.head_count()).ok_or(Int8FaithfulnessMapError::Admission)?;
            max_heads = max_heads.max(plan.head_count());
        }
        if work != self.planned_work() || heads != self.head_count() || max_heads != self.max_head_count() {
            return Err(Int8FaithfulnessMapError::Admission);
        }
        checkpoint(control)?; Ok(())
    }
    /// Test-only callers can replace this private numerical boundary while
    /// still exercising the real exact scorer finalizer and evidence reducer.
    fn collect_with_control<C: DecodeStepControl, F>(&self, control: &mut C, mut execute: F)
        -> Result<Int8FaithfulnessMapRun, Int8FaithfulnessMapError>
    where F: FnMut(usize, &PreparedInt8Judge, &mut C) -> Result<Int8JudgeRun, Int8FaithfulnessMapError> {
        let mut chunks = reserved(self.plans.len())?;
        let mut work = Int8Work::default(); let mut bytes = 0_u64;
        for (index, (plan, source)) in self.plans.iter().zip(self.chunks.chunks()).enumerate() {
            checkpoint(control)?;
            let run = execute(index, plan, control)?;
            checkpoint(control)?;
            if run.schema_version != 1 || run.execution != INT8_JUDGE_EXECUTION
                || run.numerics_profile != STRICT_INT8_PROFILE || run.head_count != plan.head_count()
                || run.model_work != plan.planned_work()
                || !matches!(&run.result, JudgeResult::Faithfulness(result) if result.policy == self.expected.policy) {
                return Err(Int8FaithfulnessMapError::Admission);
            }
            let chunk = Int8FaithfulnessMapChunk {
                chunk_id: index, source_span: source.span(), source_tokens: source.tokens(), run };
            let room = self.limits.max_retained_result_bytes.checked_sub(bytes).ok_or(Int8FaithfulnessMapError::Output)?;
            bytes = bytes.checked_add(binding::value_bytes(&chunk, room, control)?)
                .ok_or(Int8FaithfulnessMapError::Output)?;
            work = work.checked_add(chunk.run.model_work)?;
            chunks.push(chunk);
        }
        if chunks.len() != self.expected.chunks || work != self.planned_work() {
            return Err(Int8FaithfulnessMapError::Admission);
        }
        let mut views = reserved(chunks.len())?;
        for chunk in &chunks {
            let JudgeResult::Faithfulness(result) = &chunk.run.result else { return Err(Int8FaithfulnessMapError::Admission); };
            views.push(FaithfulnessChunkView { chunk_id: chunk.chunk_id, span: chunk.source_span, result });
        }
        // This shared reducer independently checks all local finalizer results,
        // finite full-vocabulary scores, exact windows and original evidence.
        let aggregate = aggregate_with_control(views, chunks.len(), self.source,
            self.expected.policy, self.limits.aggregation, control)?;
        let mut run = Int8FaithfulnessMapRun {
            schema_version: 1, execution: INT8_FAITHFULNESS_MAP_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), chunk_profile: CHUNK_PROFILE.to_owned(),
            semantics: SEMANTICS.to_owned(), single_context_equivalence: false, source_span: self.expected.source,
            chunk_limits: self.chunks.limits(), tokenizer_checks: self.chunks.tokenizer_calls(),
            planned_model_work: self.planned_work(), model_work: work, retained_result_bytes: bytes, chunks, aggregate,
            input_encoding: self.expected.encoding_record(),
            request_binding: self.expected.request_binding, result_binding: [0; 32],
        };
        run.result_binding = binding::result(&run, self.limits.max_result_bytes, control)?.0;
        checkpoint(control)?; Ok(run)
    }
}

fn source_token_count(plan: &PreparedInt8Judge) -> Option<usize> {
    plan.inner.executable.bundle().heads.first()?.ir.prompt_segments().get(2)
        .filter(|segment| segment.kind() == PromptSegmentKind::Document)
        .map(|segment| segment.token_ids().len()).filter(|&count| count != 0)
}
fn finite_result(result: &FaithfulnessResult) -> bool {
    let finite = |assessment: &crate::tasks::judge::FaithfulnessAssessment|
        assessment.candidate_conditional_weight.is_finite() && assessment.log_score_margin.is_finite();
    finite(&result.global) && result.windows.iter().all(|window| finite(&window.assessment))
        && result.heads.iter().flat_map(|head| &head.candidates)
            .all(|candidate| candidate.sequence_score.is_finite() && candidate.candidate_weight.is_finite())
}
fn fits_work(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn geometry(rows: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"fnlp-int8-faithfulness-map-source-geometry-v1");
    for (id, span, tokens) in rows {
        for value in [id, span.byte_start, span.byte_end, span.scalar_start, span.scalar_end, tokens] {
            hash.update((value as u64).to_le_bytes());
        }
    }
    hash.finalize().into()
}
fn copy_text(text: &str) -> Result<String, Int8FaithfulnessMapError> {
    let mut copy = String::new();
    copy.try_reserve_exact(text.len()).map_err(|_| Int8FaithfulnessMapError::Allocation)?;
    copy.push_str(text); Ok(copy)
}

#[cfg(test)]
mod tests;
