//! Complete native INT8 sentiment over a lossless document partition.
//!
//! Every chunk and independent axis is planned before any forward. One native
//! engine reuses its KV storage, while the whole-document work allowance and
//! tokenizer ledger never renew between chunks. The shared exact reducer keeps
//! all local results and labels its scalar-weighted mixture explicitly.
use super::*;
use std::cell::RefCell;
use sha2::{Digest, Sha256};
use crate::{
    tasks::{mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError},
        sentiment::{SentimentAggregate, SentimentAggregationPolicy,
            aggregation::{SentimentAggregationError, SentimentChunkView, aggregate_with_control}}},
    tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedEncoding},
    validation::grounded_fields::VerifiedSourceSpan,
};

mod binding;
pub mod memory;

pub const INT8_SENTIMENT_MAP_EXECUTION: &str = "portable-int8-complete-sentiment-map-v1";
pub const MAX_SENTIMENT_MAP_CHUNKS: usize = 256;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const SEMANTICS: &str = "independent-chunks-flat-source-scalar-mixture-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int8SentimentMapLimits {
    pub chunks: ChunkLimits,
    /// All five native counters cover the complete document, never one chunk.
    pub max_model_work: Int8Work,
    /// Cumulative canonical bytes of every retained complete chunk record.
    pub max_retained_result_bytes: u64,
    /// Complete map envelope including the retained chunks and aggregate.
    pub max_result_bytes: u64,
    pub aggregation: SentimentAggregationPolicy,
}
impl Int8SentimentMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int8SentimentMapError> {
        self.chunks.effective_token_limit().map_err(Int8SentimentMapError::Chunk)?;
        task.validate().map_err(|_| Int8SentimentMapError::Limits)?;
        self.aggregation.validate()?;
        if !(1..=MAX_SENTIMENT_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || !positive_work(self.max_model_work) {
            return Err(Int8SentimentMapError::Limits);
        }
        Ok(())
    }
}

pub enum Int8SentimentMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization,
    Chunk(MapReduceError), Sentiment(Int8SentimentError), Aggregation(SentimentAggregationError),
}
impl fmt::Display for Int8SentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "invalid int8 sentiment document limits",
            Self::EmptySource => "int8 sentiment document requires nonempty source",
            Self::Work => "int8 sentiment whole-document work allowance exceeded",
            Self::Admission => "int8 sentiment document identity or capacity differs",
            Self::Allocation => "int8 sentiment document allocation refused",
            Self::Output => "complete int8 sentiment document exceeds its output allowance",
            Self::Serialization => "int8 sentiment document serialization refused",
            Self::Chunk(_) => "int8 sentiment document partition refused",
            Self::Sentiment(_) => "int8 sentiment document task refused",
            Self::Aggregation(_) => "int8 sentiment complete document aggregation refused",
        })
    }
}
impl fmt::Debug for Int8SentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8SentimentMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(e) => Some(e), Self::Sentiment(e) => Some(e),
            Self::Aggregation(e) => Some(e), _ => None }
    }
}
impl From<Int8SentimentError> for Int8SentimentMapError {
    fn from(e: Int8SentimentError) -> Self { Self::Sentiment(e) }
}
impl From<SentimentPlanningError> for Int8SentimentMapError {
    fn from(e: SentimentPlanningError) -> Self { Self::Sentiment(e.into()) }
}
impl From<StrictInt8Error> for Int8SentimentMapError {
    fn from(e: StrictInt8Error) -> Self { Self::Sentiment(e.into()) }
}
impl From<SentimentAggregationError> for Int8SentimentMapError {
    fn from(e: SentimentAggregationError) -> Self { Self::Aggregation(e) }
}
impl Int8SentimentMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Sentiment(e) => e.cancellation(),
            Self::Aggregation(SentimentAggregationError::Cancelled(cause)) => Some(*cause), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int8SentimentMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Actual source tokens, excluding trusted scaffolds and continuations.
    pub source_tokens: usize,
    pub run: Int8SentimentRun,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Int8SentimentMapEncoding {
    pub policy: UntrustedEncoding,
    /// Includes every sizing probe and actual plan in this planning pass.
    pub work: SafeBpeWork,
}

/// Native completion with private request and payload seals. It is not
/// deserializable; external JSON cannot mint execution or admission authority.
#[derive(Clone, PartialEq, Serialize)]
pub struct Int8SentimentMapRun {
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
    pub chunks: Vec<Int8SentimentMapChunk>,
    pub aggregate: SentimentAggregate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int8SentimentMapEncoding>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int8SentimentMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8SentimentMapRun")
            .field("schema_version", &self.schema_version).field("execution", &self.execution)
            .field("numerics_profile", &self.numerics_profile).field("chunk_profile", &self.chunk_profile)
            .field("semantics", &self.semantics).field("single_context_equivalence", &self.single_context_equivalence)
            .field("source_span", &self.source_span).field("chunk_limits", &self.chunk_limits)
            .field("tokenizer_checks", &self.tokenizer_checks).field("planned_model_work", &self.planned_model_work)
            .field("model_work", &self.model_work).field("retained_result_bytes", &self.retained_result_bytes)
            .field("chunks", &self.chunks).field("aggregate", &self.aggregate)
            .field("input_encoding", &self.input_encoding).finish()
    }
}

/// Content-free geometry plus private request authority, retained through the
/// hosted boundary to authenticate the complete output before publication.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int8SentimentMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    heads: usize,
    work: Int8Work,
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
impl fmt::Debug for Int8SentimentMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8SentimentMapPreflight")
            .field("source_span", &self.source_span).field("chunks", &self.chunks).field("heads", &self.heads)
            .field("work", &self.work).field("max_retained_bytes", &self.max_retained_bytes)
            .field("max_result_bytes", &self.max_result_bytes).field("aggregation", &self.aggregation)
            .field("chunk_limits", &self.chunk_limits).field("tokenizer_checks", &self.tokenizer_checks)
            .field("encoding", &self.encoding).field("encoding_work", &self.encoding_work).finish_non_exhaustive()
    }
}
impl Int8SentimentMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.max_result_bytes }
    fn encoding_record(&self) -> Option<Int8SentimentMapEncoding> {
        self.encoding.map(|limits| Int8SentimentMapEncoding {
            policy: UntrustedEncoding::SafeBpeV1(limits), work: self.encoding_work })
    }
    pub fn verify_completed(&self, run: &Int8SentimentMapRun) -> Result<(), Int8SentimentMapError> {
        self.verify_completed_with_control(run, &mut binding::Continue)
    }
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8SentimentMapRun,
        control: &mut C) -> Result<(), Int8SentimentMapError> {
        checkpoint(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != binding::result_with_control(run, self.max_result_bytes, control)?
            || run.schema_version != 1 || run.execution != INT8_SENTIMENT_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.chunk_profile != CHUNK_PROFILE
            || run.semantics != SEMANTICS || run.single_context_equivalence
            || run.source_span != self.source_span || run.chunks.len() != self.chunks
            || run.planned_model_work != self.work || run.model_work != self.work
            || run.chunk_limits != self.chunk_limits || run.tokenizer_checks != self.tokenizer_checks
            || run.input_encoding != self.encoding_record()
            || run.retained_result_bytes > self.max_retained_bytes
            || geometry(run.chunks.iter().map(|chunk| (chunk.chunk_id, chunk.source_span, chunk.source_tokens)))
                != self.source_geometry {
            return Err(Int8SentimentMapError::Admission);
        }
        let mut work = Int8Work::default();
        let mut retained = 0_u64;
        for (index, chunk) in run.chunks.iter().enumerate() {
            checkpoint(control)?;
            if chunk.chunk_id != index || chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != chunk.source_span.byte_end - chunk.source_span.byte_start)
                || chunk.run.schema_version != 1 || chunk.run.execution != INT8_SENTIMENT_EXECUTION
                || chunk.run.numerics_profile != STRICT_INT8_PROFILE || chunk.run.head_count != self.heads
                || chunk.run.result.dimensions.len() != self.heads {
                return Err(Int8SentimentMapError::Admission);
            }
            work = work.checked_add(chunk.run.model_work)?;
            let remaining = self.max_retained_bytes.checked_sub(retained).ok_or(Int8SentimentMapError::Output)?;
            retained = retained.checked_add(binding::value_bytes_with_control(chunk, remaining, control)?)
                .ok_or(Int8SentimentMapError::Output)?;
        }
        // Recompute the shared finite reducer as a typed check as well as a
        // byte seal: serde maps a forged Some(NaN) to the same JSON as None.
        let aggregate = aggregate_with_control(run.chunks.iter().map(|chunk| SentimentChunkView {
            chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }),
            self.chunks, self.source_span.byte_end, self.source_span.scalar_end, self.aggregation, control)?;
        if work != self.work || retained != run.retained_result_bytes || aggregate != run.aggregate {
            return Err(Int8SentimentMapError::Admission);
        }
        binding::value_bytes_with_control(run, self.max_result_bytes, control)?;
        checkpoint(control)?;
        Ok(())
    }
}

/// Borrows the exact original source and owns every compiled native chunk.
/// No public synthetic backend or wire-plan constructor can bypass planning.
pub struct PreparedInt8SentimentMap<'a> {
    chunks: ChunkPlan<'a>,
    plans: Vec<PreparedInt8Sentiment>,
    limits: Int8SentimentMapLimits,
    expected: Int8SentimentMapPreflight,
    task: TaskBudget,
}
impl SentimentPlanner {
    pub fn plan_int8_document_with_control<'a, C: DecodeStepControl>(&self, request: &'a SentimentRequest,
        context: &PlanContext<'_>, planning: SentimentLimits, limits: Int8SentimentMapLimits,
        control: &mut C) -> Result<PreparedInt8SentimentMap<'a>, Int8SentimentMapError> {
        self.plan_int8_document_with_encoding_work(request, context, planning, limits,
            control, &mut SafeBpeWork::default())
    }
    /// Population preparation and exact reconstruction can lend one ledger;
    /// the completion records only this deterministic planning pass's delta.
    pub(crate) fn plan_int8_document_with_encoding_work<'a, C: DecodeStepControl>(&self,
        request: &'a SentimentRequest, context: &PlanContext<'_>, planning: SentimentLimits,
        limits: Int8SentimentMapLimits, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedInt8SentimentMap<'a>, Int8SentimentMapError> {
        self.plan_int8_document_with_encoding_ceiling(request, context, planning, limits, control, work, None)
    }

    /// A resident population lends the original document ledger and a tighter
    /// execution allowance. Failed probes still mutate `work`; only this pass's
    /// delta enters the exact completion binding. The semantic policy is fixed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_int8_document_with_encoding_ceiling<'a, C: DecodeStepControl>(&self,
        request: &'a SentimentRequest, context: &PlanContext<'_>, planning: SentimentLimits,
        limits: Int8SentimentMapLimits, control: &mut C, work: &mut SafeBpeWork,
        ceiling: Option<SafeBpeLimits>)
        -> Result<PreparedInt8SentimentMap<'a>, Int8SentimentMapError> {
        checkpoint(control)?;
        limits.validate(request.budget)?;
        constrained_int8::check_profile(context.execution_identity())
            .map_err(|_| Int8SentimentMapError::Admission)?;
        if request.document.is_empty() { return Err(Int8SentimentMapError::EmptySource); }
        let previous_work = *work;
        let capacity = self.source_token_capacity(&request.axes, request.budget, *context.budget_ceiling(),
            planning, limits.chunks.context_tokens)?;
        let reservation = limits.chunks.context_tokens.checked_sub(capacity).ok_or(Int8SentimentMapError::Limits)?;
        let chunk_limits = ChunkLimits { max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reservation), ..limits.chunks };
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints(&request.document, chunk_limits,
                |text| self.source_tokens_with_encoding_ceiling(text, work, &mut **control.borrow_mut(), ceiling)
                    .map_err(|cause| {
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
        let chunks = chunks.map_err(Int8SentimentMapError::Chunk)?;
        let end = chunks.chunks().last().ok_or(Int8SentimentMapError::EmptySource)?.span();
        let mut expected = Int8SentimentMapPreflight {
            source_span: VerifiedSourceSpan { byte_start: 0, byte_end: end.byte_end,
                scalar_start: 0, scalar_end: end.scalar_end },
            chunks: chunks.chunks().len(), heads: request.axes.len(), work: Int8Work::default(),
            max_retained_bytes: limits.max_retained_result_bytes, max_result_bytes: limits.max_result_bytes,
            aggregation: limits.aggregation, chunk_limits, tokenizer_checks: chunks.tokenizer_calls(),
            source_geometry: geometry(chunks.chunks().iter().map(|chunk| (chunk.id(), chunk.span(), chunk.tokens()))),
            encoding: match self.document_encoding() { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) },
            encoding_work: SafeBpeWork::default(), request_binding: [0; 32],
        };
        let mut plans = reserved(expected.chunks)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let mut document = String::new();
            document.try_reserve_exact(chunk.text().len()).map_err(|_| Int8SentimentMapError::Allocation)?;
            document.push_str(chunk.text());
            let row = SentimentRequest { document, axes: request.axes.clone(), budget: request.budget };
            let inner = self.plan_for_profile_with_encoding_ceiling(&row, context, planning,
                NumericsProfile::StrictQuantized { version: 1 }, control, work, ceiling)?;
            let plan = PreparedInt8Sentiment::compile(inner, context.execution_identity().clone(), row.budget, control)?;
            if plan.head_count() != expected.heads || plan.task_budget() != request.budget
                || plan.source_token_count()? != chunk.tokens() || plan.required_context() > chunk_limits.context_tokens {
                return Err(Int8SentimentMapError::Admission);
            }
            expected.work = expected.work.checked_add(plan.planned_work())?;
            if !fits(expected.work, limits.max_model_work) { return Err(Int8SentimentMapError::Work); }
            plans.push(plan);
        }
        expected.encoding_work = work.checked_sub(previous_work).ok_or(Int8SentimentMapError::Admission)?;
        expected.request_binding = binding::request(request, context, planning, limits,
            self.document_encoding(), expected.encoding_work, &chunks, &plans, control)?;
        checkpoint(control)?;
        Ok(PreparedInt8SentimentMap { chunks, plans, limits, expected, task: request.budget })
    }
}
impl PreparedInt8SentimentMap<'_> {
    pub fn preflight_metadata(&self) -> Int8SentimentMapPreflight { self.expected }
    pub fn identities(&self) -> Result<Vec<ExecutionIdentity>, Int8SentimentMapError> {
        let mut identities = reserved(self.plans.len())?;
        identities.extend(self.plans.iter().map(|plan| plan.execution_identity().clone()));
        Ok(identities)
    }
    pub fn planned_work(&self) -> Int8Work { self.expected.work }
    /// Independent axes per chunk; total heads are chunk_count times this value.
    pub fn head_count(&self) -> usize { self.expected.heads }
    pub fn chunk_count(&self) -> usize { self.expected.chunks }
    pub fn required_context(&self) -> usize {
        self.plans.iter().map(PreparedInt8Sentiment::required_context).max().unwrap_or(0)
    }
    pub fn max_result_bytes(&self) -> u64 { self.limits.max_result_bytes }
    pub fn task_budget(&self) -> TaskBudget { self.task }

    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<(), Int8SentimentMapError> {
        checkpoint(control)?;
        self.check_identities(admitted, budget.native, control)?;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.preflight(identity, engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes })?;
        }
        checkpoint(control)?;
        Ok(())
    }

    /// One engine and one caller control span the complete request. Every
    /// later chunk is preflighted before the first chunk can spend native work.
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8SentimentMapRun, Int8SentimentMapError> {
        self.execute_with_prefill(admitted, engine, budget, None, control)
    }
    /// Optional bounded prompt morsels retain the same native work, score
    /// space and exact finalizers. The host separately admits prefill scratch.
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, prefill: Int8PrefillLimits,
        control: &mut C) -> Result<Int8SentimentMapRun, Int8SentimentMapError> {
        self.execute_with_prefill(admitted, engine, budget, Some(prefill), control)
    }
    fn execute_with_prefill<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, prefill: Option<Int8PrefillLimits>,
        control: &mut C) -> Result<Int8SentimentMapRun, Int8SentimentMapError> {
        if let Some(limits) = prefill { limits.validate()?; }
        self.preflight_with_control(admitted, engine, budget, control)?;
        self.execute_chunks(admitted, budget.native, control, |plan, identity, control| {
            let run = plan.execute_with_prefill(identity, engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes }, prefill, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8SentimentMapError::Admission);
            }
            Ok(run)
        })
    }
    fn check_identities<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        native: Int8RunBudget, control: &mut C) -> Result<(), Int8SentimentMapError> {
        if admitted.len() != self.plans.len() || self.plans.len() != self.expected.chunks
            || self.plans.len() != self.chunks.chunks().len() || !fits(self.expected.work, self.limits.max_model_work) {
            return Err(Int8SentimentMapError::Admission);
        }
        check_work(self.expected.work, native)?;
        let mut work = Int8Work::default();
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.verify_identity(identity)?;
            if plan.task_budget() != self.task || plan.head_count() != self.expected.heads {
                return Err(Int8SentimentMapError::Admission);
            }
            work = work.checked_add(plan.planned_work())?;
        }
        if work != self.expected.work { return Err(Int8SentimentMapError::Admission); }
        checkpoint(control)?;
        Ok(())
    }

    // Private numerical seam for fault injection. Public callers cannot
    // submit fabricated chunk scores or invoke a reducer without execution.
    fn execute_chunks<C: DecodeStepControl, F>(&self, admitted: &[ExecutionIdentity],
        native: Int8RunBudget, control: &mut C, mut execute: F)
        -> Result<Int8SentimentMapRun, Int8SentimentMapError>
    where F: FnMut(&PreparedInt8Sentiment, &ExecutionIdentity, &mut C)
        -> Result<Int8SentimentRun, Int8SentimentMapError> {
        checkpoint(control)?;
        self.check_identities(admitted, native, control)?;
        let mut chunks = reserved(self.chunk_count())?;
        let mut work = Int8Work::default();
        let mut bytes = 0_u64;
        for (index, ((plan, identity), source)) in self.plans.iter().zip(admitted)
            .zip(self.chunks.chunks()).enumerate() {
            checkpoint(control)?;
            let run = execute(plan, identity, control)?;
            if run.schema_version != 1 || run.execution != INT8_SENTIMENT_EXECUTION
                || run.numerics_profile != STRICT_INT8_PROFILE || run.head_count != plan.head_count()
                || run.model_work != plan.planned_work() || run.result.dimensions.len() != plan.head_count() {
                return Err(Int8SentimentMapError::Admission);
            }
            let chunk = Int8SentimentMapChunk { chunk_id: index, source_span: source.span(),
                source_tokens: source.tokens(), run };
            let remaining = self.limits.max_retained_result_bytes.checked_sub(bytes).ok_or(Int8SentimentMapError::Output)?;
            bytes = bytes.checked_add(binding::value_bytes_with_control(&chunk, remaining, control)?)
                .ok_or(Int8SentimentMapError::Output)?;
            work = work.checked_add(chunk.run.model_work)?;
            if !fits(work, self.expected.work) { return Err(Int8SentimentMapError::Work); }
            chunks.push(chunk);
        }
        checkpoint(control)?;
        if chunks.len() != self.chunk_count() || work != self.planned_work() { return Err(Int8SentimentMapError::Admission); }
        let span = self.expected.source_span;
        let aggregate = aggregate_with_control(chunks.iter().map(|chunk| SentimentChunkView {
            chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }), chunks.len(),
            span.byte_end, span.scalar_end, self.limits.aggregation, control)?;
        let mut run = Int8SentimentMapRun { schema_version: 1, execution: INT8_SENTIMENT_MAP_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), chunk_profile: CHUNK_PROFILE.to_owned(),
            semantics: SEMANTICS.to_owned(), single_context_equivalence: false,
            source_span: span, chunk_limits: self.chunks.limits(), tokenizer_checks: self.chunks.tokenizer_calls(),
            planned_model_work: self.planned_work(), model_work: work, retained_result_bytes: bytes,
            chunks, aggregate, input_encoding: self.expected.encoding_record(),
            request_binding: self.expected.request_binding, result_binding: [0; 32] };
        run.result_binding = binding::result_with_control(&run, self.max_result_bytes(), control)?;
        self.expected.verify_completed_with_control(&run, control)?;
        checkpoint(control)?;
        Ok(run)
    }
}

fn reserved<T>(len: usize) -> Result<Vec<T>, Int8SentimentMapError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| Int8SentimentMapError::Allocation)?;
    Ok(values)
}
fn geometry(rows: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"int8-sentiment-map-source-token-geometry-v1");
    for (id, span, tokens) in rows {
        for value in [id, span.byte_start, span.byte_end, span.scalar_start, span.scalar_end, tokens] {
            hash.update((value as u64).to_le_bytes());
        }
    }
    hash.finalize().into()
}
fn positive_work(work: Int8Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

#[cfg(test)] mod tests;
