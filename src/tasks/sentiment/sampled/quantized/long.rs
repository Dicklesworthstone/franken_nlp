//! Complete-document independent sampled sentiment on portable INT8 weights.
//!
//! Every original scalar belongs to exactly one bounded chunk. All chunk,
//! axis and sample work is admitted before native execution; a failed chunk
//! produces no document result. The shared reducer reports ordinal-paired
//! scalar-weighted replicates, never full-context equivalence or confidence.
use super::*;
use std::cell::RefCell;
use crate::{tasks::mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError},
    validation::grounded_fields::VerifiedSourceSpan};
use super::super::{budget_fits, MIB,
    long::{SampledSentimentMapAggregate, SampledSentimentMapEncoding, SampledSentimentMapError,
        SAMPLED_SENTIMENT_MAP_AGGREGATION, reduction}};

mod binding;
mod execution;
pub mod memory;

pub const INT8_SAMPLED_SENTIMENT_MAP_EXECUTION: &str = "portable-int8-complete-sampled-sentiment-map-v1";
pub const MAX_INT8_SAMPLED_SENTIMENT_MAP_CHUNKS: usize = 256;
const SEMANTICS: &str = "independent-chunks-ordinal-paired-scalar-weighted-document-replicates-v1";
const MAX_BYTES: u64 = 64 * MIB;

#[derive(Clone, Copy, Debug)]
pub struct Int8SampledSentimentMapLimits {
    pub chunks: ChunkLimits,
    /// Complete cold work across all source chunks, axes and samples.
    pub max_model_work: Int8Work,
    pub max_retained_result_bytes: u64,
    pub max_result_bytes: u64,
}
impl Int8SampledSentimentMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int8SampledSentimentMapError> {
        self.chunks.effective_token_limit().map_err(Int8SampledSentimentMapError::Chunk)?;
        task.validate().map_err(|_| Int8SampledSentimentMapError::Limits)?;
        if !(1..=MAX_INT8_SAMPLED_SENTIMENT_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || self.max_model_work.forward_positions == 0 || self.max_model_work.projected_logits == 0
            || self.max_model_work.attention_pairs == 0 || self.max_model_work.projections.dot_products == 0
            || self.max_model_work.projections.multiply_accumulates == 0 {
            return Err(Int8SampledSentimentMapError::Limits);
        }
        Ok(())
    }
}

pub enum Int8SampledSentimentMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output,
    Chunk(MapReduceError), Sampled(Int8SampledSentimentError),
    Shared(SampledSentimentMapError), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for Int8SampledSentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "int8 sampled sentiment document limits refused",
            Self::EmptySource => "int8 sampled sentiment document requires nonempty source",
            Self::Work => "int8 sampled sentiment complete document work exceeds admission",
            Self::Admission => "int8 sampled sentiment document identity or completion differs",
            Self::Allocation => "int8 sampled sentiment document allocation refused",
            Self::Output => "int8 sampled sentiment complete document output exceeds admission",
            Self::Chunk(_) => "int8 sampled sentiment complete source partition refused",
            Self::Sampled(_) => "int8 sampled sentiment document task refused",
            Self::Shared(_) => "int8 sampled sentiment document reduction or binding refused",
            Self::Cancelled(_) => "int8 sampled sentiment document cancelled",
        })
    }
}
impl fmt::Debug for Int8SampledSentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8SampledSentimentMapError {}
impl From<Int8SampledSentimentError> for Int8SampledSentimentMapError {
    fn from(error: Int8SampledSentimentError) -> Self { Self::Sampled(error) }
}
impl From<SampledSentimentError> for Int8SampledSentimentMapError {
    fn from(error: SampledSentimentError) -> Self { Self::Sampled(error.into()) }
}
impl From<StrictInt8Error> for Int8SampledSentimentMapError {
    fn from(error: StrictInt8Error) -> Self { Self::Sampled(error.into()) }
}
impl From<SampledSentimentMapError> for Int8SampledSentimentMapError {
    fn from(error: SampledSentimentMapError) -> Self { Self::Shared(error) }
}
impl Int8SampledSentimentMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Cancelled(cause) => Some(*cause), Self::Sampled(error) => error.cancellation(),
            Self::Shared(error) => error.cancellation(), _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int8SampledSentimentMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub source_tokens: usize,
    pub run: Int8SampledSentimentRun,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Int8SampledSentimentMapRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub chunk_profile: String,
    pub semantics: String,
    pub single_context_equivalence: bool,
    pub source_span: VerifiedSourceSpan,
    pub chunk_limits: ChunkLimits,
    pub tokenizer_checks: usize,
    pub samples_per_axis: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub planned_mask_node_visit_charge: u64,
    pub mask_node_visit_charge: u64,
    pub generated_tokens: u64,
    pub retained_result_bytes: u64,
    pub chunks: Vec<Int8SampledSentimentMapChunk>,
    pub aggregate: SampledSentimentMapAggregate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<SampledSentimentMapEncoding>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int8SampledSentimentMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8SampledSentimentMapRun").field("schema_version", &self.schema_version)
            .field("execution", &self.execution).field("source_span", &self.source_span)
            .field("chunk_count", &self.chunks.len()).finish_non_exhaustive()
    }
}

/// Private authority retained across physical execution and output publication.
/// Wire values cannot construct it or supply a replacement content seal.
#[derive(Clone, Copy)]
pub struct Int8SampledSentimentMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    axes: usize,
    samples: usize,
    work: Int8Work,
    max_output_tokens: u64,
    max_retained_bytes: u64,
    max_result_bytes: u64,
    chunk_limits: ChunkLimits,
    tokenizer_checks: usize,
    geometry: [u8; 32],
    encoding: Option<crate::tokenizer::untrusted::SafeBpeLimits>,
    encoding_work: SafeBpeWork,
    request_binding: [u8; 32],
}
impl Int8SampledSentimentMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.max_result_bytes }
    fn encoding_record(&self) -> Option<SampledSentimentMapEncoding> {
        self.encoding.map(|limits| SampledSentimentMapEncoding {
            policy: UntrustedEncoding::SafeBpeV1(limits), work: self.encoding_work })
    }
}

struct PreparedChunk {
    id: usize,
    span: VerifiedSourceSpan,
    tokens: usize,
    plan: PreparedInt8SampledSentiment,
}
pub struct PreparedInt8SampledSentimentMap {
    chunks: Vec<PreparedChunk>,
    limits: Int8SampledSentimentMapLimits,
    expected: Int8SampledSentimentMapPreflight,
    task: TaskBudget,
    preparation_bytes: u64,
}
impl Int8SampledSentimentPlanner {
    pub fn plan_document_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: Int8SampledSentimentMapLimits, control: &mut C)
        -> Result<PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError> {
        self.plan_document_with_encoding_work(request, limits, control, &mut SafeBpeWork::default())
    }
    pub fn plan_document_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: Int8SampledSentimentMapLimits, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError> {
        self.plan_document_addressed(request, limits, None, control, work)
    }
    pub fn plan_document_for_item_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: Int8SampledSentimentMapLimits, item_id: &str, control: &mut C)
        -> Result<PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError> {
        self.plan_document_for_item_with_encoding_work(request, limits, item_id, control, &mut SafeBpeWork::default())
    }
    /// The optional item identity gives repeated documents independent sample
    /// addresses while preserving exact replay across ordering and resume.
    pub fn plan_document_for_item_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: Int8SampledSentimentMapLimits, item_id: &str, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError> {
        checkpoint(control)?;
        if item_id.is_empty() || item_id.len() > 256 || item_id.chars().any(char::is_control) {
            return Err(Int8SampledSentimentMapError::Limits);
        }
        self.plan_document_addressed(request, limits, Some(item_id), control, work)
    }
    fn plan_document_addressed<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: Int8SampledSentimentMapLimits, item_id: Option<&str>, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError> {
        checkpoint(control)?;
        limits.validate(request.budget)?; self.options.validate()?; self.limits.validate()?;
        if request.document.is_empty() { return Err(Int8SampledSentimentMapError::EmptySource); }
        if !budget_fits(request.budget, self.ceiling) || request.axes.is_empty()
            || request.axes.len() > SentimentAxis::ALL.len() { return Err(Int8SampledSentimentMapError::Limits); }
        let mut axes = request.axes.clone(); axes.sort_unstable();
        if axes.windows(2).any(|pair| pair[0] == pair[1]) { return Err(Int8SampledSentimentMapError::Limits); }
        let capacity = source_capacity(self, &axes, request.budget, limits.chunks.context_tokens)?;
        let reservation = limits.chunks.context_tokens.checked_sub(capacity).ok_or(Int8SampledSentimentMapError::Limits)?;
        let chunk_limits = ChunkLimits { max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reservation), ..limits.chunks };
        let before = *work;
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints_and_byte_cap(&request.document, chunk_limits,
                chunk_limits.max_chunk_bytes.min(self.limits.max_input_bytes), |text| {
                    if self.input_encoding() == &UntrustedEncoding::ByteFallback { return Ok(text.len()); }
                    self.compiler.encode_document(text.as_bytes(), work, &mut **control.borrow_mut())
                        .map(|document| document.ids().len()).map_err(|cause| {
                            *error.borrow_mut() = Some(cause); MapReduceError::Tokenizer
                        })
                }, || match control.borrow_mut().prefill_checkpoint(0) {
                    Some(cause) => { *error.borrow_mut() = Some(SampledSentimentError::Cancelled(cause));
                        Err(MapReduceError::Cancelled) }, None => Ok(()),
                })
        };
        if let Some(cause) = error.into_inner() { return Err(cause.into()); }
        let chunks = chunks.map_err(Int8SampledSentimentMapError::Chunk)?;
        let source_span = VerifiedSourceSpan { byte_start: 0, byte_end: request.document.len(), scalar_start: 0,
            scalar_end: chunks.chunks().last().ok_or(Int8SampledSentimentMapError::EmptySource)?.span().scalar_end };
        let document = binding::document(request, self, limits, item_id, control)?;
        let mut plans = map_reserved(chunks.chunks().len())?;
        let mut total_work = Int8Work::default();
        let mut total_output = 0_u64; let mut retained_prompt_tokens = 0_u64;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let mut text = String::new(); text.try_reserve_exact(chunk.text().len())
                .map_err(|_| Int8SampledSentimentMapError::Allocation)?;
            text.push_str(chunk.text());
            let row = SentimentRequest { document: text, axes: axes.clone(), budget: request.budget };
            let mut plan = self.plan_with_encoding_work(&row, control, work)?;
            let address = binding::chunk_address(document, chunk.id(), chunk.span(), chunk.tokens(), control)?;
            for axis in &mut plan.inner.axes {
                axis.identity.decision_policy_digest = item_address_digest(&axis.identity, &address)?;
                axis.identity.validate().map_err(|_| Int8SampledSentimentMapError::Admission)?;
                retained_prompt_tokens = retained_prompt_tokens.checked_add(axis.prompt.len() as u64)
                    .ok_or(Int8SampledSentimentMapError::Work)?;
                let template = self.compiler.axes.iter().find(|entry| entry.axis == axis.axis)
                    .ok_or(Int8SampledSentimentMapError::Admission)?;
                let scaffold = self.compiler.global.len() + template.instruction.len() + template.scaffold.len();
                if axis.prompt.len().checked_sub(scaffold) != Some(chunk.tokens()) {
                    return Err(Int8SampledSentimentMapError::Admission);
                }
            }
            plan.inner.identity.decision_policy_digest = item_address_digest(&plan.inner.identity, &address)?;
            plan.inner.identity.validate().map_err(|_| Int8SampledSentimentMapError::Admission)?;
            if plan.inner.axes.len() != axes.len() || plan.required_context() > chunk_limits.context_tokens {
                return Err(Int8SampledSentimentMapError::Admission);
            }
            total_work = total_work.checked_add(plan.planned_work())?;
            if !fits_work(total_work, limits.max_model_work) { return Err(Int8SampledSentimentMapError::Work); }
            total_output = total_output.checked_add(plan.max_output_tokens()).ok_or(Int8SampledSentimentMapError::Work)?;
            plans.push(PreparedChunk { id: chunk.id(), span: chunk.span(), tokens: chunk.tokens(), plan });
        }
        let encoding_work = work.checked_sub(before).ok_or(Int8SampledSentimentMapError::Admission)?;
        let geometry = binding::geometry(plans.iter().map(|chunk| (chunk.id, chunk.span, chunk.tokens)), control)?;
        let request_binding = binding::request(document, &plans, encoding_work, control)?;
        let expected = Int8SampledSentimentMapPreflight { source_span, chunks: plans.len(), axes: axes.len(),
            samples: self.options.samples, work: total_work, max_output_tokens: total_output,
            max_retained_bytes: limits.max_retained_result_bytes, max_result_bytes: limits.max_result_bytes,
            chunk_limits, tokenizer_checks: chunks.tokenizer_calls(), geometry,
            encoding: match self.input_encoding() { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) }, encoding_work, request_binding };
        let preparation_bytes = super::super::long::memory::prepared_bytes_bound(self.limits,
            limits.chunks, limits.max_retained_result_bytes, plans.len(), retained_prompt_tokens,
            self.encoding_workspace_bytes()?)?;
        checkpoint(control)?;
        Ok(PreparedInt8SampledSentimentMap { chunks: plans, limits, expected, task: request.budget, preparation_bytes })
    }
}
impl PreparedInt8SampledSentimentMap {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.chunks[0].plan.execution_identity() }
    pub fn task_budget(&self) -> TaskBudget { self.task }
    pub fn required_context(&self) -> usize { self.chunks.iter().map(|c| c.plan.required_context()).max().unwrap_or(0) }
    pub fn planned_work(&self) -> Int8Work { self.expected.work }
    pub fn chunk_count(&self) -> usize { self.expected.chunks }
    pub fn sample_count(&self) -> usize { self.expected.chunks * self.expected.axes * self.expected.samples }
    pub fn max_output_tokens(&self) -> u64 { self.expected.max_output_tokens }
    pub fn max_result_bytes(&self) -> u64 { self.expected.max_result_bytes }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn preflight_metadata(&self) -> Int8SampledSentimentMapPreflight { self.expected }
}

fn source_capacity(planner: &Int8SampledSentimentPlanner, axes: &[SentimentAxis], task: TaskBudget,
    context: usize) -> Result<usize, Int8SampledSentimentMapError> {
    let generations = axes.len().checked_mul(planner.options.samples).ok_or(Int8SampledSentimentMapError::Work)?;
    if generations.checked_mul(planner.options.max_sample_tokens).is_none_or(|n| n > task.max_output_tokens as usize) {
        return Err(Int8SampledSentimentMapError::Work);
    }
    let per_prompt = (planner.limits.max_total_prompt_tokens / generations)
        .min(task.max_input_tokens as usize)
        .min(context.checked_sub(planner.options.max_sample_tokens - 1).ok_or(Int8SampledSentimentMapError::Limits)?);
    let mut capacity = usize::MAX;
    for axis in axes {
        let template = planner.compiler.axes.iter().find(|entry| entry.axis == *axis)
            .ok_or(Int8SampledSentimentMapError::Admission)?;
        let overhead = planner.compiler.global.len().checked_add(template.instruction.len())
            .and_then(|n| n.checked_add(template.scaffold.len())).ok_or(Int8SampledSentimentMapError::Limits)?;
        capacity = capacity.min(per_prompt.checked_sub(overhead).filter(|&n| n != 0)
            .ok_or(Int8SampledSentimentMapError::Limits)?);
    }
    Ok(capacity)
}
fn map_reserved<T>(count: usize) -> Result<Vec<T>, Int8SampledSentimentMapError> {
    let mut values = Vec::new(); values.try_reserve_exact(count)
        .map_err(|_| Int8SampledSentimentMapError::Allocation)?; Ok(values)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int8SampledSentimentMapError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int8SampledSentimentMapError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
