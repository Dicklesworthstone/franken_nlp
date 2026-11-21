//! Complete-document sampled sentiment with explicitly paired document replicates.
//!
//! The existing lossless source partition owns every original scalar exactly
//! once. Each chunk receives distinct private sample addresses and retains its
//! complete score/explanation bundle. No missing chunk, partial bundle, or
//! generated explanation becomes source evidence or calibrated uncertainty.
use super::*;
use std::cell::RefCell;
use crate::{tasks::mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError},
    validation::grounded_fields::VerifiedSourceSpan};

mod binding;
mod execution;
mod reduction;
pub mod memory;
pub use reduction::{SampledSentimentMapAggregate, SampledSentimentMapAggregateDimension,
    SampledSentimentMapReplicate, SAMPLED_SENTIMENT_MAP_AGGREGATION};

pub const SAMPLED_SENTIMENT_MAP_EXECUTION: &str = "grouped-int4-complete-sampled-sentiment-map-v1";
pub const SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION: &str =
    "grouped-int4-complete-sampled-sentiment-map-reused-prefill-v1";
/// Stable population item addressing. Transport order, request sequence and
/// flush epochs never enter the private complete-document or chunk keys.
pub const SAMPLED_SENTIMENT_MAP_ITEM_ADDRESS_VERSION: &str =
    "sampled-sentiment-complete-document-item-address-v1";
pub const MAX_SAMPLED_SENTIMENT_MAP_CHUNKS: usize = 256;
const MAX_BYTES: u64 = 64 * MIB;
const SEMANTICS: &str = "independent-chunks-ordinal-paired-scalar-weighted-document-replicates-v1";

#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentMapLimits {
    pub chunks: ChunkLimits,
    /// Complete cold-work ceiling across every chunk, axis and sample.
    pub max_model_work: Int4Work,
    /// Cumulative complete chunk envelopes, including every explanation.
    pub max_retained_result_bytes: u64,
    /// Complete document envelope, including chunks and document replicates.
    pub max_result_bytes: u64,
}
impl SampledSentimentMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), SampledSentimentMapError> {
        self.chunks.effective_token_limit().map_err(SampledSentimentMapError::Chunk)?;
        task.validate().map_err(|_| SampledSentimentMapError::Limits)?;
        if !(1..=MAX_SAMPLED_SENTIMENT_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || self.max_model_work.forward_positions == 0 || self.max_model_work.projected_logits == 0
            || self.max_model_work.attention_pairs == 0 || self.max_model_work.projections.dot_products == 0
            || self.max_model_work.projections.multiply_accumulates == 0 {
            return Err(SampledSentimentMapError::Limits);
        }
        Ok(())
    }
}

pub enum SampledSentimentMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization, Aggregation,
    Chunk(MapReduceError), Sampled(SampledSentimentError), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for SampledSentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "sampled sentiment document limits refused",
            Self::EmptySource => "sampled sentiment document requires nonempty source",
            Self::Work => "sampled sentiment complete document work exceeds admission",
            Self::Admission => "sampled sentiment document identity or accounting differs",
            Self::Allocation => "sampled sentiment document allocation refused",
            Self::Output => "sampled sentiment complete document output exceeds admission",
            Self::Serialization => "sampled sentiment document serialization refused",
            Self::Aggregation => "sampled sentiment complete document aggregation refused",
            Self::Chunk(_) => "sampled sentiment complete source partition refused",
            Self::Sampled(_) => "sampled sentiment document task refused",
            Self::Cancelled(_) => "sampled sentiment document cancelled",
        })
    }
}
impl fmt::Debug for SampledSentimentMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for SampledSentimentMapError {}
impl From<SampledSentimentError> for SampledSentimentMapError {
    fn from(error: SampledSentimentError) -> Self { Self::Sampled(error) }
}
impl From<StrictInt4Error> for SampledSentimentMapError {
    fn from(error: StrictInt4Error) -> Self { Self::Sampled(error.into()) }
}
impl SampledSentimentMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(cause) => Some(*cause), Self::Sampled(error) => error.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SampledSentimentMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub source_tokens: usize,
    pub run: SampledSentimentRun,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SampledSentimentMapEncoding {
    pub policy: UntrustedEncoding,
    /// One cumulative ledger includes chunk probes and actual preparation.
    pub work: SafeBpeWork,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct SampledSentimentMapRun {
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
    pub samples_per_axis: usize,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub planned_mask_node_visit_charge: u64,
    pub mask_node_visit_charge: u64,
    pub generated_tokens: u64,
    pub retained_result_bytes: u64,
    pub chunks: Vec<SampledSentimentMapChunk>,
    pub aggregate: SampledSentimentMapAggregate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<SampledSentimentMapEncoding>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for SampledSentimentMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SampledSentimentMapRun").field("schema_version", &self.schema_version)
            .field("execution", &self.execution).field("group_width", &self.group_width)
            .field("source_span", &self.source_span).field("chunks", &self.chunks)
            .field("aggregate", &self.aggregate).finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub struct SampledSentimentMapPreflight {
    source_span: VerifiedSourceSpan,
    chunks: usize,
    axes: usize,
    samples: usize,
    group_width: usize,
    work: Int4Work,
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
impl SampledSentimentMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int4Work { self.work }
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
    plan: PreparedSampledSentiment,
}
/// Owned, sealed plans for every source chunk. No public result constructor
/// or supplied scoring callback can replace actual native execution.
pub struct PreparedSampledSentimentMap {
    chunks: Vec<PreparedChunk>,
    limits: SampledSentimentMapLimits,
    expected: SampledSentimentMapPreflight,
    task: TaskBudget,
    preparation_bytes: u64,
}
impl SampledSentimentPlanner {
    /// Admission-only access to the actual immutable sampling policy. The
    /// private seed is never added to a public result or diagnostic envelope.
    pub(crate) fn sampling_options(&self) -> &SampledSentimentOptions { &self.options }

    pub fn plan_document_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: SampledSentimentMapLimits, control: &mut C)
        -> Result<PreparedSampledSentimentMap, SampledSentimentMapError> {
        self.plan_document_with_encoding_work(request, limits, control, &mut SafeBpeWork::default())
    }

    /// Probes and chunk preparation spend one shared tokenizer allowance.
    /// Failed preparation retains its debit in the caller-owned ledger.
    pub fn plan_document_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: SampledSentimentMapLimits, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedSampledSentimentMap, SampledSentimentMapError> {
        self.plan_document_addressed_with_encoding_work(request, limits, None, control, work)
    }

    /// Bind a complete population document to its caller-owned stable item ID.
    /// Identical source under different IDs receives independent sample
    /// addresses; a reordered or resumed item keeps its original addresses.
    /// The one-shot document methods retain their existing identities.
    pub fn plan_document_for_item_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: SampledSentimentMapLimits, item_id: &str, control: &mut C)
        -> Result<PreparedSampledSentimentMap, SampledSentimentMapError> {
        self.plan_document_for_item_with_encoding_work(request, limits, item_id,
            control, &mut SafeBpeWork::default())
    }

    /// Retain failed preparation debits in one caller-owned compact-tokenizer
    /// ledger while binding every original chunk to this population item.
    pub fn plan_document_for_item_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: SampledSentimentMapLimits, item_id: &str, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedSampledSentimentMap, SampledSentimentMapError> {
        checkpoint(control)?;
        if item_id.is_empty() || item_id.len() > 256 || item_id.chars().any(char::is_control) {
            return Err(SampledSentimentMapError::Limits);
        }
        self.plan_document_addressed_with_encoding_work(request, limits, Some(item_id), control, work)
    }

    fn plan_document_addressed_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        limits: SampledSentimentMapLimits, item_id: Option<&str>, control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedSampledSentimentMap, SampledSentimentMapError> {
        checkpoint(control)?;
        limits.validate(request.budget)?;
        self.options.validate()?; self.limits.validate()?;
        if request.document.is_empty() { return Err(SampledSentimentMapError::EmptySource); }
        if !budget_fits(request.budget, self.ceiling) || request.axes.is_empty()
            || request.axes.len() > SentimentAxis::ALL.len() { return Err(SampledSentimentMapError::Limits); }
        let mut axes = request.axes.clone(); axes.sort_unstable();
        if axes.windows(2).any(|pair| pair[0] == pair[1]) { return Err(SampledSentimentMapError::Limits); }
        let capacity = source_capacity(self, &axes, request.budget, limits.chunks.context_tokens)?;
        let reservation = limits.chunks.context_tokens.checked_sub(capacity).ok_or(SampledSentimentMapError::Limits)?;
        let chunk_limits = ChunkLimits { max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reservation), ..limits.chunks };
        let before = *work;
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints_and_byte_cap(&request.document, chunk_limits,
                chunk_limits.max_chunk_bytes.min(self.limits.max_input_bytes),
                |text| {
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
        let chunks = chunks.map_err(SampledSentimentMapError::Chunk)?;
        let source_span = VerifiedSourceSpan { byte_start: 0, byte_end: request.document.len(),
            scalar_start: 0, scalar_end: chunks.chunks().last().ok_or(SampledSentimentMapError::EmptySource)?.span().scalar_end };
        // Hash the complete source once, never once per chunk. The private
        // binding also names the aggregate contract and immutable task caps.
        let document_binding = binding::document(request, self, limits, control)?;
        let document_binding = match item_id {
            Some(item_id) => binding::item_document(document_binding, item_id, control)?,
            None => document_binding,
        };
        let mut plans = map_reserved(chunks.chunks().len())?;
        let mut total_work = Int4Work::default();
        let mut total_output = 0_u64;
        let mut retained_prompt_tokens = 0_u64;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            let mut text = String::new(); text.try_reserve_exact(chunk.text().len()).map_err(|_| SampledSentimentMapError::Allocation)?;
            text.push_str(chunk.text());
            let row = SentimentRequest { document: text, axes: axes.clone(), budget: request.budget };
            let mut plan = self.plan_with_encoding_work(&row, control, work)?;
            let address = binding::chunk_address(document_binding, chunk.id(), chunk.span(), chunk.tokens(), control)?;
            for axis in &mut plan.axes {
                axis.identity.decision_policy_digest = item_address_digest(&axis.identity, &address)?;
                axis.identity.validate().map_err(|_| SampledSentimentMapError::Admission)?;
                retained_prompt_tokens = retained_prompt_tokens.checked_add(axis.prompt.len() as u64)
                    .ok_or(SampledSentimentMapError::Work)?;
            }
            plan.identity.decision_policy_digest = item_address_digest(&plan.identity, &address)?;
            plan.identity.validate().map_err(|_| SampledSentimentMapError::Admission)?;
            if plan.group_width() != self.group_width || plan.axes.len() != axes.len()
                || plan.required_context() > chunk_limits.context_tokens {
                return Err(SampledSentimentMapError::Admission);
            }
            for axis in &plan.axes {
                let template = self.compiler.axes.iter().find(|entry| entry.axis == axis.axis)
                    .ok_or(SampledSentimentMapError::Admission)?;
                let scaffold = self.compiler.global.len() + template.instruction.len() + template.scaffold.len();
                if axis.prompt.len().checked_sub(scaffold) != Some(chunk.tokens()) {
                    return Err(SampledSentimentMapError::Admission);
                }
            }
            total_work = total_work.checked_add(plan.planned_work())?;
            if !fits_work(total_work, limits.max_model_work) { return Err(SampledSentimentMapError::Work); }
            let output = (plan.sample_count() as u64).checked_mul(self.options.max_sample_tokens as u64)
                .ok_or(SampledSentimentMapError::Work)?;
            total_output = total_output.checked_add(output).ok_or(SampledSentimentMapError::Work)?;
            plans.push(PreparedChunk { id: chunk.id(), span: chunk.span(), tokens: chunk.tokens(), plan });
        }
        let encoding_work = work.checked_sub(before).ok_or(SampledSentimentMapError::Admission)?;
        let geometry = binding::geometry(&plans, control)?;
        let request_binding = binding::request(document_binding, &plans, encoding_work, control)?;
        let expected = SampledSentimentMapPreflight { source_span, chunks: plans.len(), axes: axes.len(),
            samples: self.options.samples, group_width: self.group_width, work: total_work,
            max_output_tokens: total_output, max_retained_bytes: limits.max_retained_result_bytes,
            max_result_bytes: limits.max_result_bytes, chunk_limits, tokenizer_checks: chunks.tokenizer_calls(),
            geometry, encoding: match self.input_encoding() { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) }, encoding_work, request_binding };
        let preparation_bytes = memory::prepared_bytes(self.limits, limits, plans.len(),
            retained_prompt_tokens, self.compiler.encoding_workspace_bytes()?)?;
        checkpoint(control)?;
        Ok(PreparedSampledSentimentMap { chunks: plans, limits, expected, task: request.budget, preparation_bytes })
    }
}
impl PreparedSampledSentimentMap {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.chunks[0].plan.execution_identity() }
    pub fn group_width(&self) -> usize { self.expected.group_width }
    pub fn task_budget(&self) -> TaskBudget { self.task }
    pub fn required_context(&self) -> usize { self.chunks.iter().map(|chunk| chunk.plan.required_context()).max().unwrap_or(0) }
    pub fn planned_work(&self) -> Int4Work { self.expected.work }
    pub fn chunk_count(&self) -> usize { self.expected.chunks }
    pub fn sample_count(&self) -> usize { self.expected.chunks * self.expected.axes * self.expected.samples }
    pub fn max_output_tokens(&self) -> u64 { self.expected.max_output_tokens }
    pub fn max_result_bytes(&self) -> u64 { self.expected.max_result_bytes }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn preflight_metadata(&self) -> SampledSentimentMapPreflight { self.expected }
}

fn source_capacity(planner: &SampledSentimentPlanner, axes: &[SentimentAxis], task: TaskBudget,
    context: usize) -> Result<usize, SampledSentimentMapError> {
    let mut capacity = usize::MAX;
    let generations = axes.len().checked_mul(planner.options.samples).ok_or(SampledSentimentMapError::Work)?;
    if generations.checked_mul(planner.options.max_sample_tokens).is_none_or(|n| n > task.max_output_tokens as usize) {
        return Err(SampledSentimentMapError::Work);
    }
    let per_prompt = (planner.limits.max_total_prompt_tokens / generations)
        .min(task.max_input_tokens as usize)
        .min(context.checked_sub(planner.options.max_sample_tokens - 1).ok_or(SampledSentimentMapError::Limits)?);
    for axis in axes {
        let template = planner.compiler.axes.iter().find(|entry| entry.axis == *axis)
            .ok_or(SampledSentimentMapError::Admission)?;
        let overhead = planner.compiler.global.len().checked_add(template.instruction.len())
            .and_then(|n| n.checked_add(template.scaffold.len())).ok_or(SampledSentimentMapError::Limits)?;
        capacity = capacity.min(per_prompt.checked_sub(overhead).filter(|&n| n != 0)
            .ok_or(SampledSentimentMapError::Limits)?);
    }
    Ok(capacity)
}
fn map_reserved<T>(count: usize) -> Result<Vec<T>, SampledSentimentMapError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| SampledSentimentMapError::Allocation)?; Ok(values)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), SampledSentimentMapError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(SampledSentimentMapError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
