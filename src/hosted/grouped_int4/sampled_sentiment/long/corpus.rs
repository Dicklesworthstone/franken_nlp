//! Complete sampled documents on one owned resident invocation.
//!
//! Chunk plans, independent draws and all evidence are per-document. Native
//! buffers, the pinned compiler/vocabulary, transport and six work ceilings
//! belong to the whole stream. A flush cannot renew any of those ceilings.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchLimits, BatchSummary,
        extract::ExtractionMaskBudget,
        sentiment::sampled::{SampledSentimentBatchLimits,
            long::{NativeSampledSentimentDocumentBatch, SampledSentimentDocumentAdmission,
                SampledSentimentDocumentBatchAdmission, SampledSentimentDocumentPlanner}}},
    hosted::corpus::CorpusLimits,
    native_engine::constrained_int4,
};

/// The document policy is fixed for the complete invocation. Records may
/// select axes and tighten their per-chunk task budget; they cannot replace
/// the seed, chunk rules, native schedule or whole-run work/mask allowance.
/// The seed stays private through the sampled options' redacted Debug view.
#[derive(Clone, Debug)]
pub struct SampledSentimentDocumentBatchHostConfig {
    pub document: SampledSentimentMapHostConfig,
    pub task: TaskBudget,
    pub encoding: UntrustedEncoding,
    pub transport: BatchLimits,
    pub io_reserve_bytes: u64,
    pub max_model_work: Int4Work,
    pub max_mask_node_visits: u64,
}
impl SampledSentimentDocumentBatchHostConfig {
    /// Pure complete peak quote before input/model IO. One map quote owns
    /// all plans and chunk evidence; transport and supplied IO are added once.
    /// Output remains a per-document guard through serialization and flush.
    pub fn requirements(&self, group_width: usize)
        -> Result<SampledSentimentMapMemoryRequirement, HostedError> {
        self.work_limits().validate().map_err(HostedError::BatchSetup)?;
        let mut required = self.document.requirements_with_encoding(group_width,
            self.task, SentimentAxis::ALL.len(), &self.encoding)?;
        if self.document.mapping.max_result_bytes > self.transport.max_output_line_bytes as u64 {
            return Err(HostedError::Limits("complete sampled document output exceeds stream line cap"));
        }
        let transport = CorpusLimits { native: self.document.native, transport: self.transport,
            preparation_reserve_bytes: self.document.preparation_reserve_bytes,
            io_reserve_bytes: self.io_reserve_bytes }.reservation_bytes()?;
        // The map quote already includes the compiler/preparation reserve.
        let extra = transport.checked_sub(self.document.preparation_reserve_bytes)
            .ok_or(HostedError::Limits("sampled document stream preparation arithmetic"))?;
        required.input_bytes = sum(&[required.input_bytes, extra])?;
        sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }

    /// The physical team is additional workspace, not additional work
    /// authority. This quote rejects team/prefill conflicts before any IO.
    pub fn requirements_with_threads(&self, group_width: usize, compute_threads: usize)
        -> Result<SampledSentimentMapMemoryRequirement, HostedError> {
        check_schedule(&self.document, Some(compute_threads))?;
        let mut required = self.requirements(group_width)?;
        required.workspace_bytes = sum(&[required.workspace_bytes,
            int4_task_team_workspace_bytes(compute_threads, group_width)?])?;
        sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }

    pub(super) fn work_limits(&self) -> SampledSentimentBatchLimits {
        SampledSentimentBatchLimits {
            max_item_work: self.document.mapping.max_model_work,
            max_model_work: self.max_model_work,
            masks: ExtractionMaskBudget { per_mask: self.document.mask_limits,
                max_visits_per_item: self.document.max_mask_node_visits,
                max_visits_per_run: self.max_mask_node_visits },
            max_sampler_bytes: self.document.max_sampler_bytes,
        }
    }
}

struct StreamInput<R, W> {
    config: SampledSentimentDocumentBatchHostConfig,
    reader: R,
    writer: W,
}

// Separate shared ownership keeps every process charge and the resource host
// alive through physical join, including coordinator error or unwind. The
// team's finalizer owns another Arc outside the scoped computation itself.
struct StreamOwners {
    _input: CommittedMemory,
    _kv: CommittedMemory,
    _scratch: CommittedMemory,
    lease: EngineLease,
}
struct StreamExecution<R, W> {
    input: StreamInput<R, W>,
    required: SampledSentimentMapMemoryRequirement,
    owners: Arc<StreamOwners>,
}
struct DeferredDocumentStream<R, W> {
    completion: Result<stream::deferred::DeferredCompletion, HostedError>,
    input: StreamInput<R, W>,
    _owners: Arc<StreamOwners>,
}

impl NlpEngine {
    /// Score complete original documents from bounded ordered NDJSON while
    /// keeping one authenticated model, compiler, vocabulary and native engine
    /// resident. Only a complete validated map publishes a document result.
    /// Original item ids address the draws; sequence and flush do not.
    ///
    /// Supplied reader/writer storage must fit the explicit IO reservation.
    /// IO is cooperative and cannot be safely preempted inside an OS call.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment_documents<R, W>(&self, model: &ResidentInt4,
        config: SampledSentimentDocumentBatchHostConfig, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_documents(model, config, kernel, None,
            reader, writer, cancellation)
    }

    /// One fixed numerical row team spans every document, chunk, axis and
    /// sample. It joins before the terminal footer is written. Input, native
    /// charges and the original cancellation allowance survive finalization.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment_documents_with_threads<R, W>(&self, model: &ResidentInt4,
        config: SampledSentimentDocumentBatchHostConfig, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_documents(model, config, kernel, Some(compute_threads),
            reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sampled_sentiment_documents<R, W>(&self, model: &ResidentInt4,
        config: SampledSentimentDocumentBatchHostConfig, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        check_schedule(&config.document, compute_threads)?;
        match compute_threads {
            None => dispatch::preflight(self, config.document.native.run)?,
            Some(width) => dispatch::preflight_int4_team(self, config.document.native.run, width)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        let required = config.requirements(width)?;
        if let Some(threads) = compute_threads { config.requirements_with_threads(width, threads)?; }
        let run = config.document.native.run;
        let lease = self.resources().acquire_lease();
        let input = Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let owners = Arc::new(StreamOwners { _input: input.commit()?, _kv: kv.commit()?,
            _scratch: scratch.commit()?, lease });
        let execution = StreamExecution { input: StreamInput { config, reader, writer },
            required, owners: Arc::clone(&owners) };
        let model = model.clone();
        match compute_threads {
            None => dispatch::run(self, run, cancellation, move |control| {
                let pending = execution.run(&model.inner.loaded.value, kernel, None, control);
                let result = pending.and_then(|pending| pending.publish(control));
                drop(model); drop(owners);
                result
            }),
            Some(threads) => dispatch::run_int4_team_with_finalize(self, run, cancellation,
                model, threads, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control),
                move |pending, control| {
                    let result = pending.publish(control);
                    drop(owners);
                    result
                }),
        }
    }
}

impl<R: BufRead, W: Write> StreamExecution<R, W> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<DeferredDocumentStream<R, W>, HostedError> {
        let Self { mut input, required, owners } = self;
        // Even a task/transport failure is returned with its charged owners so
        // the team can join before the original error escapes the invocation.
        let completion = (|| {
            checkpoint(control)?;
            let config = &input.config;
            let planner = SampledSentimentPlanner::for_model_with_encoding(weights.identity(),
                config.document.options.clone(), config.task, config.document.planning,
                config.encoding.clone()).map_err(task_error)?;
            let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().map_err(task_error)?);
            let planner = SampledSentimentDocumentPlanner::new(planner, vocabulary, config.task,
                config.document.mapping, config.work_limits()).map_err(HostedError::BatchSetup)?;
            if planner.preparation_requirement_bytes() > required.input_bytes {
                return Err(HostedError::Limits("sampled document planner exceeds its input reservation"));
            }
            let mut engine = StrictInt4Engine::new(weights, config.document.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map_err(native_error)?.with_kernel(kernel);
            if let Some(team) = projection {
                engine = engine.with_projection_executor(team).map_err(native_error)?;
            }
            let admission = DocumentAdmission::for_job(&owners.lease, weights.identity(),
                config.document.native.context_tokens, required.native.kv_bytes,
                config.document.mapping, config.work_limits(), required.input_bytes,
                required.output_reservation_bytes, required.max_output_bytes, required.max_output_tokens);
            let result = {
                let mut processor = if config.document.reuse_prefill {
                    NativeSampledSentimentDocumentBatch::new_reusing_prefill(planner,
                        &mut engine, admission, config.document.prefill)
                } else { match config.document.prefill {
                    None => NativeSampledSentimentDocumentBatch::new(planner, &mut engine, admission),
                    Some(prefill) => NativeSampledSentimentDocumentBatch::new_layer_major(
                        planner, &mut engine, admission, prefill),
                } }.map_err(HostedError::BatchSetup)?;
                stream::deferred::run_ndjson_deferred(&mut input.reader, &mut input.writer,
                    &mut processor, config.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            result
        })();
        Ok(DeferredDocumentStream { completion, input, _owners: owners })
    }
}
impl<R, W: Write> DeferredDocumentStream<R, W> {
    fn publish(self, control: &mut RunControl) -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input, _owners } = self;
        let result = completion.and_then(|completion|
            completion.publish(&mut input.writer, control).map_err(HostedError::Batch));
        drop(input);
        drop(_owners);
        result
    }
}

fn check_schedule(config: &SampledSentimentMapHostConfig, threads: Option<usize>) -> Result<(), HostedError> {
    if threads.is_some() && config.prefill.is_some() {
        return Err(HostedError::Limits("sampled document stream team/prefill schedule"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct DocumentCapacity {
    context: usize,
    kv_bytes: u64,
    mapping: SampledSentimentMapLimits,
    work: SampledSentimentBatchLimits,
    preparation_bytes: u64,
    output_reservation_bytes: u64,
    max_output_bytes: u64,
    max_output_tokens: u64,
}
pub(super) struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    capacity: DocumentCapacity,
}
impl<'a> DocumentAdmission<'a> {
    /// Shared by retained jobs: the same exact document admission owns the
    /// result until the durable spool/journal acknowledgement is complete.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_job(lease: &'a EngineLease, model: &'a Int4ModelIdentity,
        context: usize, kv_bytes: u64, mapping: SampledSentimentMapLimits,
        work: SampledSentimentBatchLimits, preparation_bytes: u64,
        output_reservation_bytes: u64, max_output_bytes: u64, max_output_tokens: u64) -> Self {
        Self { lease, model, capacity: DocumentCapacity { context, kv_bytes, mapping, work,
            preparation_bytes, output_reservation_bytes, max_output_bytes, max_output_tokens } }
    }
}
impl SampledSentimentDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, Pending), BatchItemFailure> {
        constrained_int4::check_model(request.facts.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_request(self.model.group_width(), self.capacity, &request)?;
        let guard = output_claim(self.lease, request.facts.max_result_bytes, request.facts.max_output_tokens)
            .map_err(|error| match error { HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution) })?;
        Ok((request.facts.identity.clone(), guard))
    }
}

fn check_request(width: usize, capacity: DocumentCapacity, request: &SampledSentimentDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let facts = &request.facts;
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    constrained_int4::check_profile(facts.identity, width).map_err(|_| fail())?;
    if facts.identity.task_spec != "sentiment-v1"
        || facts.identity.sampler_version != constrained_int4::JSON_SAMPLER_VERSION
        || facts.group_width != width || request.kv_reservation_bytes != capacity.kv_bytes
        || facts.original_task.validate().is_err() || facts.preparation_bytes == 0
        || facts.preparation_bytes > capacity.preparation_bytes {
        return Err(fail());
    }
    let native = Int4MemoryRequirement::for_context(capacity.context, width).map_err(|_| fail())?;
    if native.kv_bytes != capacity.kv_bytes || native.kv_bytes > facts.original_task.max_kv_bytes
        || facts.required_context == 0 || facts.required_context > capacity.context
        || facts.chunk_count == 0 || facts.chunk_count > capacity.mapping.chunks.max_chunks
        || facts.sample_count == 0 || facts.model_work.forward_positions == 0
        || facts.model_work.projected_logits == 0 || facts.model_work.attention_pairs == 0
        || facts.model_work.projections.dot_products == 0 || facts.model_work.projections.multiply_accumulates == 0
        || !fits(facts.model_work, capacity.work.max_item_work)
        || !fits(facts.model_work, capacity.mapping.max_model_work)
        || request.mask_node_visits == 0 || request.mask_node_visits > capacity.work.masks.max_visits_per_item {
        return Err(fail());
    }
    let output = facts.max_result_bytes.checked_mul(4)
        .and_then(|bytes| bytes.checked_add(facts.max_output_tokens.checked_mul(8)?))
        .and_then(|bytes| bytes.checked_add(4096)).ok_or_else(fail)?;
    if facts.max_result_bytes == 0 || facts.max_result_bytes != capacity.mapping.max_result_bytes
        || facts.max_result_bytes > capacity.max_output_bytes || facts.max_output_tokens == 0
        || facts.max_output_tokens > capacity.max_output_tokens || output > capacity.output_reservation_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

#[cfg(test)] mod tests;
