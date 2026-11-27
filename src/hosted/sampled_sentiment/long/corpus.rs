//! Complete sampled INT8 documents on one admitted resident invocation.
//! Native buffers, the compiler, vocabulary and all six work ceilings belong
//! to the whole stream. The final footer waits for physical worker completion.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchLimits, BatchSummary,
        extract::ExtractionMaskBudget,
        sentiment::sampled::quantized::{Int8SampledSentimentBatchLimits,
            long::{Int8SampledSentimentDocumentAdmission, Int8SampledSentimentDocumentBatchAdmission,
                Int8SampledSentimentDocumentPlanner, NativeInt8SampledSentimentDocumentBatch}}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int8, strict_int8::projection::Int8ProjectionExecutor},
};

/// Records may choose axes and tighten the task budget. The source policy,
/// seed, schedule and whole-stream native/mask allowances stay fixed.
#[derive(Clone, Debug)]
pub struct Int8SampledSentimentDocumentBatchHostConfig {
    pub document: Int8SampledSentimentMapHostConfig,
    pub task: TaskBudget,
    pub encoding: UntrustedEncoding,
    pub transport: BatchLimits,
    pub io_reserve_bytes: u64,
    pub max_model_work: Int8Work,
    pub max_mask_node_visits: u64,
}
impl Int8SampledSentimentDocumentBatchHostConfig {
    /// Pure complete peak quote before any input/model IO. One document's
    /// plans and evidence coexist with transport and caller-supplied IO.
    pub fn requirements(&self) -> Result<Int8SampledSentimentMapMemoryRequirement, HostedError> {
        self.work_limits().validate().map_err(HostedError::BatchSetup)?;
        let mut required = self.document.requirements_with_encoding(self.task,
            SentimentAxis::ALL.len(), &self.encoding)?;
        if self.document.mapping.max_result_bytes > self.transport.max_output_line_bytes as u64 {
            return Err(HostedError::Limits("complete INT8 sampled document output exceeds stream line cap"));
        }
        let transport = CorpusLimits { native: self.document.native, transport: self.transport,
            preparation_reserve_bytes: self.document.preparation_reserve_bytes,
            io_reserve_bytes: self.io_reserve_bytes }.reservation_bytes()?;
        // Preparation is already part of the complete-document input quote.
        let extra = transport.checked_sub(self.document.preparation_reserve_bytes)
            .ok_or(HostedError::Limits("INT8 sampled document stream preparation arithmetic"))?;
        required.input_bytes = sum(&[required.input_bytes, extra])?;
        sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }

    /// Prompt morsels and decode share one fixed numerical team. Its worker
    /// rows are additional physical workspace, with no additional work grant.
    pub fn requirements_with_threads(&self, compute_threads: usize)
        -> Result<Int8SampledSentimentMapMemoryRequirement, HostedError> {
        let mut required = self.requirements()?;
        let rows = self.document.prefill.map_or(1, |limits| limits.max_batch_rows);
        required.workspace_bytes = sum(&[required.workspace_bytes,
            dispatch::int8_prefill_team_workspace_bytes(compute_threads, rows)?])?;
        sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }

    pub(super) fn work_limits(&self) -> Int8SampledSentimentBatchLimits {
        Int8SampledSentimentBatchLimits {
            max_item_work: self.document.mapping.max_model_work, max_model_work: self.max_model_work,
            masks: ExtractionMaskBudget { per_mask: self.document.mask_limits,
                max_visits_per_item: self.document.max_mask_node_visits,
                max_visits_per_run: self.max_mask_node_visits },
            max_sampler_bytes: self.document.max_sampler_bytes,
        }
    }
}

struct StreamInput<R, W> {
    // The actual compiler/vocabulary allocations survive processor drop,
    // cancellation and unwind until all native workers physically join.
    _retained: Option<(Arc<Int8SampledSentimentPlanner>, Arc<SampledSentimentVocabulary>)>,
    config: Int8SampledSentimentDocumentBatchHostConfig,
    reader: R,
    writer: W,
}
struct StreamExecution<R, W> {
    input: Charged<StreamInput<R, W>>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    required: Int8SampledSentimentMapMemoryRequirement,
    lease: EngineLease,
}

impl NlpEngine {
    /// Keep one authenticated INT8 model and native engine resident across
    /// complete original documents in ordered NDJSON. Stable original ids
    /// address independent draws; sequence and flush do not change them.
    /// Supplied IO storage must fit the explicit IO reservation.
    pub fn batch_int8_sampled_sentiment_documents<R, W>(&self, model: &ResidentInt8,
        config: Int8SampledSentimentDocumentBatchHostConfig, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, config.document.native.run)?;
        self.check_resident_domain(model)?;
        let required = config.requirements()?;
        let run = config.document.native.run;
        let mut execution = admit_stream(self, config, reader, writer, required)?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let completion = execution.run(&model.inner.loaded.value, None, None, control)?;
            completion.publish(&mut execution.input.value.writer, control).map_err(HostedError::Batch)
        })
    }

    /// The same fixed workers execute prompt morsels and sampled tokens for
    /// the entire stream. The EOF footer is published only after they join,
    /// with the original deadline, input owners and process charges retained.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sampled_sentiment_documents_with_threads<R, W>(&self, model: &ResidentInt8,
        config: Int8SampledSentimentDocumentBatchHostConfig, kernel: Int8Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight_int8_team(self, config.document.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        let required = config.requirements()?;
        config.requirements_with_threads(compute_threads)?;
        let run = config.document.native.run;
        let rows = config.document.prefill.map_or(1, |limits| limits.max_batch_rows);
        let execution = admit_stream(self, config, reader, writer, required)?;
        dispatch::run_int8_prefill_team_with_finalize(self, run, cancellation, model.clone(),
            compute_threads, kernel, rows, execution,
            move |model, execution, control, team| execution.run(model, Some(kernel), Some(team), control),
            |completion, execution, control| {
                completion.publish(&mut execution.input.value.writer, control).map_err(HostedError::Batch)
            })
    }
}

fn admit_stream<R, W>(host: &NlpEngine, config: Int8SampledSentimentDocumentBatchHostConfig,
    reader: R, writer: W, required: Int8SampledSentimentMapMemoryRequirement)
    -> Result<StreamExecution<R, W>, HostedError> {
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
        || Ok(StreamInput { _retained: None, config, reader, writer }))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
    Ok(StreamExecution { input, kv: Some(kv), scratch: Some(scratch), _kv: None, _scratch: None,
        required, lease })
}

impl<R: BufRead, W: Write> StreamExecution<R, W> {
    fn run<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model, kernel: Option<Int8Kernel>,
        projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>, control: &mut RunControl)
        -> Result<stream::deferred::DeferredCompletion, HostedError> {
        checkpoint(control)?;
        let config = self.input.value.config.clone();
        let planner = Int8SampledSentimentPlanner::for_model_with_encoding(model.artifact_identity(),
            config.document.options.clone(), config.task, config.document.planning, config.encoding.clone())
            .map_err(HostedError::SampledSentiment)?;
        let vocabulary = Arc::new(SampledSentimentVocabulary::pinned()
            .map_err(Int8SampledSentimentError::from).map_err(HostedError::SampledSentiment)?);
        let planner = Int8SampledSentimentDocumentPlanner::new(planner, vocabulary, config.task,
            config.document.mapping, config.work_limits()).map_err(HostedError::BatchSetup)?;
        if planner.preparation_requirement_bytes() > self.required.input_bytes {
            return Err(HostedError::Limits("INT8 sampled document planner exceeds its input reservation"));
        }
        self.input.value._retained = Some(planner.retained_owners());
        checkpoint(control)?;
        let mut engine = model.engine(config.document.native.context_tokens, memory_budget(self.required.native))
            .map_err(HostedError::Model)?;
        if let Some(kernel) = kernel { engine = engine.with_kernel(kernel); }
        if let Some(team) = projection {
            engine = engine.with_projection_executor(team).map_err(HostedError::Native)?;
        }
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("INT8 sampled stream KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("INT8 sampled stream scratch admission consumed"))?.commit()?);
        let admission = DocumentAdmission::for_job(&self.lease, model.artifact_identity(),
            config.document.native.context_tokens, self.required.native.kv_bytes,
            config.document.mapping, config.work_limits(), self.required.input_bytes,
            self.required.output_reservation_bytes, self.required.max_output_bytes,
            self.required.max_output_tokens, &config.document.options);
        let result = {
            let mut processor = match config.document.prefill {
                None => NativeInt8SampledSentimentDocumentBatch::new(planner, &mut engine, admission),
                Some(prefill) => NativeInt8SampledSentimentDocumentBatch::new_layer_major(
                    planner, &mut engine, admission, prefill),
            }.map_err(HostedError::BatchSetup)?;
            let input = &mut self.input.value;
            stream::deferred::run_ndjson_deferred(&mut input.reader, &mut input.writer,
                &mut processor, config.transport, control).map_err(HostedError::Batch)
        };
        if result.is_ok() && (engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)) {
            return Err(HostedError::Limits("INT8 sampled document stream native completion"));
        }
        drop(engine);
        result
    }
}

#[derive(Clone, Copy)]
pub(super) struct DocumentCapacity {
    context: usize,
    kv_bytes: u64,
    mapping: Int8SampledSentimentMapLimits,
    work: Int8SampledSentimentBatchLimits,
    preparation_bytes: u64,
    output_reservation_bytes: u64,
    max_output_bytes: u64,
    max_output_tokens: u64,
    samples_per_axis: usize,
    max_sample_tokens: usize,
}
pub(super) struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a ArtifactIdentity,
    capacity: DocumentCapacity,
}
impl<'a> DocumentAdmission<'a> {
    /// Retained jobs use the same admission and retain the guarded complete
    /// output through spool synchronization and journal acknowledgement.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_job(lease: &'a EngineLease, model: &'a ArtifactIdentity,
        context: usize, kv_bytes: u64, mapping: Int8SampledSentimentMapLimits,
        work: Int8SampledSentimentBatchLimits, preparation_bytes: u64,
        output_reservation_bytes: u64, max_output_bytes: u64, max_output_tokens: u64,
        options: &SampledSentimentOptions) -> Self {
        Self { lease, model, capacity: DocumentCapacity { context, kv_bytes, mapping, work,
            preparation_bytes, output_reservation_bytes, max_output_bytes, max_output_tokens,
            samples_per_axis: options.samples, max_sample_tokens: options.max_sample_tokens } }
    }
}
impl Int8SampledSentimentDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, Pending), BatchItemFailure> {
        check_model_identity(self.model, request.facts.identity)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_request(self.capacity, &request)?;
        let guard = output_claim(self.lease, request.facts.max_result_bytes, request.facts.max_output_tokens)
            .map_err(|error| match error {
                HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
            })?;
        Ok((request.facts.identity.clone(), guard))
    }
}

fn check_request(capacity: DocumentCapacity, request: &Int8SampledSentimentDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let facts = &request.facts;
    let fail = || BatchItemFailure::fatal(BatchCode::Admission);
    constrained_int8::check_profile(facts.identity).map_err(|_| fail())?;
    if facts.identity.task_spec != "sentiment-v1"
        || facts.identity.sampler_version != constrained_int8::JSON_SAMPLER_VERSION
        || request.kv_reservation_bytes != capacity.kv_bytes
        || facts.original_task.validate().is_err() || facts.preparation_bytes == 0
        || facts.preparation_bytes > capacity.preparation_bytes {
        return Err(fail());
    }
    let native = Int8MemoryRequirement::for_context(capacity.context).map_err(|_| fail())?;
    let samples_per_chunk_set = facts.chunk_count.checked_mul(capacity.samples_per_axis).ok_or_else(fail)?;
    let axes = facts.sample_count.checked_div(samples_per_chunk_set).ok_or_else(fail)?;
    let tokens = (facts.sample_count as u64).checked_mul(capacity.max_sample_tokens as u64).ok_or_else(fail)?;
    let logits = tokens.checked_mul(crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64).ok_or_else(fail)?;
    if native.kv_bytes != capacity.kv_bytes || native.kv_bytes > facts.original_task.max_kv_bytes
        || facts.required_context == 0 || facts.required_context > capacity.context
        || facts.chunk_count == 0 || facts.chunk_count > capacity.mapping.chunks.max_chunks
        || !(1..=SentimentAxis::ALL.len()).contains(&axes)
        || facts.sample_count % samples_per_chunk_set != 0 || facts.max_output_tokens != tokens
        || facts.model_work.forward_positions == 0 || facts.model_work.projected_logits != logits
        || facts.model_work.attention_pairs == 0 || facts.model_work.projections.dot_products == 0
        || facts.model_work.projections.multiply_accumulates == 0
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
