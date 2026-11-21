//! Owned sampled-sentiment populations on one resident native allocation.
//! The reader, writer, compiler, vocabulary and their charges remain in the
//! physical blocking invocation. No record creates another runtime or team.

use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        sentiment::sampled::{NativeSampledSentimentBatch, SampledSentimentAdmission,
            SampledSentimentBatchAdmission, SampledSentimentBatchLimits, SampledSentimentBatchPlanner}},
    hosted::corpus::CorpusLimits,
    native_engine::constrained_int4,
};

struct StreamInput<R, W> { planner: Option<SampledSentimentBatchPlanner>, reader: R, writer: W }
// On failure the pending footer drops before the charged private input/IO.
struct DeferredSampledStream<R, W> {
    completion: stream::deferred::DeferredCompletion,
    input: Charged<StreamInput<R, W>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements { native: Int4MemoryRequirement, workspace_bytes: u64, input_bytes: u64 }

impl NlpEngine {
    /// Execute bounded ordered {id,text,task_args?} records with one fixed
    /// independently sampled sentiment policy. The immutable vocabulary and
    /// native engine are reused; every axis/sample owns a clean logical KV.
    ///
    /// Input and output are owned through delivery. Work/mask ceilings include
    /// failed admitted attempts and never renew at record or flush boundaries.
    /// The task planner's preparation bound must fit the explicit corpus
    /// reservation before the host reads any private population bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment<R, W>(&self, model: &ResidentInt4,
        planner: SampledSentimentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_corpus(model, planner, limits, kernel,
            None, None, false, reader, writer, cancellation)
    }

    /// Execute each fresh sample prompt in bounded causal layer-major morsels.
    /// One resident engine and one admitted prompt workspace serve the complete
    /// stream; input order, item seeds and aggregate work allowances are unchanged.
    /// Generation remains sequential, and success is published after native drain.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment_layer_major<R, W>(&self, model: &ResidentInt4,
        planner: SampledSentimentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_corpus(model, planner, limits, kernel,
            None, Some(prefill), false, reader, writer, cancellation)
    }

    /// Retain only the current document axis's causal prompt between its
    /// independently addressed samples. Sample continuations are discarded
    /// before the next draw; no prefix crosses an axis or document boundary.
    /// Cold worst-case native/mask allowances still cover every admitted
    /// attempt. Each completed bundle reports the work actually performed.
    /// An optional prompt morsel applies to each axis's first prefill only.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment_reusing_prefill<R, W>(&self, model: &ResidentInt4,
        planner: SampledSentimentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        prefill: Option<Int4PrefillLimits>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_corpus(model, planner, limits, kernel,
            None, prefill, true, reader, writer, cancellation)
    }

    /// Preserve the identical item semantics with one persistent numerical
    /// output-row team. Its total width includes the coordinator and must be
    /// 2..=64. Every worker joins before the terminal run_complete is published;
    /// the original control and charged owned IO survive that finalization.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: SampledSentimentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_corpus(model, planner, limits, kernel,
            Some(compute_threads), None, false, reader, writer, cancellation)
    }

    /// Combine per-axis prompt reuse with one persistent bounded projection
    /// team. Native cleanup and the team's join precede terminal publication;
    /// reuse changes no admission, cancellation or output ownership boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sampled_sentiment_reusing_prefill_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: SampledSentimentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sampled_sentiment_corpus(model, planner, limits, kernel,
            Some(compute_threads), None, true, reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sampled_sentiment_corpus<R, W>(&self, model: &ResidentInt4,
        planner: SampledSentimentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, prefill: Option<Int4PrefillLimits>, reuse_prefill: bool,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        if compute_threads.is_some() && prefill.is_some() {
            return Err(HostedError::Limits("sampled corpus prefill and compute teams are separate schedules"));
        }
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        let work = planner.limits();
        let selected_rows = planner.selected_rows();
        let required = scheduled_requirements(model.model_identity().group_width(), work, planner.task_budget(),
            planner.preparation_requirement_bytes(), limits, prefill)?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, model.model_identity().group_width()))
            .transpose()?.unwrap_or(0);
        sum(&[required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { planner: Some(planner), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        match compute_threads {
        None => dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = Admission { lease: &lease, model: model.model_identity(),
                context: limits.native.context_tokens, kv_bytes: required.native.kv_bytes,
                work, selected_rows, preparation_bytes: limits.preparation_reserve_bytes,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let completion = {
                let mut processor = if reuse_prefill {
                    NativeSampledSentimentBatch::new_reusing_prefill(planner, &mut engine.value, admission, prefill)
                } else { match prefill {
                    None => NativeSampledSentimentBatch::new(planner, &mut engine.value, admission),
                    Some(prefill) => NativeSampledSentimentBatch::new_layer_major(
                        planner, &mut engine.value, admission, prefill),
                } }.map_err(HostedError::BatchSetup)?;
                stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)?
            };
            // No final success until the native allocation is physically gone.
            // Retain the charged IO and original control through publication.
            drop(engine);
            let result = completion.publish(&mut input.value.writer, control).map_err(HostedError::Batch);
            drop(input);
            drop(lease);
            result
        }),
        Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
            model, width, kernel, move |weights, control, team| {
                let mut input = input;
                checkpoint(control)?;
                let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(weights, limits.native.context_tokens,
                        Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                            max_rope_bytes: required.native.rope_bytes,
                            max_scratch_payload_bytes: required.native.scratch_payload_bound })
                        .map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let admission = Admission { lease: &lease, model: weights.identity(),
                    context: limits.native.context_tokens, kv_bytes: required.native.kv_bytes,
                    work, selected_rows, preparation_bytes: limits.preparation_reserve_bytes,
                    output_bytes: limits.transport.max_output_line_bytes as u64 };
                let completion = {
                    let mut processor = if reuse_prefill {
                        NativeSampledSentimentBatch::new_reusing_prefill(planner, &mut engine.value, admission, None)
                    } else { NativeSampledSentimentBatch::new(planner, &mut engine.value, admission) }
                    .map_err(HostedError::BatchSetup)?;
                    stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)?
                };
                drop(engine);
                drop(lease);
                Ok(DeferredSampledStream { completion, input })
            }, |pending, control| {
                let DeferredSampledStream { completion, mut input } = pending;
                let result = completion.publish(&mut input.value.writer, control).map_err(HostedError::Batch);
                drop(input);
                result
            }),
        }
    }
}

fn requirements(width: usize, work: SampledSentimentBatchLimits, task: TaskBudget,
    preparation_bytes: u64, limits: CorpusLimits) -> Result<Requirements, HostedError> {
    work.validate().map_err(HostedError::BatchSetup)?;
    task.validate().map_err(|_| HostedError::Limits("sampled sentiment corpus task budget"))?;
    let input_bytes = limits.reservation_bytes()?;
    if preparation_bytes == 0 || preparation_bytes > limits.preparation_reserve_bytes
        || limits.native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("sampled sentiment corpus preparation or allocator reserve"));
    }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if native.kv_bytes > task.max_kv_bytes {
        return Err(HostedError::Limits("sampled sentiment corpus complete resident KV"));
    }
    let sampler = json_sampling_workspace_bytes().map_err(SampledSentimentError::from).map_err(task_error)?;
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, sampler])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

/// Price the full declared prompt morsel once, even for a shorter document.
/// Native KV, result claims and transport/preparation owners are unchanged.
fn scheduled_requirements(width: usize, work: SampledSentimentBatchLimits, task: TaskBudget,
    preparation_bytes: u64, limits: CorpusLimits, prefill: Option<Int4PrefillLimits>)
    -> Result<Requirements, HostedError> {
    let mut required = requirements(width, work, task, preparation_bytes, limits)?;
    required.workspace_bytes = super::super::prefill::with_extra_workspace(required.workspace_bytes, prefill)?;
    Ok(required)
}

pub(super) struct Admission<'a> {
    lease: &'a EngineLease, model: &'a Int4ModelIdentity, context: usize, kv_bytes: u64,
    work: SampledSentimentBatchLimits, preparation_bytes: u64, output_bytes: u64,
    selected_rows: Option<Int4JsonSparseLimits>,
}
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
impl<'a> Admission<'a> {
    /// Durable callers retain the same sealed item admission and output guard
    /// through spool synchronization and the journal's commit acknowledgement.
    pub(super) fn for_job(lease: &'a EngineLease, model: &'a Int4ModelIdentity,
        context: usize, kv_bytes: u64, work: SampledSentimentBatchLimits,
        preparation_bytes: u64, output_bytes: u64, selected_rows: Option<Int4JsonSparseLimits>) -> Self {
        Self { lease, model, context, kv_bytes, work, preparation_bytes, output_bytes, selected_rows }
    }
}
impl SampledSentimentBatchAdmission for Admission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: SampledSentimentAdmission<'_>)
        -> Result<(ExecutionIdentity, Pending), BatchItemFailure> {
        constrained_int4::check_model(request.facts.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_request(self.model.group_width(), self.context, self.kv_bytes, self.work,
            self.preparation_bytes, self.output_bytes, self.selected_rows, &request)?;
        // Results retain explanation strings and counters, not native token
        // arrays. Keep the full result charge through serialization and flush.
        let guard = output_claim(self.lease, request.facts.max_result_bytes, 0).map_err(|error| match error {
            HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
            _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        })?;
        Ok((request.facts.identity.clone(), guard))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_request(width: usize, context: usize, kv_bytes: u64, work: SampledSentimentBatchLimits,
    preparation_bytes: u64, output_bytes: u64, selected_rows: Option<Int4JsonSparseLimits>,
    request: &SampledSentimentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let facts = &request.facts;
    constrained_int4::check_profile(facts.identity, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if facts.identity.task_spec != "sentiment-v1"
        || facts.identity.sampler_version != constrained_int4::JSON_SAMPLER_VERSION
        || facts.group_width != width || request.kv_reservation_bytes != kv_bytes
        || facts.selected_rows != selected_rows
        || !projection_matches(facts.model_work, facts.max_output_tokens, selected_rows)
        || facts.max_output_tokens > u64::from(facts.task_budget.max_output_tokens)
        || facts.task_budget.validate().is_err() || facts.preparation_bytes == 0
        || facts.preparation_bytes > preparation_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if required.kv_bytes != kv_bytes || facts.required_context == 0 || facts.required_context > context
        || required.kv_bytes > facts.task_budget.max_kv_bytes || facts.sample_count == 0
        || facts.model_work.forward_positions == 0 || facts.model_work.projected_logits == 0
        || facts.model_work.attention_pairs == 0 || facts.model_work.projections.dot_products == 0
        || facts.model_work.projections.multiply_accumulates == 0 || !fits(facts.model_work, work.max_item_work)
        || request.mask_node_visits == 0 || request.mask_node_visits > work.masks.max_visits_per_item {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    if facts.max_result_bytes == 0 || facts.max_result_bytes > facts.task_budget.max_output_bytes
        || facts.max_result_bytes > output_bytes { return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit)); }
    Ok(())
}

#[cfg(test)] mod tests;
