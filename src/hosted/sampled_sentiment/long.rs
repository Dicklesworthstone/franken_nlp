//! Complete INT8 sampled documents on one admitted native lifetime.
//! Every source chunk, independent axis and sample shares the same deadline,
//! cold-work allowance, process ledger, native allocation and optional team.
mod corpus;
pub use corpus::Int8SampledSentimentDocumentBatchHostConfig;
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod job;
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use job::{Int8SampledSentimentDocumentJobRequirements, int8_sampled_sentiment_document_job_requirements};
use super::*;
use crate::{
    native_engine::{decode::DecodeStepControl, constrained_int8::sparse::Int8JsonSparseLimits},
    tasks::sentiment::{SentimentAxis, SentimentRequest, sampled::{
        SampledSentimentLimits, SampledSentimentOptions,
        quantized::{Int8SampledSentimentPlanner, long::{
            PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError,
            Int8SampledSentimentMapLimits, Int8SampledSentimentMapRun,
            memory::{Int8SampledSentimentMapMemoryLimits, Int8SampledSentimentMapMemoryRequirement},
        }},
    }},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Per-chunk sampling and complete-document work, mask and output ceilings.
/// Prefill is a physical schedule; it changes neither independent sample
/// addresses nor any whole-document allowance. The seed stays private.
#[derive(Clone, Debug)]
pub struct Int8SampledSentimentMapHostConfig {
    pub native: NativeLimits,
    pub planning: SampledSentimentLimits,
    pub mapping: Int8SampledSentimentMapLimits,
    pub options: SampledSentimentOptions,
    pub preparation_reserve_bytes: u64,
    pub max_sampler_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
    pub prefill: Option<Int8PrefillLimits>,
    /// Exact complete legal-row ceiling; None retains full-head execution.
    /// This changes admitted projection work, not sampler or native storage.
    pub selected_rows: Option<Int8JsonSparseLimits>,
}
impl Int8SampledSentimentMapHostConfig {
    /// Pure memory quote, before source/model IO or tokenizer construction.
    /// Four axes prices any valid request before its private input is read.
    pub fn requirements(&self, task: TaskBudget, max_axes: usize)
        -> Result<Int8SampledSentimentMapMemoryRequirement, HostedError> {
        self.requirements_with_encoding(task, max_axes, &UntrustedEncoding::ByteFallback)
    }

    /// Input plans, retained chunk results, sampler and optional prompt
    /// morsels coexist. The team dispatcher separately charges its workers.
    pub fn requirements_with_encoding(&self, task: TaskBudget, max_axes: usize,
        encoding: &UntrustedEncoding) -> Result<Int8SampledSentimentMapMemoryRequirement, HostedError> {
        self.native.run.validate()?;
        if let Some(limits) = self.selected_rows {
            limits.validate().map_err(Int8SampledSentimentError::from)
                .map_err(HostedError::SampledSentiment)?;
        }
        if self.mask_limits.max_trie_node_visits == 0
            || self.mask_limits.checkpoint_interval_nodes == 0 || self.max_mask_node_visits == 0 {
            return Err(HostedError::Limits("INT8 sampled sentiment document mask allowance"));
        }
        let mut required = Int8SampledSentimentMapMemoryLimits {
            context_tokens: self.native.context_tokens,
            preparation_reserve_bytes: self.preparation_reserve_bytes,
            allocator_reserve_bytes: self.native.allocator_reserve_bytes,
            max_axes, max_sampler_bytes: self.max_sampler_bytes,
        }.requirements(task, self.planning, self.mapping, &self.options, encoding)
            .map_err(HostedError::SampledSentimentMap)?;
        required.workspace_bytes = super::super::scored::scoring_scratch(
            required.workspace_bytes, self.prefill)?;
        sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }
}

struct MapInput {
    request: SentimentRequest,
    config: Int8SampledSentimentMapHostConfig,
    encoding: UntrustedEncoding,
}

/// These owners remain in the dispatcher's admission, outside the native
/// callback, through physical worker drain on success, error and unwind.
struct MapPrepared {
    plan: PreparedInt8SampledSentimentMap,
    vocabulary: SampledSentimentVocabulary,
    _planner: Int8SampledSentimentPlanner,
    budget: Int8JsonBudget,
}
struct MapAdmission {
    prepared: Option<MapPrepared>,
    input: Charged<MapInput>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    output: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    _lease: EngineLease,
}
impl MapAdmission {
    fn prepare(&mut self, model: &ArtifactIdentity,
        required: Int8SampledSentimentMapMemoryRequirement, control: &mut RunControl)
        -> Result<(), HostedError> {
        checkpoint(control)?;
        let input = &self.input.value;
        let config = &input.config;
        let planner = Int8SampledSentimentPlanner::for_model_with_encoding(model,
            config.options.clone(), input.request.budget, config.planning, input.encoding.clone())
            .map_err(HostedError::SampledSentiment)?;
        let planner = match config.selected_rows {
            Some(limits) => planner.with_selected_rows(limits).map_err(HostedError::SampledSentiment)?,
            None => planner,
        };
        let vocabulary = SampledSentimentVocabulary::pinned()
            .map_err(Int8SampledSentimentError::from).map_err(HostedError::SampledSentiment)?;
        let plan = planner.plan_document_with_control(&input.request, config.mapping, control)
            .map_err(HostedError::SampledSentimentMap)?;
        validate_prepared(&plan, input, required, model)?;
        let budget = document_budget(&plan, config, required.native.kv_bytes);
        let mask = plan.complete_mask_charge(&vocabulary, budget, config.max_sampler_bytes)
            .map_err(HostedError::SampledSentimentMap)?;
        if mask == 0 || mask > config.max_mask_node_visits {
            return Err(HostedError::Limits("INT8 sampled sentiment complete-document mask charge"));
        }
        self.prepared = Some(MapPrepared { plan, vocabulary, _planner: planner, budget });
        checkpoint(control)
    }

    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("INT8 sampled document KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("INT8 sampled document scratch admission consumed"))?.commit()?);
        Ok(())
    }

    fn execute(&self, engine: &mut StrictInt8Engine<'_>, control: &mut RunControl)
        -> Result<Int8SampledSentimentMapRun, HostedError> {
        let prepared = self.prepared.as_ref().ok_or(HostedError::CompletionMissing)?;
        let config = &self.input.value.config;
        let expected = prepared.plan.preflight_metadata();
        let run = prepared.plan.execute_scheduled_with_control(engine, &prepared.vocabulary,
            prepared.budget, config.max_sampler_bytes, config.prefill, control)
            .map_err(HostedError::SampledSentimentMap)?;
        expected.verify_completed_with_control(&run, control).map_err(HostedError::SampledSentimentMap)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(HostedError::Limits("INT8 sampled document native completion"));
        }
        checkpoint(control)?;
        Ok(run)
    }

    fn finish(&mut self, run: Int8SampledSentimentMapRun)
        -> Result<HostedOutput<Int8SampledSentimentMapRun>, HostedError> {
        let output = self.output.take().ok_or(HostedError::Limits("INT8 sampled document output consumed"))?;
        match output.commit() {
            Ok(memory) => Ok(GuardedOutput::new(run, memory)),
            Err(error) => { drop(run); Err(error) }
        }
    }
}

impl NlpEngine {
    /// Preserve the complete source as independently sampled chunks, then
    /// reduce their ordinal-paired document replicates. Every sample starts
    /// from empty logical KV; no truncated source or partial bundle succeeds.
    pub fn map_int8_sampled_sentiment(&self, model: &ResidentInt8,
        request: SentimentRequest, config: Int8SampledSentimentMapHostConfig,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SampledSentimentMapRun>, HostedError> {
        self.map_int8_sampled_sentiment_with_encoding(model, request, config,
            UntrustedEncoding::ByteFallback, cancellation)
    }

    /// Compact encoding is explicit and receives one complete-document work
    /// allowance covering both source partition probes and actual chunk plans.
    pub fn map_int8_sampled_sentiment_with_encoding(&self, model: &ResidentInt8,
        request: SentimentRequest, config: Int8SampledSentimentMapHostConfig,
        encoding: UntrustedEncoding, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SampledSentimentMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        validate_request(&request, &config)?;
        let required = config.requirements_with_encoding(request.budget, request.axes.len(), &encoding)?;
        let native = config.native;
        let mut admission = admit(self, request, config, encoding, required)?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            admission.prepare(model.artifact_identity(), required, control)?;
            let mut engine = model.inner.loaded.value.engine(native.context_tokens, memory_budget(required.native))
                .map_err(HostedError::Model)?;
            admission.commit_native()?;
            let run = admission.execute(&mut engine, control)?;
            drop(engine);
            admission.finish(run)
        })
    }

    /// One fixed numerical team spans all chunks, axes and samples. The
    /// optional prompt morsels use these same workers and their admitted rows.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_sampled_sentiment_with_threads(&self, model: &ResidentInt8,
        request: SentimentRequest, config: Int8SampledSentimentMapHostConfig,
        kernel: Int8Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SampledSentimentMapRun>, HostedError> {
        self.map_int8_sampled_sentiment_with_encoding_and_threads(model, request, config,
            UntrustedEncoding::ByteFallback, kernel, compute_threads, cancellation)
    }

    /// Return the guarded complete document only after every worker physically
    /// joins. Input, compiler, vocabulary and all memory claims remain owned by
    /// the dispatcher through cancellation, allocation failure and panic.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_sampled_sentiment_with_encoding_and_threads(&self, model: &ResidentInt8,
        request: SentimentRequest, config: Int8SampledSentimentMapHostConfig,
        encoding: UntrustedEncoding, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8SampledSentimentMapRun>, HostedError> {
        dispatch::preflight_int8_team(self, config.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        validate_request(&request, &config)?;
        let required = config.requirements_with_encoding(request.budget, request.axes.len(), &encoding)?;
        let max_batch_rows = config.prefill.map_or(1, |limits| limits.max_batch_rows);
        let worker_bytes = dispatch::int8_prefill_team_workspace_bytes(compute_threads, max_batch_rows)?;
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, worker_bytes])?;
        let native = config.native;
        let admission = admit(self, request, config, encoding, required)?;
        dispatch::run_int8_prefill_team(self, native.run, cancellation, model.clone(),
            compute_threads, kernel, max_batch_rows, admission, move |model, admission, control, team| {
                admission.prepare(model.artifact_identity(), required, control)?;
                let mut engine = model.engine(native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.commit_native()?;
                let run = admission.execute(&mut engine, control)?;
                drop(engine);
                let compute = team.stats().into();
                Ok(HostedInt8TeamResult::from_guarded(admission.finish(run)?, compute))
            })
    }
}

fn admit(host: &NlpEngine, request: SentimentRequest, config: Int8SampledSentimentMapHostConfig,
    encoding: UntrustedEncoding, required: Int8SampledSentimentMapMemoryRequirement)
    -> Result<MapAdmission, HostedError> {
    let input_bytes = owned_input_bytes(&request, &config, required.input_bytes)?;
    sum(&[input_bytes, required.native.kv_bytes, required.workspace_bytes, required.output_reservation_bytes])?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
        || Ok(MapInput { request, config, encoding }))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
    let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes)?;
    Ok(MapAdmission { prepared: None, input, kv: Some(kv), scratch: Some(scratch), output: Some(output),
        _kv: None, _scratch: None, _lease: lease })
}

fn validate_request(request: &SentimentRequest, config: &Int8SampledSentimentMapHostConfig)
    -> Result<(), HostedError> {
    if request.document.is_empty() { return Err(HostedError::SampledSentimentMap(Int8SampledSentimentMapError::EmptySource)); }
    if request.document.len() > config.mapping.chunks.max_input_bytes
        || !(1..=SentimentAxis::ALL.len()).contains(&request.axes.len())
        || request.axes.iter().enumerate().any(|(index, axis)| request.axes[..index].contains(axis)) {
        return Err(HostedError::Limits("INT8 sampled sentiment complete-document input"));
    }
    Ok(())
}

fn owned_input_bytes(request: &SentimentRequest, config: &Int8SampledSentimentMapHostConfig, declared: u64)
    -> Result<u64, HostedError> {
    let source = request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64;
    let axes = (request.axes.capacity().saturating_sub(SentimentAxis::ALL.len()) as u64)
        .checked_mul(std::mem::size_of::<SentimentAxis>() as u64)
        .ok_or(HostedError::Limits("INT8 sampled document input capacity"))?;
    sum(&[declared, source, axes])
}

fn validate_prepared(prepared: &PreparedInt8SampledSentimentMap, input: &MapInput,
    required: Int8SampledSentimentMapMemoryRequirement, model: &ArtifactIdentity) -> Result<(), HostedError> {
    check_model_identity(model, prepared.execution_identity())?;
    let config = &input.config;
    let samples = prepared.chunk_count().checked_mul(input.request.axes.len())
        .and_then(|count| count.checked_mul(config.options.samples))
        .ok_or(HostedError::Limits("INT8 sampled document sample count"))?;
    if prepared.chunk_count() == 0 || prepared.chunk_count() > config.mapping.chunks.max_chunks
        || prepared.selected_rows() != config.selected_rows
        || prepared.sample_count() != samples || prepared.task_budget() != input.request.budget
        || prepared.required_context() == 0 || prepared.required_context() > config.native.context_tokens
        || prepared.max_result_bytes() != config.mapping.max_result_bytes
        || prepared.max_result_bytes() > required.max_output_bytes
        || prepared.max_output_tokens() > required.max_output_tokens
        || prepared.preparation_requirement_bytes() == 0
        || prepared.preparation_requirement_bytes() > required.input_bytes
        || !fits(prepared.planned_work(), config.mapping.max_model_work) {
        return Err(HostedError::Limits("INT8 sampled sentiment complete-document preparation"));
    }
    Ok(())
}

fn document_budget(prepared: &PreparedInt8SampledSentimentMap, config: &Int8SampledSentimentMapHostConfig,
    kv_bytes: u64) -> Int8JsonBudget {
    let work = prepared.planned_work();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, max_total_mask_node_visits: config.max_mask_node_visits,
        mask_limits: config.mask_limits,
    } }
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::SampledSentimentMap(Int8SampledSentimentMapError::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
