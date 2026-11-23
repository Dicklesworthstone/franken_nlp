//! Independent sampled sentiment generations under one admitted native call.
//! The complete bundle shares one native allocation, sampler workspace, mask
//! allowance and cancellation scope. Explicit prefill reuse retains only the
//! current axis's causal prompt prefix between independently addressed samples.
//! Explicit parallel execution keeps one fixed projection team alive across
//! the entire bundle; sample addressing and per-row reductions stay unchanged.

mod corpus;
mod long;
pub use long::{SampledSentimentDocumentBatchHostConfig, SampledSentimentMapHostConfig};
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use long::{SampledSentimentDocumentJobRequirements, int4_sampled_sentiment_document_job_requirements};
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod job;
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use job::{SampledSentimentJobRequirements, int4_sampled_sentiment_job_requirements};

use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{
        constrained::JsonWorkBudget,
        constrained_int4::{Int4JsonBudget, json_sampling_workspace_bytes, sparse::Int4JsonSparseLimits},
        strict_int4::{Int4Work, prefill::Int4PrefillLimits},
    },
    tasks::{ir::TaskBudget, sentiment::sampled::{PreparedSampledSentiment,
        SampledSentimentError, SampledSentimentRun, SampledSentimentVocabulary}},
};

/// All work ceilings apply to the whole axis-by-sample bundle. Preparation
/// includes the sealed prompts and immutable vocabulary; callers also retain
/// their own preparation charge while constructing these values before transfer.
#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
    pub max_sampler_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

impl NlpEngine {
    /// Run independently addressed, schema-constrained score/explanation
    /// samples on a resident candidate. Only a complete validated bundle is
    /// returned. The output guard survives caller serialization and delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sampled_sentiment(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<SampledSentimentRun>, HostedError> {
        self.execute_int4_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, kernel, None, false, cancellation)
    }

    /// Admit the full declared causal prompt workspace once for the complete
    /// sample bundle. Each sample retains its independent empty-KV session,
    /// exact grammar and addressed draw; decoding remains sequential.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sampled_sentiment_layer_major(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<SampledSentimentRun>, HostedError> {
        self.execute_int4_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, kernel, Some(prefill), false, cancellation)
    }

    /// Reuse the current axis's private causal prompt prefix between samples.
    /// The first sample may use layer-major prefill; subsequent samples execute
    /// only the final prompt token and their own generated continuation. The
    /// complete cold-work ceiling remains reserved, while receipts report the
    /// actual work. Prefix state never survives an axis or request boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sampled_sentiment_reusing_prefill(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel,
        prefill: Option<Int4PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<SampledSentimentRun>, HostedError> {
        self.execute_int4_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, kernel, prefill, true, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int4_sampled_sentiment_scheduled(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel,
        prefill: Option<Int4PrefillLimits>, reuse_prefill: bool, cancellation: CancellationToken)
        -> Result<HostedOutput<SampledSentimentRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain);
        }
        crate::native_engine::constrained_int4::check_model(
            prepared.execution_identity(), model.model_identity())
            .map_err(SampledSentimentError::from).map_err(task_error)?;
        if prepared.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = requirements(limits, prepared.task_budget(),
            prepared.required_context(), prepared.group_width(), work,
            prepared.preparation_requirement_bytes())?;
        let scratch_bytes = super::prefill::with_extra_workspace(scratch_bytes, prefill)?;
        prepared.complete_mask_charge(&vocabulary, execution_budget(work, required.kv_bytes, limits),
            limits.max_sampler_bytes).map_err(task_error)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok((prepared, vocabulary)))?;
        let output = output_claim(&lease, input.value.0.max_result_bytes(),
            u64::from(input.value.0.task_budget().max_output_tokens))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
                        max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let (prepared, vocabulary) = &input.value;
                execute_prepared(prepared, vocabulary, &mut engine.value, limits,
                    required.kv_bytes, work, prefill, reuse_prefill, control)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Execute every independently addressed sample on one persistent native
    /// output-row team. The requested width must fit the installed resource
    /// host; it never silently narrows. Preparation, sampler and complete
    /// output stay charged through physical completion and guarded delivery.
    /// The returned compute receipt describes the physical schedule only.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sampled_sentiment_with_threads(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<SampledSentimentRun>, HostedError> {
        self.execute_int4_sampled_sentiment_threads_scheduled(model, prepared, vocabulary,
            limits, kernel, compute_threads, false, cancellation)
    }

    /// Keep one numerical row team and one axis-local causal prompt prefix
    /// across the independent sample schedule. Admission and complete-result
    /// cleanup are identical to the cold parallel path.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sampled_sentiment_reusing_prefill_with_threads(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<SampledSentimentRun>, HostedError> {
        self.execute_int4_sampled_sentiment_threads_scheduled(model, prepared, vocabulary,
            limits, kernel, compute_threads, true, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int4_sampled_sentiment_threads_scheduled(&self, model: &ResidentInt4,
        prepared: PreparedSampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: SampledSentimentHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        reuse_prefill: bool, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<SampledSentimentRun>, HostedError> {
        dispatch::preflight_int4_team(self, limits.native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain);
        }
        crate::native_engine::constrained_int4::check_model(
            prepared.execution_identity(), model.model_identity())
            .map_err(SampledSentimentError::from).map_err(task_error)?;
        if prepared.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = requirements(limits, prepared.task_budget(),
            prepared.required_context(), prepared.group_width(), work,
            prepared.preparation_requirement_bytes())?;
        prepared.complete_mask_charge(&vocabulary, execution_budget(work, required.kv_bytes, limits),
            limits.max_sampler_bytes).map_err(task_error)?;
        let memory = parallel::TaskMemory {
            required, scratch_bytes, output_bytes: prepared.max_result_bytes(),
            output_tokens: u64::from(prepared.task_budget().max_output_tokens),
            preparation_bytes: Some(limits.preparation_reserve_bytes),
        };
        parallel::execute_task(self, model, (prepared, vocabulary), limits.native,
            kernel, compute_threads, memory, None, cancellation, move |input, engine, control| {
                execute_prepared(&input.0, &input.1, engine, limits, required.kv_bytes, work, None,
                    reuse_prefill, control)
            })
    }
}

/// The schedule cannot change sample execution, complete-work validation or
/// the terminal checkpoint. All hosts use this one task/finalization boundary.
#[allow(clippy::too_many_arguments)]
fn execute_prepared(prepared: &PreparedSampledSentiment, vocabulary: &SampledSentimentVocabulary,
    engine: &mut StrictInt4Engine<'_>, limits: SampledSentimentHostLimits, kv_bytes: u64,
    work: Int4Work, prefill: Option<Int4PrefillLimits>, reuse_prefill: bool, control: &mut RunControl)
    -> Result<SampledSentimentRun, HostedError> {
    let budget = execution_budget(work, kv_bytes, limits);
    let run = if reuse_prefill {
        prepared.execute_reusing_prefill_with_control(engine, prepared.execution_identity(), vocabulary,
            budget, limits.max_sampler_bytes, prefill, control)
    } else {
        match prefill {
            None => prepared.execute_with_control(engine, prepared.execution_identity(), vocabulary,
                budget, limits.max_sampler_bytes, control),
            Some(prefill) => prepared.execute_layer_major_with_control(engine, prepared.execution_identity(),
                vocabulary, budget, limits.max_sampler_bytes, prefill, control),
        }
    }.map_err(task_error)?;
    let execution = prepared.execution_version(reuse_prefill);
    if run.execution != execution || !fits(run.model_work, work) || run.group_width != prepared.group_width()
        || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::Limits("sampled sentiment completion accounting"));
    }
    checkpoint(control)?;
    Ok(run)
}

// The sealed plan reserves every possible generation step at the declared
// projection width. Early EOS and smaller actual legal sets never lower this
// cold attempt debit, and selected execution does not shrink native buffers.
fn projection_matches(work: Int4Work, max_output_tokens: u64,
    selected: Option<Int4JsonSparseLimits>) -> bool {
    let rows = match selected {
        Some(limits) => match limits.validate() { Ok(rows) => rows, Err(_) => return false },
        None => crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    };
    max_output_tokens != 0
        && max_output_tokens.checked_mul(rows as u64) == Some(work.projected_logits)
}

fn execution_budget(work: Int4Work, kv_bytes: u64, limits: SampledSentimentHostLimits)
    -> Int4JsonBudget {
    Int4JsonBudget {
        native: Int4RunBudget::exact(work),
        json: JsonWorkBudget { max_forward_positions: work.forward_positions,
            max_projected_logits: work.projected_logits, max_kv_bytes: kv_bytes,
            max_total_mask_node_visits: limits.max_mask_node_visits,
            mask_limits: limits.mask_limits },
    }
}

fn task_error(error: SampledSentimentError) -> HostedError {
    HostedError::Int4(Int4HostError::SampledSentiment(error))
}

fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

fn requirements(limits: SampledSentimentHostLimits, task: TaskBudget, context: usize,
    width: usize, work: Int4Work, preparation_bytes: u64)
    -> Result<(Int4MemoryRequirement, u64), HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("sampled sentiment task budget"))?;
    let sampler_bytes = json_sampling_workspace_bytes()
        .map_err(SampledSentimentError::from).map_err(task_error)?;
    if preparation_bytes == 0 || preparation_bytes > limits.preparation_reserve_bytes
        || limits.native.allocator_reserve_bytes == 0
        || context == 0 || context > limits.native.context_tokens
        || work.forward_positions == 0 || work.projected_logits == 0
        || !fits(work, limits.max_model_work) || sampler_bytes > limits.max_sampler_bytes
        || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 || limits.max_mask_node_visits == 0 {
        return Err(HostedError::Limits("sampled sentiment complete work, sampler or preparation"));
    }
    let required = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
        .map_err(native_error)?;
    if required.kv_bytes > task.max_kv_bytes {
        return Err(HostedError::Limits("sampled sentiment complete resident KV"));
    }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, sampler_bytes])?;
    Ok((required, scratch))
}

#[cfg(test)]
mod tests;
