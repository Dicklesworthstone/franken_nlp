//! One charged native replay for all documents, affect axes and paired arms.
use super::*;
use crate::{assay::{self as metrics, AssayError, native_sentiment::{self, Collector, Dataset,
        EvalOptions, Evaluation, Gates, NativeReplay, QualificationReport, ReplayError}},
    native_engine::{lmhead::scoring::{ScoringLimits, ScoringMode},
        strict_int4::{Int4Work, scoring::Int4ScoringBudget}},
    receipt::{CommitmentDomain, CommitmentKey},
    tasks::{ir::TaskBudget, sentiment::{SentimentLimits, SentimentOptions,
        grouped_int4::{Int4SentimentError, Int4SentimentPlanner}}}};

#[derive(Clone, Copy, Debug)]
pub struct Int4SentimentAssayHostLimits {
    pub native: NativeLimits,
    pub task: TaskBudget,
    pub max_axes: usize,
    pub preparation_reserve_bytes: u64,
    /// One nonrenewable five-axis allowance across all rows AND paired arms.
    pub max_model_work: Int4Work,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SentimentAssayHostRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
}
impl Int4SentimentAssayHostLimits {
    /// Both resident models are separate. Sequential arms share one native
    /// peak; all predictions needed for paired statistics remain admitted.
    /// This is a payload/allocator reserve quotation, not observed process RSS.
    pub fn requirements(self, group_widths: &[usize], limits: metrics::Limits)
        -> Result<Int4SentimentAssayHostRequirements, HostedError> {
        limits.validate().map_err(HostedError::Assay)?; self.native.run.validate()?;
        self.task.validate().map_err(|_| HostedError::Limits("sentiment Assay task budget"))?;
        if !(1..=2).contains(&group_widths.len()) || !(1..=native_sentiment::MAX_AXES).contains(&self.max_axes)
            || self.preparation_reserve_bytes == 0 || self.native.allocator_reserve_bytes == 0
            || self.task.max_output_tokens < 5 || self.task.max_grammar_states < 17
            || (self.task.max_input_tokens as usize).checked_add(4)
                .is_none_or(|tokens| tokens > self.native.context_tokens)
            || !nonzero(self.max_model_work) {
            return Err(HostedError::Limits("sentiment Assay complete resource envelope"));
        }
        let mut native = Int4MemoryRequirement::for_context(self.native.context_tokens, group_widths[0]).map_err(native_error)?;
        for &width in &group_widths[1..] {
            let row = Int4MemoryRequirement::for_context(self.native.context_tokens, width).map_err(native_error)?;
            native.kv_bytes = native.kv_bytes.max(row.kv_bytes);
            native.rope_bytes = native.rope_bytes.max(row.rope_bytes);
            native.scratch_payload_bound = native.scratch_payload_bound.max(row.scratch_payload_bound);
        }
        if native.kv_bytes > self.task.max_kv_bytes { return Err(HostedError::Limits("sentiment Assay complete reusable KV")); }
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound, self.native.allocator_reserve_bytes])?;
        let replay = native_sentiment::required_replay_bytes(limits, self.max_axes, group_widths.len()).map_err(HostedError::Assay)?;
        let tokens = self.native.context_tokens.checked_mul(self.max_axes)
            .ok_or(HostedError::Limits("sentiment Assay complete head geometry"))?;
        let preparation = sum(&[multiply(tokens as u64, 256)?, multiply(self.max_axes as u64, 65_536)?])?;
        let input_bytes = sum(&[multiply(limits.max_input_bytes as u64, 8)?, multiply(limits.max_manifest_bytes as u64, 8)?,
            self.preparation_reserve_bytes, preparation, replay, multiply(self.task.max_output_bytes, 4)?])?;
        let output_reservation_bytes = sum(&[multiply(limits.max_output_bytes as u64, 4)?, 4096])?;
        Ok(Int4SentimentAssayHostRequirements { native, workspace_bytes, input_bytes, output_reservation_bytes })
    }
}
struct Input { dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: Option<String> }
impl NlpEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_int4_sentiment(&self, model: &ResidentInt4, dataset: Dataset,
        key: CommitmentKey, namespace: String, options: EvalOptions, limits: metrics::Limits,
        host: Int4SentimentAssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Evaluation>, HostedError> {
        dispatch::preflight(self, host.native.run)?; check_domain(self, model)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, false, host.max_axes)?;
        let required = host.requirements(&[model.model_identity().group_width()], limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: None }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let model = model.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input;
            // Both component charges outlive every sequential native engine.
            let native_claims = (kv.commit()?, workspace.commit()?);
            let result = allocate(output, || {
                let mut work = Int4Work::default();
                let replay = replay_arm(&model, &input.value.dataset, host, limits, required.native,
                    kernel, &mut work, control)?;
                let report = native_sentiment::evaluate_with_checkpoint::<HostedError>(&replay, &input.value.key,
                    &input.value.namespace, options, limits, &mut || checkpoint(control))?;
                checkpoint(control)?; Ok(report)
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn qualify_int4_sentiment(&self, candidate: &ResidentInt4, baseline: &ResidentInt4,
        dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: String, gates: Gates,
        options: EvalOptions, limits: metrics::Limits, host: Int4SentimentAssayHostLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken) -> Result<HostedOutput<QualificationReport>, HostedError> {
        dispatch::preflight(self, host.native.run)?; check_domain(self, candidate)?; check_domain(self, baseline)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, true, host.max_axes)?;
        gates.validate(limits).map_err(HostedError::Assay)?;
        native_sentiment::validate_scope(&namespace, Some(&host_scope)).map_err(HostedError::Assay)?;
        let required = host.requirements(&[candidate.model_identity().group_width(), baseline.model_identity().group_width()], limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: Some(host_scope) }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let candidate = candidate.clone(); let baseline = baseline.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input; let native_claims = (kv.commit()?, workspace.commit()?);
            let result = allocate(output, || {
                let mut work = Int4Work::default();
                let candidate = replay_arm(&candidate, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                let baseline = replay_arm(&baseline, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                let report = native_sentiment::compare_with_checkpoint::<HostedError>(&candidate, &baseline,
                    &input.value.key, &input.value.namespace,
                    input.value.host_scope.as_deref().ok_or(HostedError::CompletionMissing)?,
                    gates, options, limits, &mut || checkpoint(control))?;
                checkpoint(control)?; Ok(report)
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
#[allow(clippy::too_many_arguments)]
fn replay_arm(model: &ResidentInt4, dataset: &Dataset, host: Int4SentimentAssayHostLimits,
    limits: metrics::Limits, memory: Int4MemoryRequirement, kernel: PackedInt4Kernel,
    work: &mut Int4Work, control: &mut RunControl) -> Result<NativeReplay, HostedError> {
    checkpoint(control)?;
    let registry = crate::tokenizer::pinned_controls::pinned().map_err(|_| HostedError::Assay(AssayError::Identity))?;
    let mut ends = registry.template_controls().entries().iter()
        .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
    let eos_token_id = ends.next().ok_or(HostedError::Assay(AssayError::Identity))?.id;
    if ends.next().is_some() { return Err(HostedError::Assay(AssayError::Identity)); }
    let options = SentimentOptions { mode: ScoringMode::FullVocabulary, eos_token_id, policy: dataset.manifest().policy };
    let planner = Int4SentimentPlanner::for_model(model.model_identity(), options, host.task, planning_limits(host)?)
        .map_err(sentiment_error)?;
    let mut collector = Collector::new(dataset, model.model_identity(), limits).map_err(HostedError::Assay)?;
    checkpoint(control)?;
    let mut engine = StrictInt4Engine::new(&model.inner.loaded.value, host.native.context_tokens,
        Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
            max_scratch_payload_bytes: memory.scratch_payload_bound }).map_err(native_error)?.with_kernel(kernel);
    for index in 0..dataset.replay_len() {
        checkpoint(control)?;
        let request = dataset.request(index, host.task).map_err(HostedError::Assay)?;
        let prepared = planner.plan_with_control(&request, control).map_err(sentiment_error)?;
        drop(request);
        if prepared.head_count() != dataset.manifest().axes.len() || prepared.head_count() > host.max_axes
            || prepared.head_count() == 0 || prepared.required_context() == 0
            || prepared.required_context() > host.native.context_tokens || prepared.task_budget() != host.task {
            return Err(sentiment_error(Int4SentimentError::Accounting));
        }
        let row_work = prepared.planned_work(); charge_work(work, row_work, host.max_model_work)?;
        let run = prepared.execute_with_control(prepared.execution_identity(), &mut engine,
            Int4ScoringBudget { native: Int4RunBudget::exact(row_work), max_kv_bytes: memory.kv_bytes }, control)
            .map_err(sentiment_error)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(sentiment_error(Int4SentimentError::Accounting));
        }
        collector.accept(index, &prepared, run, control).map_err(|error| match error {
            ReplayError::Assay(error) => HostedError::Assay(error),
            ReplayError::Cancelled(kind) => sentiment_error(Int4SentimentError::Cancelled(kind)),
        })?;
    }
    checkpoint(control)?; drop(engine); drop(planner);
    collector.finish().map_err(HostedError::Assay)
}
fn planning_limits(host: Int4SentimentAssayHostLimits) -> Result<SentimentLimits, HostedError> {
    let prompts = host.native.context_tokens.checked_mul(host.max_axes)
        .ok_or(HostedError::Limits("sentiment Assay aggregate prompts"))?;
    Ok(SentimentLimits { per_axis: ScoringLimits { max_candidates: 5, max_total_tokens: 16,
            max_nodes: 17, max_depth: 5, max_candidate_id_bytes: 16,
            max_projected_logits: host.max_model_work.projected_logits },
        max_total_prompt_tokens: prompts, max_total_candidate_tokens: host.max_axes * 16,
        // Admission uses the scorer's conservative sum(depths)+root bound,
        // even though sharing numerical prefixes creates only 15 trie nodes.
        max_total_nodes: host.max_axes * 17, max_total_projected_logits: host.max_model_work.projected_logits,
        max_output_bytes: host.task.max_output_bytes })
}
#[allow(clippy::too_many_arguments)]
fn validate_inputs(dataset: &Dataset, key: &CommitmentKey, namespace: &str, options: EvalOptions,
    limits: metrics::Limits, paired: bool, max_axes: usize) -> Result<(), HostedError> {
    native_sentiment::validate_scope(namespace, None).map_err(HostedError::Assay)?;
    dataset.validate_limits(limits).map_err(HostedError::Assay)?;
    dataset.validate_axis_limit(max_axes).map_err(HostedError::Assay)?;
    native_sentiment::validate_replay(options, dataset.replay_len(), dataset.manifest().axes.len(), paired, limits)
        .map_err(HostedError::Assay)?;
    key.commit(CommitmentDomain::Configuration, namespace, "native_sentiment_assay_preflight_v1", &[])
        .map_err(|_| HostedError::Assay(AssayError::Commitment))?;
    Ok(())
}
fn check_domain(host: &NlpEngine, model: &ResidentInt4) -> Result<(), HostedError> {
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); } Ok(())
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn charge_work(total: &mut Int4Work, row: Int4Work, ceiling: Int4Work) -> Result<(), HostedError> {
    if !nonzero(row) { return Err(sentiment_error(Int4SentimentError::Accounting)); }
    let next = total.checked_add(row).map_err(|_| sentiment_error(Int4SentimentError::WorkBudget))?;
    if next.forward_positions > ceiling.forward_positions || next.projected_logits > ceiling.projected_logits
        || next.attention_pairs > ceiling.attention_pairs || !next.projections.fits(ceiling.projections) {
        return Err(sentiment_error(Int4SentimentError::WorkBudget));
    }
    *total = next; Ok(())
}
fn sentiment_error(error: Int4SentimentError) -> HostedError { HostedError::Int4(Int4HostError::Sentiment(error)) }
fn multiply(a: u64, b: u64) -> Result<u64, HostedError> {
    a.checked_mul(b).ok_or(HostedError::Limits("sentiment Assay memory arithmetic"))
}

#[cfg(test)] mod tests;
