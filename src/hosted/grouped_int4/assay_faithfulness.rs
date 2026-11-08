//! One guarded native replay with all claims, heads and paired arms charged.
use super::*;
use crate::{assay::{self as metrics, AssayError, native_faithfulness::{self, Collector, Dataset,
        EvalOptions, Evaluation, Gates, NativeReplay, QualificationReport, ReplayError}},
    native_engine::{lmhead::scoring::ScoringLimits,
        strict_int4::{Int4Work, scoring::Int4ScoringBudget}},
    receipt::{CommitmentDomain, CommitmentKey},
    tasks::{ir::TaskBudget, judge::{JudgeLimits, grouped_int4::{Int4JudgeError, Int4JudgePlanner}}}};

#[derive(Clone, Copy, Debug)]
pub struct Int4FaithfulnessAssayHostLimits {
    pub native: NativeLimits,
    pub task: TaskBudget,
    /// At most 31 evidence windows plus the complete source. A single window
    /// reuses the full-source head; no source or uncertain window is dropped.
    pub max_heads: usize,
    pub preparation_reserve_bytes: u64,
    /// One nonrenewable five-axis allowance for all rows and both model arms.
    pub max_model_work: Int4Work,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4FaithfulnessAssayHostRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
}
impl Int4FaithfulnessAssayHostLimits {
    /// Pure quotation: both resident models are separate; the two sequential
    /// engines reuse one native envelope. These are modeled reserves, not RSS.
    pub fn requirements(self, group_widths: &[usize], limits: metrics::Limits)
        -> Result<Int4FaithfulnessAssayHostRequirements, HostedError> {
        limits.validate().map_err(HostedError::Assay)?; self.native.run.validate()?;
        self.task.validate().map_err(|_| HostedError::Limits("faithfulness Assay task budget"))?;
        if !(1..=2).contains(&group_widths.len()) || !(1..=32).contains(&self.max_heads)
            || self.preparation_reserve_bytes == 0 || self.native.allocator_reserve_bytes == 0
            || self.task.max_output_tokens < 2 || !nonzero(self.max_model_work) {
            return Err(HostedError::Limits("faithfulness Assay complete resource envelope"));
        }
        let mut native = Int4MemoryRequirement::for_context(self.native.context_tokens, group_widths[0]).map_err(native_error)?;
        for &width in &group_widths[1..] {
            let row = Int4MemoryRequirement::for_context(self.native.context_tokens, width).map_err(native_error)?;
            native.kv_bytes = native.kv_bytes.max(row.kv_bytes);
            native.rope_bytes = native.rope_bytes.max(row.rope_bytes);
            native.scratch_payload_bound = native.scratch_payload_bound.max(row.scratch_payload_bound);
        }
        if native.kv_bytes > self.task.max_kv_bytes {
            return Err(HostedError::Limits("faithfulness Assay complete reusable KV"));
        }
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound, self.native.allocator_reserve_bytes])?;
        let replay = native_faithfulness::required_replay_bytes(limits, group_widths.len()).map_err(HostedError::Assay)?;
        let tokens = self.native.context_tokens.checked_mul(self.max_heads)
            .ok_or(HostedError::Limits("faithfulness Assay complete head geometry"))?;
        let preparation = sum(&[multiply(tokens as u64, 256)?, multiply(self.max_heads as u64, 65_536)?])?;
        let input_bytes = sum(&[multiply(limits.max_input_bytes as u64, 8)?,
            multiply(limits.max_manifest_bytes as u64, 8)?, self.preparation_reserve_bytes,
            preparation, replay, multiply(self.task.max_output_bytes, 4)?])?;
        let output_reservation_bytes = sum(&[multiply(limits.max_output_bytes as u64, 4)?, 4096])?;
        Ok(Int4FaithfulnessAssayHostRequirements { native, workspace_bytes, input_bytes, output_reservation_bytes })
    }
}
struct Input { dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: Option<String> }
impl NlpEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_int4_faithfulness(&self, model: &ResidentInt4, dataset: Dataset,
        key: CommitmentKey, namespace: String, options: EvalOptions, limits: metrics::Limits,
        host: Int4FaithfulnessAssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Evaluation>, HostedError> {
        dispatch::preflight(self, host.native.run)?; check_domain(self, model)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, false, host.max_heads)?;
        let required = host.requirements(&[model.model_identity().group_width()], limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: None }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let model = model.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input; let native_claims = (kv, workspace);
            let result = allocate(output, || {
                let mut work = Int4Work::default();
                let replay = replay_arm(&model, &input.value.dataset, host, limits, required.native,
                    kernel, &mut work, control)?;
                let report = native_faithfulness::evaluate_with_checkpoint::<HostedError>(&replay, &input.value.key,
                    &input.value.namespace, options, limits, &mut || checkpoint(control))?;
                checkpoint(control)?; Ok(report)
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn qualify_int4_faithfulness(&self, candidate: &ResidentInt4, baseline: &ResidentInt4,
        dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: String, gates: Gates,
        options: EvalOptions, limits: metrics::Limits, host: Int4FaithfulnessAssayHostLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken) -> Result<HostedOutput<QualificationReport>, HostedError> {
        dispatch::preflight(self, host.native.run)?; check_domain(self, candidate)?; check_domain(self, baseline)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, true, host.max_heads)?;
        gates.validate(limits).map_err(HostedError::Assay)?;
        native_faithfulness::validate_scope(&namespace, Some(&host_scope)).map_err(HostedError::Assay)?;
        let required = host.requirements(&[candidate.model_identity().group_width(), baseline.model_identity().group_width()], limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: Some(host_scope) }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let candidate = candidate.clone(); let baseline = baseline.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input; let native_claims = (kv, workspace);
            let result = allocate(output, || {
                let mut work = Int4Work::default();
                let candidate = replay_arm(&candidate, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                let baseline = replay_arm(&baseline, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                let report = native_faithfulness::compare_with_checkpoint::<HostedError>(&candidate, &baseline,
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
fn replay_arm(model: &ResidentInt4, dataset: &Dataset, host: Int4FaithfulnessAssayHostLimits,
    limits: metrics::Limits, memory: Int4MemoryRequirement, kernel: PackedInt4Kernel,
    work: &mut Int4Work, control: &mut RunControl) -> Result<NativeReplay, HostedError> {
    checkpoint(control)?;
    let planner = Int4JudgePlanner::for_model(model.model_identity(), host.task, planning_limits(host)?)
        .map_err(judge_error)?;
    let mut collector = Collector::new(dataset, model.model_identity(), limits).map_err(HostedError::Assay)?;
    checkpoint(control)?;
    let mut engine = StrictInt4Engine::new(&model.inner.loaded.value, host.native.context_tokens,
        Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
            max_scratch_payload_bytes: memory.scratch_payload_bound }).map_err(native_error)?.with_kernel(kernel);
    for index in 0..dataset.replay_len() {
        checkpoint(control)?;
        let request = dataset.request(index, host.task).map_err(HostedError::Assay)?;
        let prepared = planner.plan_with_control(&request, control).map_err(judge_error)?;
        drop(request);
        if prepared.head_count() > host.max_heads || prepared.head_count() == 0
            || prepared.required_context() == 0 || prepared.required_context() > host.native.context_tokens
            || prepared.task_budget() != host.task { return Err(judge_error(Int4JudgeError::Accounting)); }
        let row_work = prepared.planned_work(); charge_work(work, row_work, host.max_model_work)?;
        let run = prepared.execute_with_control(prepared.execution_identity(), &mut engine,
            Int4ScoringBudget { native: Int4RunBudget::exact(row_work), max_kv_bytes: memory.kv_bytes }, control)
            .map_err(judge_error)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(judge_error(Int4JudgeError::Accounting));
        }
        collector.accept(index, &prepared, run, control).map_err(|error| match error {
            ReplayError::Assay(error) => HostedError::Assay(error),
            ReplayError::Cancelled(kind) => judge_error(Int4JudgeError::Cancelled(kind)),
        })?;
    }
    checkpoint(control)?; drop(engine); drop(planner);
    collector.finish().map_err(HostedError::Assay)
}
fn planning_limits(host: Int4FaithfulnessAssayHostLimits) -> Result<JudgeLimits, HostedError> {
    let prompts = host.native.context_tokens.checked_mul(host.max_heads)
        .ok_or(HostedError::Limits("faithfulness Assay aggregate prompts"))?;
    Ok(JudgeLimits { per_head: ScoringLimits { max_candidates: 3, max_total_tokens: 6,
            max_nodes: 7, max_depth: 2, max_candidate_id_bytes: 16,
            max_projected_logits: host.max_model_work.projected_logits },
        max_total_prompt_tokens: prompts, max_total_candidate_tokens: host.max_heads * 6,
        max_total_projected_logits: host.max_model_work.projected_logits, max_output_bytes: host.task.max_output_bytes })
}
#[allow(clippy::too_many_arguments)]
fn validate_inputs(dataset: &Dataset, key: &CommitmentKey, namespace: &str, options: EvalOptions,
    limits: metrics::Limits, paired: bool, max_heads: usize) -> Result<(), HostedError> {
    native_faithfulness::validate_scope(namespace, None).map_err(HostedError::Assay)?;
    dataset.validate_limits(limits).map_err(HostedError::Assay)?;
    dataset.validate_head_limit(max_heads).map_err(HostedError::Assay)?;
    if paired { options.validate_paired_rows(dataset.locked_test_sources(), limits) }
    else { options.validate_rows(dataset.locked_test_sources(), limits) }.map_err(HostedError::Assay)?;
    key.commit(CommitmentDomain::Configuration, namespace, "native_faithfulness_assay_preflight_v1", &[])
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
    if !nonzero(row) { return Err(judge_error(Int4JudgeError::Accounting)); }
    let next = total.checked_add(row).map_err(|_| judge_error(Int4JudgeError::WorkBudget))?;
    if next.forward_positions > ceiling.forward_positions || next.projected_logits > ceiling.projected_logits
        || next.attention_pairs > ceiling.attention_pairs || !next.projections.fits(ceiling.projections) {
        return Err(judge_error(Int4JudgeError::WorkBudget));
    }
    *total = next; Ok(())
}
fn multiply(value: u64, factor: u64) -> Result<u64, HostedError> {
    value.checked_mul(factor).ok_or(HostedError::Limits("faithfulness Assay memory arithmetic"))
}
fn judge_error(error: Int4JudgeError) -> HostedError { HostedError::Int4(Int4HostError::Judge(error)) }

#[cfg(test)] mod tests;
