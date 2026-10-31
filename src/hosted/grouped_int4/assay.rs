//! Frozen labeled text -> actual native classification -> guarded Assay reports.
//! One physical invocation owns both paired arms, their nonrenewable work,
//! private source/score storage and finalization. No supplied-score injection.

use super::*;
use crate::{
    assay::{self as metrics, native_classify::{self, Collector, Dataset, Evaluation,
        Gates, NativeReplay, QualificationReport}, AssayError, EvalOptions},
    batch::BatchWork,
    native_engine::{lmhead::scoring::ScoringLimits,
        strict_int4::{Int4Work, scoring::Int4ScoringBudget}},
    receipt::{CommitmentDomain, CommitmentKey},
    tasks::{classify::{ClassificationLimits, grouped_int4::Int4ClassificationPlanner}, ir::TaskBudget},
};

/// Complete operation limits. Both qualification arms spend the same native
/// work and run allowance. Each row retains this exact original task budget.
#[derive(Clone, Copy, Debug)]
pub struct Int4AssayHostLimits {
    pub native: NativeLimits,
    pub task: TaskBudget,
    /// Caller-priced tokenizer, compiler, request and allocator overhead.
    /// Dataset, retained scores, row results and native storage are separate.
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
}

/// Complete transient peak, excluding resident weights, outer caller input
/// staging and the separately retained RunLimits cleanup allowance. Native
/// engines run sequentially, so this is their maximum payload, not their sum.
/// These are modeled storage reservations, never a process-RSS guarantee.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4AssayHostRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
}
impl Int4AssayHostLimits {
    /// Pure pre-IO quotation. Exactly one width denotes evaluation; exactly
    /// two denote explicit candidate/baseline qualification. No model is loaded.
    pub fn requirements(self, group_widths: &[usize], limits: metrics::Limits)
        -> Result<Int4AssayHostRequirements, HostedError> {
        limits.validate().map_err(HostedError::Assay)?;
        self.native.run.validate()?;
        self.task.validate().map_err(|_| HostedError::Limits("INT4 Assay task budget"))?;
        if !(1..=2).contains(&group_widths.len()) || self.preparation_reserve_bytes == 0
            || self.native.allocator_reserve_bytes == 0 || !nonzero(self.max_model_work) {
            return Err(HostedError::Limits("INT4 Assay complete resource envelope"));
        }
        let mut native = Int4MemoryRequirement::for_context(self.native.context_tokens, group_widths[0])
            .map_err(native_error)?;
        for &width in &group_widths[1..] {
            let row = Int4MemoryRequirement::for_context(self.native.context_tokens, width).map_err(native_error)?;
            native.kv_bytes = native.kv_bytes.max(row.kv_bytes);
            native.rope_bytes = native.rope_bytes.max(row.rope_bytes);
            native.scratch_payload_bound = native.scratch_payload_bound.max(row.scratch_payload_bound);
        }
        if native.kv_bytes > self.task.max_kv_bytes {
            return Err(HostedError::Limits("INT4 Assay complete reusable KV allocation"));
        }
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let replay = native_classify::required_replay_bytes(limits, group_widths.len()).map_err(HostedError::Assay)?;
        let input_bytes = sum(&[multiply(limits.max_input_bytes as u64, 8)?,
            multiply(limits.max_manifest_bytes as u64, 8)?, self.preparation_reserve_bytes,
            replay, multiply(self.task.max_output_bytes, 4)?])?;
        let output_reservation_bytes = sum(&[multiply(limits.max_output_bytes as u64, 4)?, 4096])?;
        Ok(Int4AssayHostRequirements { native, workspace_bytes, input_bytes, output_reservation_bytes })
    }
}

struct Input { dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: Option<String> }

impl NlpEngine {
    /// Evaluate every frozen document using one resident INT4 model. Native
    /// observations and identities are collected only from actual task runs;
    /// no private document or raw score collection escapes in the report.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_int4_classification(&self, model: &ResidentInt4, dataset: Dataset,
        key: CommitmentKey, namespace: String, options: EvalOptions, limits: metrics::Limits,
        host: Int4AssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Evaluation>, HostedError> {
        dispatch::preflight(self, host.native.run)?;
        check_domain(self, model)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, 1)?;
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
            // Live reservations cover the actual engine until it is dropped;
            // empty or failed work never fabricates committed model payload.
            let native_claims = (kv, workspace);
            let result = allocate(output, || {
                let mut work = Int4Work::default();
                let replay = replay_arm(&model, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                checkpoint(control)?;
                let report = native_classify::evaluate_with_checkpoint::<HostedError>(&replay, &input.value.key,
                    &input.value.namespace, options, limits, &mut || checkpoint(control))?;
                checkpoint(control)?;
                Ok(report)
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Paired native replay of the exact same frozen labeled population. Both
    /// resident models are explicit candidates; neither is inferred to be the
    /// active model. One deadline and five-axis work ledger cover both arms.
    #[allow(clippy::too_many_arguments)]
    pub fn qualify_int4_classification(&self, candidate: &ResidentInt4, baseline: &ResidentInt4,
        dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: String,
        gates: Gates, options: EvalOptions, limits: metrics::Limits, host: Int4AssayHostLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<QualificationReport>, HostedError> {
        dispatch::preflight(self, host.native.run)?;
        check_domain(self, candidate)?; check_domain(self, baseline)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, 2)?;
        gates.validate(limits).map_err(HostedError::Assay)?;
        native_classify::validate_scope(&namespace, Some(&host_scope)).map_err(HostedError::Assay)?;
        let required = host.requirements(&[candidate.model_identity().group_width(),
            baseline.model_identity().group_width()], limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: Some(host_scope) }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let candidate = candidate.clone(); let baseline = baseline.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input;
            let native_claims = (kv, workspace);
            let result = allocate(output, || {
                let mut work = Int4Work::default();
                let candidate = replay_arm(&candidate, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                // The first engine is gone. Its scores remain private and
                // charged while the second engine reuses the native envelope.
                let baseline = replay_arm(&baseline, &input.value.dataset, host, limits,
                    required.native, kernel, &mut work, control)?;
                checkpoint(control)?;
                let report = native_classify::compare_with_checkpoint::<HostedError>(&candidate, &baseline, &input.value.key,
                    &input.value.namespace, input.value.host_scope.as_deref().ok_or(HostedError::CompletionMissing)?,
                    gates, options, limits, &mut || checkpoint(control))?;
                checkpoint(control)?;
                Ok(report)
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_arm(model: &ResidentInt4, dataset: &Dataset, host: Int4AssayHostLimits,
    limits: metrics::Limits, memory: Int4MemoryRequirement, kernel: PackedInt4Kernel,
    work: &mut Int4Work, control: &mut RunControl) -> Result<NativeReplay, HostedError> {
    checkpoint(control)?;
    let planner = Int4ClassificationPlanner::for_model(model.model_identity(), host.task,
        planning_limits(host, limits)).map_err(classification_error)?;
    let mut collector = Collector::new(dataset, model.model_identity(), limits).map_err(HostedError::Assay)?;
    checkpoint(control)?;
    let mut engine = StrictInt4Engine::new(&model.inner.loaded.value, host.native.context_tokens,
        Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
            max_scratch_payload_bytes: memory.scratch_payload_bound }).map_err(native_error)?.with_kernel(kernel);
    for index in 0..dataset.len() {
        checkpoint(control)?;
        let request = dataset.classification_request(index, host.task).map_err(HostedError::Assay)?;
        let prepared = planner.plan_with_control(&request, control).map_err(classification_error)?;
        drop(request);
        if prepared.head_count() != 1 || prepared.required_context() == 0
            || prepared.required_context() > host.native.context_tokens || prepared.task_budget() != host.task {
            return Err(classification_error(Int4ClassificationError::Accounting));
        }
        let row_work = prepared.planned_work();
        charge_work(work, row_work, host.max_model_work)?;
        let run = prepared.execute_with_control(prepared.execution_identity(), &mut engine,
            Int4ScoringBudget { native: Int4RunBudget::exact(row_work), max_kv_bytes: memory.kv_bytes }, control)
            .map_err(classification_error)?;
        if run.model_work != row_work || run.head_count != 1 || run.group_width != planner.group_width()
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(classification_error(Int4ClassificationError::Accounting));
        }
        collector.accept(index, &prepared, run).map_err(HostedError::Assay)?;
    }
    checkpoint(control)?;
    drop(engine); drop(planner);
    collector.finish().map_err(HostedError::Assay)
}

fn planning_limits(host: Int4AssayHostLimits, limits: metrics::Limits) -> ClassificationLimits {
    ClassificationLimits { max_labels: 2, max_input_bytes: limits.max_line_bytes,
        max_context_tokens: host.native.context_tokens, max_total_prompt_tokens: host.native.context_tokens,
        max_work: BatchWork { forward_positions: host.max_model_work.forward_positions,
            projected_logits: host.max_model_work.projected_logits },
        scoring: ScoringLimits { max_candidates: 2,
            max_projected_logits: host.max_model_work.projected_logits, ..ScoringLimits::default() },
        ..ClassificationLimits::default() }
}
fn validate_inputs(dataset: &Dataset, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: metrics::Limits, arms: usize) -> Result<(), HostedError> {
    native_classify::validate_scope(namespace, None).map_err(HostedError::Assay)?;
    dataset.validate_limits(limits).map_err(HostedError::Assay)?;
    options.validate_rows(dataset.locked_test_len(), limits).map_err(HostedError::Assay)?;
    validate_bootstrap(dataset.locked_test_len(), arms, options, limits)?;
    // Validate the existing receipt namespace grammar before any model work.
    // This disposable commitment is not reported as execution evidence.
    key.commit(CommitmentDomain::Configuration, namespace, "native_assay_preflight_v1", &[])
        .map_err(|_| HostedError::Assay(AssayError::Commitment))?;
    Ok(())
}
fn validate_bootstrap(rows: usize, arms: usize, options: EvalOptions, limits: metrics::Limits)
    -> Result<(), HostedError> {
    if !(1..=2).contains(&arms) || (rows as u64).checked_mul(options.bootstrap.resamples as u64)
        .and_then(|draws| draws.checked_mul(arms as u64)).is_none_or(|draws| draws > limits.max_bootstrap_draws) {
        return Err(HostedError::Assay(AssayError::InputLimit));
    }
    Ok(())
}
fn check_domain(host: &NlpEngine, model: &ResidentInt4) -> Result<(), HostedError> {
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    Ok(())
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn charge_work(total: &mut Int4Work, row: Int4Work, ceiling: Int4Work) -> Result<(), HostedError> {
    if !nonzero(row) { return Err(classification_error(Int4ClassificationError::Accounting)); }
    let next = total.checked_add(row).map_err(|_| classification_error(Int4ClassificationError::WorkBudget))?;
    if next.forward_positions > ceiling.forward_positions || next.projected_logits > ceiling.projected_logits
        || next.attention_pairs > ceiling.attention_pairs || !next.projections.fits(ceiling.projections) {
        return Err(classification_error(Int4ClassificationError::WorkBudget));
    }
    *total = next; Ok(())
}
fn multiply(value: u64, factor: u64) -> Result<u64, HostedError> {
    value.checked_mul(factor).ok_or(HostedError::Limits("INT4 Assay memory arithmetic"))
}
fn classification_error(error: Int4ClassificationError) -> HostedError {
    HostedError::Int4(Int4HostError::Classification(error))
}

#[cfg(test)] mod tests;
