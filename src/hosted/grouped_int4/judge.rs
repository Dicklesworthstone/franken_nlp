//! Complete native INT4 judgments under one charged physical invocation.
use super::*;
use crate::{
    native_engine::strict_int4::{Int4Work, STRICT_INT4_PROFILE, prefill::Int4PrefillLimits, scoring::Int4ScoringBudget},
    tasks::{ir::TaskBudget, judge::grouped_int4::{self as judge_task, Int4JudgeError,
        Int4JudgeRun, PreparedInt4Judge, INT4_JUDGE_EXECUTION}},
};

/// Nonrenewable work for both comparison orders, every criterion or the
/// complete full-source/evidence bundle. Callers admit compiled-plan storage
/// before transfer; this host retains its own preparation charge through the
/// physical run. These modeled reservations are not measurements of RSS.
#[derive(Clone, Copy, Debug)]
pub struct Int4JudgeHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
}
impl NlpEngine {
    /// Execute one sealed judgment against the actual resident model and
    /// group partition. Every head is admitted before the first native run;
    /// one failed order, criterion or evidence head publishes no partial result.
    pub fn execute_int4_judge(&self, model: &ResidentInt4, prepared: PreparedInt4Judge,
        limits: Int4JudgeHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4JudgeRun>, HostedError> {
        self.execute_int4_judge_scheduled(model, prepared, limits, kernel, None, cancellation)
    }
    /// Group each order, criterion and evidence prompt under one additional
    /// admitted workspace. No head receives a renewed work allowance and no
    /// partial judgment can escape the existing guarded completion boundary.
    pub fn execute_int4_judge_layer_major(&self, model: &ResidentInt4, prepared: PreparedInt4Judge,
        limits: Int4JudgeHostLimits, kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4JudgeRun>, HostedError> {
        self.execute_int4_judge_scheduled(model, prepared, limits, kernel, Some(prefill), cancellation)
    }
    fn execute_int4_judge_scheduled(&self, model: &ResidentInt4, prepared: PreparedInt4Judge,
        limits: Int4JudgeHostLimits, kernel: PackedInt4Kernel, prefill: Option<Int4PrefillLimits>,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4JudgeRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        judge_task::check_model(prepared.execution_identity(), model.model_identity()).map_err(judge_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = requirements(limits, prepared.task_budget(),
            prepared.required_context(), prepared.group_width(), work)?;
        let scratch_bytes = super::prefill::with_extra_workspace(scratch_bytes, prefill)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?,
            || Ok(prepared))?;
        let output = output_claim(&lease, input.value.max_result_bytes(),
            u64::from(input.value.task_budget().max_output_tokens))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let budget = Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes };
                let result = match prefill {
                    None => input.value.execute_with_control(input.value.execution_identity(), &mut engine.value, budget, control),
                    Some(limits) => input.value.execute_layer_major_with_control(input.value.execution_identity(),
                        &mut engine.value, budget, limits, control),
                }.map_err(judge_error)?;
                if result.schema_version != 1 || result.execution != INT4_JUDGE_EXECUTION
                    || result.numerics_profile != STRICT_INT4_PROFILE || result.model_work != work
                    || result.head_count != input.value.head_count() || result.group_width != input.value.group_width()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(judge_error(Int4JudgeError::Accounting));
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn judge_error(error: Int4JudgeError) -> HostedError { HostedError::Int4(Int4HostError::Judge(error)) }
fn requirements(limits: Int4JudgeHostLimits, task: TaskBudget, context: usize,
    width: usize, work: Int4Work) -> Result<(Int4MemoryRequirement, u64), HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("INT4 judge task budget"))?;
    if limits.preparation_reserve_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || context == 0 || context > limits.native.context_tokens {
        return Err(HostedError::Limits("INT4 judge context or preparation admission"));
    }
    let required = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if required.kv_bytes > task.max_kv_bytes { return Err(HostedError::Limits("INT4 judge complete resident KV capacity")); }
    let cap = limits.max_model_work;
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(HostedError::Limits("INT4 judge complete model work"));
    }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?;
    Ok((required, scratch))
}

#[cfg(test)]
mod tests;
