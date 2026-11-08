//! One admission and physical ownership boundary for complete native tasks.
use super::*;

/// Values are produced by the existing task-specific admission functions.
/// Team scratch is separate and reserved by the sealed-team dispatcher.
pub(in crate::hosted::grouped_int4) struct TaskMemory {
    pub required: Int4MemoryRequirement,
    pub scratch_bytes: u64,
    pub output_bytes: u64,
    pub output_tokens: u64,
    pub preparation_bytes: Option<u64>,
}

// Keep the entire preparation owner intact through native finalization. In
// particular, early refusal drops its storage before its optional host charge.
struct AdmittedInput<P> { prepared: P, _memory: Option<CommittedMemory> }

/// The callback may borrow the native engine and request, but cannot return
/// either borrow. Every task uses its ordinary exact driver/finalizer; this
/// helper only installs the already-admitted fixed projection team.
#[allow(clippy::too_many_arguments)]
pub(in crate::hosted::grouped_int4) fn execute_task<P, T, F>(host: &NlpEngine,
    model: &ResidentInt4, prepared: P, native: NativeLimits, kernel: PackedInt4Kernel,
    compute_threads: usize, memory: TaskMemory, cancellation: CancellationToken, execute: F)
    -> Result<HostedInt4TeamResult<T>, HostedError>
where P: Send + 'static, T: Send + 'static,
    F: for<'weights> FnOnce(&P, &mut StrictInt4Engine<'weights>, &mut RunControl)
        -> Result<T, HostedError> + Send + 'static {
    dispatch::preflight_int4_team(host, native.run, compute_threads)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    // Detect the combined scratch overflow before any reservation. The team
    // quote remains owned by the dispatcher until all children have joined.
    sum(&[memory.scratch_bytes, int4_task_team_workspace_bytes(compute_threads,
        model.model_identity().group_width())?])?;
    let lease = host.resources().acquire_lease();
    let input_memory = memory.preparation_bytes.map(|bytes|
        Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?.commit()).transpose()?;
    let input = AdmittedInput { prepared, _memory: input_memory };
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, memory.required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, memory.scratch_bytes)?;
    let output = output_claim(&lease, memory.output_bytes, memory.output_tokens)?;
    dispatch::run_int4_team(host, native.run, cancellation, model.clone(), compute_threads,
        kernel, move |weights, control, team| {
            // Whole-aggregate capture preserves storage-before-charge drop
            // order even if a queued request is discarded before invocation.
            let input = input;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(weights, native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: memory.required.kv_bytes,
                    max_rope_bytes: memory.required.rope_bytes,
                    max_scratch_payload_bytes: memory.required.scratch_payload_bound,
                }).map_err(native_error)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = execute(&input.prepared, &mut engine.value, control)?;
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(input);
            let compute = team.stats().into();
            drop(lease);
            Ok(HostedInt4TeamResult { output: GuardedOutput::new(result.value, result._memory), compute })
        })
}

#[cfg(test)]
mod tests;
