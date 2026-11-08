//! Original command deadline and charged ownership through physical publication.
use super::*;
use std::time::{Duration, Instant};
use super::super::native::{ALLOCATOR_BYTES, CLEANUP_BYTES, sum,
    runtime::{Model, host_failure, publish_hosted}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, Int4FaithfulnessAssayHostLimits, NativeLimits, RunLimits},
    receipt::CommitmentDomain};

struct Plan {
    candidate: Model,
    baseline: Option<Model>,
    host: Int4FaithfulnessAssayHostLimits,
    kernel: PackedInt4Kernel,
    memory_bytes: u64,
    raw_staging_bytes: u64,
    minimum_memory_bytes: u64,
    elapsed: Duration,
}
impl Plan {
    /// Pure quotation precedes filesystem, tokenizer and model access. Keep
    /// both model residents, all typed rows and the larger sequential engine.
    fn new(command: &Command, limits: Limits) -> Result<Self, Failure> {
        let args = &command.args; let elapsed = Duration::from_secs(args.timeout_seconds);
        let load_run = RunLimits { max_elapsed: elapsed, max_checkpoints: args.snapshot_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES };
        let candidate = Model::new(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(), args.group_width, load_run)?;
        let baseline = command.comparison.as_ref().map(|comparison| Model::new(
            comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
            comparison.baseline_group_width, load_run)).transpose()?;
        let host = Int4FaithfulnessAssayHostLimits { native: NativeLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: elapsed,
                max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
            task: args.task()?, max_heads: args.max_heads,
            preparation_reserve_bytes: mib(args.preparation_mib)?, max_model_work: args.work() };
        let widths = [candidate.pin.group_width, baseline.as_ref().map_or(args.group_width, |model| model.pin.group_width)];
        let need = host.requirements(&widths[..1 + usize::from(baseline.is_some())], limits).map_err(host_failure)?;
        let multiply = |count: usize, bytes: u64| (count as u64).checked_mul(bytes).ok_or_else(invalid_options);
        let raw_staging_bytes = sum(&[multiply(limits.max_input_bytes, 16)?, multiply(limits.max_manifest_bytes, 8)?,
            multiply(limits.max_rows, 1024)?, multiply(limits.max_output_bytes, 4)?, CLEANUP_BYTES])?;
        let residents = sum(&[candidate.memory.resident_bytes,
            baseline.as_ref().map_or(0, |model| model.memory.resident_bytes)])?;
        let codec = candidate.memory.staging_bytes.max(baseline.as_ref().map_or(0, |model| model.memory.staging_bytes));
        let execution = sum(&[need.native.kv_bytes, need.workspace_bytes, need.input_bytes, need.output_reservation_bytes])?;
        let minimum_memory_bytes = sum(&[raw_staging_bytes, residents, codec.max(execution), CLEANUP_BYTES])?;
        let memory_bytes = mib(args.memory_mib.ok_or_else(invalid_options)?)?;
        if minimum_memory_bytes > memory_bytes {
            return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_memory_ceiling"));
        }
        Ok(Self { candidate, baseline, host,
            kernel: PackedInt4Kernel::detect(args.kernel).map_err(|_| invalid_options())?,
            memory_bytes, raw_staging_bytes, minimum_memory_bytes, elapsed })
    }
}
pub(super) fn execute(command: &Command, limits: Limits, input: &mut impl Read, output: &mut impl Write)
    -> Result<(), Failure> {
    let started = Instant::now(); let mut plan = Plan::new(command, limits)?;
    debug_assert!(plan.minimum_memory_bytes <= plan.memory_bytes);
    if !local_io::PROFILE_SUPPORTED {
        return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "protected_file_profile_unavailable"));
    }
    let remaining = || plan.elapsed.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| Failure::new(ErrorCode::BudgetOrTimeout, "native_assay_deadline"));
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: plan.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_resource_host_refused"))?;
    let lease = engine.resources().acquire_lease();
    let _raw_staging = lease.reserve(MemoryClass::JobBuffers, plan.raw_staging_bytes)
        .map_err(|_| Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_input_reservation"))?
        .commit().map_err(|_| Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_input_reservation"))?;
    remaining()?;
    let common = &command.args.common;
    let destination = common.output.as_deref().filter(|path| path.as_os_str() != "-")
        .map(Destination::prepare).transpose().map_err(file_failure)?;
    let key = read_key(common.key_file.as_deref().ok_or_else(invalid_options)?, common.key_id.as_deref().ok_or_else(invalid_options)?)?;
    remaining()?;
    let manifest = read_file(common.suite.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
    let cases = if common.observations.as_os_str() == "-" { read_bytes(input, limits.max_input_bytes)? }
        else { read_file(&common.observations, limits.max_input_bytes)? };
    remaining()?;
    let dataset = native_faithfulness::Dataset::parse(&manifest, &cases, limits)?;
    let gates = command.comparison.as_ref().map(|comparison| {
        let bytes = read_file(comparison.gates.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
        native_faithfulness::parse_gates(&bytes, limits).map_err(Failure::from)
    }).transpose()?;
    drop(manifest); drop(cases);
    let namespace = common.namespace.as_deref().ok_or_else(invalid_options)?.to_owned();
    let options = command.args.statistics.options();
    preflight_replay(&dataset, &key, &namespace, options, limits, command.comparison.is_some(), plan.host.max_heads)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let load_run = |duration| RunLimits { max_elapsed: duration,
        max_checkpoints: command.args.snapshot_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES };
    let candidate = engine.load_int4_snapshot(plan.candidate.path.clone(), plan.candidate.pin,
        plan.candidate.budget, ALLOCATOR_BYTES, load_run(remaining()?), cancellation.clone()).map_err(host_failure)?;
    let baseline = plan.baseline.as_ref().map(|model| engine.load_int4_snapshot(model.path.clone(), model.pin,
        model.budget, ALLOCATOR_BYTES, load_run(remaining()?), cancellation.clone()).map_err(host_failure)).transpose()?;
    plan.host.native.run.max_elapsed = remaining()?;
    match (&command.comparison, baseline.as_ref(), gates) {
        (None, None, None) => {
            let report = engine.evaluate_int4_faithfulness(&candidate, dataset, key, namespace,
                options, limits, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            publish_hosted(report, FAITHFULNESS_PROTOCOL, command.operation(), limits, destination, output, &remaining)
        }
        (Some(comparison), Some(baseline), Some(gates)) => {
            let scope = comparison.host_scope.as_deref().ok_or_else(invalid_options)?.to_owned();
            let report = engine.qualify_int4_faithfulness(&candidate, baseline, dataset, key, namespace,
                scope, gates, options, limits, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            publish_hosted(report, FAITHFULNESS_PROTOCOL, command.operation(), limits, destination, output, &remaining)
        }
        _ => Err(invalid_options()),
    }
}
fn preflight_replay(dataset: &native_faithfulness::Dataset, key: &CommitmentKey, namespace: &str,
    options: native_faithfulness::EvalOptions, limits: Limits, paired: bool, max_heads: usize) -> Result<(), Failure> {
    dataset.validate_head_limit(max_heads)?;
    if paired { options.validate_paired_rows(dataset.locked_test_sources(), limits)?; }
    else { options.validate_rows(dataset.locked_test_sources(), limits)?; }
    key.commit(CommitmentDomain::Configuration, namespace, "native_faithfulness_assay_preflight_v1", &[])
        .map_err(|_| Failure::from(AssayError::Commitment))?;
    Ok(())
}
#[cfg(test)] mod tests;
