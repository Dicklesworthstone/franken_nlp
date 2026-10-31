//! One process ledger owns raw inputs, resident models, replay and publication.
use super::*;
use std::time::{Duration, Instant};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    assay::native_classify,
    hosted::{CancellationToken, HostedError, Int4AssayHostLimits, Int4LoadRequirements,
        Int4HostError, NativeLimits, RunLimits},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringError,
        strict_int4::{StrictInt4Error, scoring::Int4ScoringError,
        weights::snapshot::{Int4SnapshotBudget, Int4SnapshotError}}},
    receipt::CommitmentDomain,
    tasks::classify::{ClassificationError, ClassificationPlanningError, grouped_int4::Int4ClassificationError}};

struct Model {
    path: PathBuf,
    pin: Int4SnapshotPin,
    budget: Int4SnapshotBudget,
    memory: Int4LoadRequirements,
}
struct Plan {
    candidate: Model,
    baseline: Option<Model>,
    host: Int4AssayHostLimits,
    kernel: PackedInt4Kernel,
    memory_bytes: u64,
    raw_staging_bytes: u64,
    minimum_memory_bytes: u64,
    elapsed: Duration,
}
impl Model {
    fn new(path: Option<&Path>, digest: Option<&str>, width: usize, run: RunLimits) -> Result<Self, Failure> {
        let (pin, need) = model_input(path, digest, width)?;
        let budget = Int4SnapshotBudget { max_weight_bytes: need.weight_payload_bytes,
            max_scratch_bytes: need.scratch_bytes,
            max_io_bytes: need.read_io_bytes().map_err(|_| invalid_options())? };
        let memory = Int4LoadRequirements::for_snapshot(pin, budget, ALLOCATOR_BYTES, run)
            .map_err(host_failure)?;
        Ok(Self { path: path.ok_or_else(invalid_options)?.to_owned(), pin, budget, memory })
    }
}
impl Plan {
    /// Pure complete-peak quotation: no filesystem, parser, runtime or model IO.
    fn new(command: &Command, limits: Limits) -> Result<Self, Failure> {
        let args = &command.args;
        let elapsed = Duration::from_secs(args.timeout_seconds);
        let load_run = RunLimits { max_elapsed: elapsed, max_checkpoints: args.snapshot_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES };
        let candidate = Model::new(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(),
            args.group_width, load_run)?;
        let baseline = command.comparison.as_ref().map(|comparison| Model::new(
            comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
            comparison.baseline_group_width, load_run)).transpose()?;
        let host = Int4AssayHostLimits { native: NativeLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: elapsed,
                max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
            task: args.task()?, preparation_reserve_bytes: mib(args.preparation_mib)?,
            max_model_work: args.work() };
        let widths = [candidate.pin.group_width, baseline.as_ref().map(|model| model.pin.group_width).unwrap_or(args.group_width)];
        let need = host.requirements(&widths[..1 + usize::from(baseline.is_some())], limits).map_err(host_failure)?;
        let multiply = |count: usize, bytes: u64| (count as u64).checked_mul(bytes).ok_or_else(invalid_options);
        // Input bytes, strict parsing, text/ID indices and complete encoded
        // report staging exist before/after the hosted ownership transfer.
        // This guard stays live through the final physical output flush.
        let raw_staging_bytes = sum(&[multiply(limits.max_input_bytes, 8)?,
            multiply(limits.max_manifest_bytes, 8)?, multiply(limits.max_rows, 1024)?,
            multiply(limits.max_output_bytes, 4)?, CLEANUP_BYTES])?;
        let residents = sum(&[candidate.memory.resident_bytes,
            baseline.as_ref().map_or(0, |model| model.memory.resident_bytes)])?;
        let codec = candidate.memory.staging_bytes.max(baseline.as_ref().map_or(0, |model| model.memory.staging_bytes));
        let execution = sum(&[need.native.kv_bytes, need.workspace_bytes, need.input_bytes, need.output_reservation_bytes])?;
        let minimum_memory_bytes = sum(&[raw_staging_bytes, residents, codec.max(execution), CLEANUP_BYTES])?;
        let memory_bytes = mib(args.memory_mib.ok_or_else(invalid_options)?)?;
        if minimum_memory_bytes > memory_bytes {
            return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_memory_ceiling"));
        }
        Ok(Self { candidate, baseline, host, memory_bytes, raw_staging_bytes, minimum_memory_bytes,
            kernel: PackedInt4Kernel::detect(args.kernel).map_err(|_| invalid_options())?, elapsed })
    }
}

pub(super) fn execute(command: &Command, limits: Limits, input: &mut impl Read,
    output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let mut plan = Plan::new(command, limits)?;
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
    let key = read_key(common.key_file.as_deref().ok_or_else(invalid_options)?,
        common.key_id.as_deref().ok_or_else(invalid_options)?)?;
    remaining()?;
    let manifest = read_file(common.suite.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
    let cases = if common.observations.as_os_str() == "-" { read_bytes(input, limits.max_input_bytes)? }
        else { read_file(&common.observations, limits.max_input_bytes)? };
    remaining()?;
    let dataset = native_classify::Dataset::parse(&manifest, &cases, limits)?;
    let gates = command.comparison.as_ref().map(|comparison| {
        let bytes = read_file(comparison.gates.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
        native_classify::parse_gates(&bytes, limits).map_err(Failure::from)
    }).transpose()?;
    drop(manifest); drop(cases);
    let namespace = common.namespace.as_deref().ok_or_else(invalid_options)?.to_owned();
    let options = command.args.metrics.options();
    preflight_replay(&dataset, &key, &namespace, options, limits, command.comparison.is_some())?;
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
            let report = engine.evaluate_int4_classification(&candidate, dataset, key, namespace,
                options, limits, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            remaining()?;
            let bytes = encode(command.operation(), report.result(), limits)?;
            remaining()?;
            publish(destination, &bytes, output)?;
            // The report's output claim and outer staging both cover flush.
            drop(report);
        }
        (Some(comparison), Some(baseline), Some(gates)) => {
            let host_scope = comparison.host_scope.as_deref().ok_or_else(invalid_options)?.to_owned();
            let report = engine.qualify_int4_classification(&candidate, baseline, dataset, key, namespace,
                host_scope, gates, options, limits, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            remaining()?;
            let bytes = encode(command.operation(), report.result(), limits)?;
            remaining()?;
            publish(destination, &bytes, output)?;
            drop(report);
        }
        _ => return Err(invalid_options()),
    }
    Ok(())
}

/// Reject population-dependent work and receipt namespace failures before
/// either expensive snapshot load. The host repeats these ownership checks.
fn preflight_replay(dataset: &native_classify::Dataset, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits, paired: bool) -> Result<(), Failure> {
    dataset.validate_limits(limits)?;
    options.validate_rows(dataset.locked_test_len(), limits)?;
    let arms = if paired { 2 } else { 1 };
    if (dataset.locked_test_len() as u64).checked_mul(options.bootstrap.resamples as u64)
        .and_then(|draws| draws.checked_mul(arms)).is_none_or(|draws| draws > limits.max_bootstrap_draws) {
        return Err(AssayError::InputLimit.into());
    }
    key.commit(CommitmentDomain::Configuration, namespace, "native_assay_preflight_v1", &[])
        .map_err(|_| Failure::from(AssayError::Commitment))?;
    Ok(())
}

#[derive(Serialize)]
struct Envelope<'a, T: Serialize> {
    schema_version: u32, protocol: &'static str, operation: &'static str,
    evidence: &'static str, activation_authority: bool, report: &'a T,
}
fn encode(operation: &'static str, report: &impl Serialize, limits: Limits) -> Result<Vec<u8>, Failure> {
    assay::render(&Envelope { schema_version: 1, protocol: NATIVE_PROTOCOL, operation,
        evidence: "native_model_replay", activation_authority: false, report }, limits).map_err(Into::into)
}
fn publish(destination: Option<Destination>, bytes: &[u8], output: &mut impl Write) -> Result<(), Failure> {
    match destination {
        Some(destination) => destination.stage(bytes).and_then(|stage| stage.publish()).map_err(file_failure),
        None => output.write_all(bytes).and_then(|()| output.flush()).map_err(|_| output_failure()),
    }
}

fn cancelled(kind: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let code = match kind {
        Timeout | Deadline | PollQuota | CostBudget => ErrorCode::BudgetOrTimeout,
        User | Shutdown | ParentCancelled => ErrorCode::Cancelled,
        ResourceUnavailable => ErrorCode::AdmissionOrResourceLimit,
        FailFast | LinkedExit | RaceLost => ErrorCode::Generic,
    };
    Failure::new(code, "native_assay_stopped")
}
fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::Assay(error) => error.into(),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error { return host_failure(*error); }
            }
            cancelled(stop.kind)
        }
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_failure(*error),
        HostedError::Int4(Int4HostError::Snapshot(error)) => match error {
            Int4SnapshotError::Cancelled(kind) => cancelled(kind),
            Int4SnapshotError::Format | Int4SnapshotError::Integrity | Int4SnapshotError::Values =>
                Failure::new(ErrorCode::ArtifactIntegrityOrFormatOrVersion, "snapshot_integrity_or_format"),
            Int4SnapshotError::Budget | Int4SnapshotError::Allocation =>
                Failure::new(ErrorCode::AdmissionOrResourceLimit, "snapshot_admission_refused"),
            Int4SnapshotError::Io => Failure::new(ErrorCode::Generic, "snapshot_io_failed"),
        },
        HostedError::Int4(Int4HostError::Classification(error)) => match error {
            Int4ClassificationError::WorkBudget
                | Int4ClassificationError::Planning(ClassificationPlanningError::WorkBudget) =>
                Failure::new(ErrorCode::BudgetOrTimeout, "native_work_budget"),
            Int4ClassificationError::Planning(ClassificationPlanningError::AllocationRefused
                | ClassificationPlanningError::Task(ClassificationError::AllocationRefused
                    | ClassificationError::Scoring(ScoringError::AllocationRefused)))
                | Int4ClassificationError::Scoring(Int4ScoringError::Allocation
                    | Int4ScoringError::Scoring(ScoringError::AllocationRefused)) =>
                Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_admission_refused"),
            Int4ClassificationError::Native(error)
                | Int4ClassificationError::Scoring(Int4ScoringError::Native(error)) =>
                host_failure(HostedError::Int4(Int4HostError::Native(error))),
            Int4ClassificationError::Identity | Int4ClassificationError::Accounting =>
                Failure::new(ErrorCode::ArtifactIntegrityOrFormatOrVersion, "native_execution_identity"),
            error => error.cancellation().map(cancelled).unwrap_or_else(||
                Failure::new(ErrorCode::StructuredTaskNoResult, "native_classification_failed")),
        },
        HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(kind))) => cancelled(kind),
        HostedError::Int4(Int4HostError::Native(StrictInt4Error::Work)) =>
            Failure::new(ErrorCode::BudgetOrTimeout, "native_work_budget"),
        HostedError::Int4(Int4HostError::Native(StrictInt4Error::Allocation | StrictInt4Error::Memory | StrictInt4Error::Context))
            | HostedError::Limits(_) | HostedError::Reservation(_) =>
                Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_admission_refused"),
        HostedError::ModelIdentity | HostedError::ResourceDomain =>
            Failure::new(ErrorCode::ArtifactIntegrityOrFormatOrVersion, "native_model_identity"),
        _ => Failure::new(ErrorCode::Generic, "native_execution_failed"),
    }
}

#[cfg(test)] mod tests;
