//! One bounded ownership chain from strict source input through physical flush.
use super::*;
use std::time::{Duration, Instant};
use super::super::native::{ALLOCATOR_BYTES, CLEANUP_BYTES, sum,
    runtime::{Model, host_failure, publish_hosted}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    grammar::{compiler::CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits,
        source_index::SourceLanguageLimits},
    hosted::{CancellationToken, Int4NerAssayHostLimits, NativeLimits, RunLimits},
    receipt::CommitmentDomain,
    tasks::source_planning::SourcePlanningLimits,
    validation::grounded_fields::GroundingBudget};

struct Plan {
    candidate: Model,
    baseline: Option<Model>,
    host: Int4NerAssayHostLimits,
    kernel: PackedInt4Kernel,
    memory_bytes: u64,
    raw_staging_bytes: u64,
    minimum_memory_bytes: u64,
    elapsed: Duration,
}
impl Plan {
    /// Quote both resident models, one sequential native workspace, complete
    /// typed gold, compiler/source preparation, replay and encoded report
    /// before filesystem, tokenizer, resource-runtime or model access.
    fn new(command: &Command, limits: Limits, ner: native_ner::NerLimits) -> Result<Self, Failure> {
        let args = &command.args;
        let elapsed = Duration::from_secs(args.timeout_seconds);
        let load_run = RunLimits { max_elapsed: elapsed, max_checkpoints: args.snapshot_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES };
        let candidate = Model::new(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(),
            args.group_width, load_run)?;
        let baseline = command.comparison.as_ref().map(|comparison| Model::new(
            comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
            comparison.baseline_group_width, load_run)).transpose()?;
        let planning = SourcePlanningLimits { max_input_bytes: limits.max_line_bytes,
            max_context_tokens: args.context_tokens, max_passages: 1,
            compiler: CompileLimits { max_output_bytes: args.max_result_bytes,
                max_states: args.max_grammar_states as usize, max_array_items: ner.max_spans_per_row,
                ..CompileLimits::default() },
            source: SourceRuntimeLimits {
                index: SourceLanguageLimits { max_source_bytes: limits.max_line_bytes,
                    max_matches: ner.max_occurrences_per_row, ..SourceLanguageLimits::default() },
                // Fixed declared per-row policy; the host additionally applies
                // one nonrenewable allowance across all three source scans.
                verification: GroundingBudget { max_fields: ner.max_spans_per_row,
                    max_matches: ner.max_occurrences_per_row, max_scan_steps: ner.max_verification_steps } } };
        let host = Int4NerAssayHostLimits { native: NativeLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: elapsed,
                max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
            task: args.task()?, planning, preparation_reserve_bytes: mib(args.preparation_mib)?,
            max_model_work: args.work(), mask_limits: MaskWorkLimits {
                max_trie_node_visits: args.mask_step_node_visits, checkpoint_interval_nodes: 256 },
            max_mask_node_visits: args.max_mask_node_visits };
        let widths = [candidate.pin.group_width, baseline.as_ref().map_or(args.group_width, |model| model.pin.group_width)];
        let need = host.requirements(&widths[..1 + usize::from(baseline.is_some())], limits, ner).map_err(host_failure)?;
        let multiply = |count: usize, bytes: u64| (count as u64).checked_mul(bytes).ok_or_else(invalid_options);
        // Gold validation additionally owns one usize per possible source
        // scalar boundary while the raw input and strict/typed rows coexist.
        let raw_staging_bytes = sum(&[multiply(limits.max_input_bytes, 16)?,
            multiply(limits.max_manifest_bytes, 8)?, multiply(limits.max_rows, 1024)?,
            multiply(limits.max_rows, multiply(ner.max_spans_per_row, 64)?)?,
            multiply(ner.max_languages, 256)?, multiply(limits.max_output_bytes, 4)?, CLEANUP_BYTES])?;
        let residents = sum(&[candidate.memory.resident_bytes,
            baseline.as_ref().map_or(0, |model| model.memory.resident_bytes)])?;
        let codec = candidate.memory.staging_bytes.max(baseline.as_ref().map_or(0, |model| model.memory.staging_bytes));
        let execution = sum(&[need.native.kv_bytes, need.workspace_bytes, need.input_bytes, need.output_reservation_bytes])?;
        let minimum_memory_bytes = sum(&[raw_staging_bytes, residents, codec.max(execution), CLEANUP_BYTES])?;
        let memory_bytes = mib(args.memory_mib.ok_or_else(invalid_options)?)?;
        if minimum_memory_bytes > memory_bytes {
            return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "native_memory_ceiling"));
        }
        Ok(Self { candidate, baseline, host, kernel: PackedInt4Kernel::detect(args.kernel).map_err(|_| invalid_options())?,
            memory_bytes, raw_staging_bytes, minimum_memory_bytes, elapsed })
    }
}

pub(super) fn execute(command: &Command, limits: Limits, ner: native_ner::NerLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let mut plan = Plan::new(command, limits, ner)?;
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
    let dataset = native_ner::Dataset::parse(&manifest, &cases, limits, ner)?;
    let gates = command.comparison.as_ref().map(|comparison| {
        let bytes = read_file(comparison.gates.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
        native_ner::parse_gates(&bytes, limits).map_err(Failure::from)
    }).transpose()?;
    drop(manifest); drop(cases);
    let namespace = common.namespace.as_deref().ok_or_else(invalid_options)?.to_owned();
    let options = command.args.statistics.options();
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
            let report = engine.evaluate_int4_ner(&candidate, dataset, key, namespace,
                options, limits, ner, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            publish_hosted(report, NER_PROTOCOL, command.operation(), limits, destination, output, &remaining)
        }
        (Some(comparison), Some(baseline), Some(gates)) => {
            let host_scope = comparison.host_scope.as_deref().ok_or_else(invalid_options)?.to_owned();
            let report = engine.qualify_int4_ner(&candidate, baseline, dataset, key, namespace,
                host_scope, gates, options, limits, ner, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            publish_hosted(report, NER_PROTOCOL, command.operation(), limits, destination, output, &remaining)
        }
        _ => Err(invalid_options()),
    }
}

/// Population-dependent bootstrap and private receipt authority are checked
/// before either expensive snapshot is opened. The host repeats these checks.
fn preflight_replay(dataset: &native_ner::Dataset, key: &CommitmentKey, namespace: &str,
    options: native_ner::EvalOptions, limits: Limits, paired: bool) -> Result<(), Failure> {
    if paired { options.validate_paired_rows(dataset.replay_len(), limits)?; }
    else { options.validate_rows(dataset.replay_len(), limits)?; }
    key.commit(CommitmentDomain::Configuration, namespace, "native_ner_assay_preflight_v1", &[])
        .map_err(|_| Failure::from(AssayError::Commitment))?;
    Ok(())
}

#[cfg(test)] mod tests;
