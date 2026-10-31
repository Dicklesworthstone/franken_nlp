//! Complete owned input, exact-JSON validation and bounded native publication.
use super::*;
use std::time::{Duration, Instant};
use super::super::native::{ALLOCATOR_BYTES, CLEANUP_BYTES, sum,
    runtime::{Model, host_failure, publish_hosted}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    grammar::mask::MaskWorkLimits,
    hosted::{CancellationToken, Int4ExtractAssayHostLimits, NativeLimits, RunLimits},
    receipt::CommitmentDomain};

struct Plan {
    candidate: Model,
    baseline: Option<Model>,
    host: Int4ExtractAssayHostLimits,
    kernel: PackedInt4Kernel,
    memory_bytes: u64,
    raw_staging_bytes: u64,
    minimum_memory_bytes: u64,
    elapsed: Duration,
}
impl Plan {
    /// Pure complete peak quotation before files, tokenizers or runtime access.
    /// Exact gold trees are bounded by both encoded input and declared nodes;
    /// source-schema roots survive parsing, but only one compiler is active.
    fn new(command: &Command, limits: Limits, extract: native_extract::ExtractLimits) -> Result<Self, Failure> {
        let args = &command.args;
        let elapsed = Duration::from_secs(args.timeout_seconds);
        let load_run = RunLimits { max_elapsed: elapsed, max_checkpoints: args.snapshot_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES };
        let candidate = Model::new(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(),
            args.group_width, load_run)?;
        let baseline = command.comparison.as_ref().map(|comparison| Model::new(
            comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
            comparison.baseline_group_width, load_run)).transpose()?;
        let host = Int4ExtractAssayHostLimits { native: NativeLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: elapsed,
                max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
            task: args.task()?, compiler: args.compiler(), source: args.source(limits),
            preparation_reserve_bytes: mib(args.preparation_mib)?, max_model_work: args.work(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: args.mask_step_node_visits, checkpoint_interval_nodes: 256 },
            max_mask_node_visits: args.max_mask_node_visits };
        let widths = [candidate.pin.group_width, baseline.as_ref().map_or(args.group_width, |model| model.pin.group_width)];
        let need = host.requirements(&widths[..1 + usize::from(baseline.is_some())], limits, extract).map_err(host_failure)?;
        let multiply = |count: usize, bytes: u64| (count as u64).checked_mul(bytes).ok_or_else(invalid_options);
        let gold_nodes = multiply(limits.max_rows, extract.max_json_nodes_per_row as u64)?
            .min(limits.max_input_bytes as u64);
        // Exact JSON trees, original strings and bounded duplicate-key parser
        // state coexist. Value-only parsing retains no path map, but malformed
        // nested input still needs one active stack of temporary path prefixes.
        let parser_stack = multiply(limits.max_line_bytes, 2)?.checked_mul(
            (extract.max_json_depth as u64).checked_add(1).ok_or_else(invalid_options)?)
            .ok_or_else(invalid_options)?;
        let compiler = host.compiler;
        let schema_shapes = multiply(extract.max_schemas, sum(&[
            multiply(compiler.max_states, 512)?, multiply(compiler.max_transitions, 128)?])?)?;
        let compiler_staging = sum(&[multiply(compiler.max_schema_bytes, 64)?,
            multiply(compiler.max_states, 512)?, multiply(compiler.max_transitions, 128)?,
            compiler.max_mask_bytes as u64, multiply(compiler.max_string_bytes, 16)?,
            multiply(host.source.index.max_index_bytes, 2)?,
            multiply(extract.max_occurrences_per_row, 256)?])?;
        let raw_staging_bytes = sum(&[multiply(limits.max_input_bytes, 16)?,
            // Typed catalogue roots retain bounded metadata while each
            // compiled automaton and source-verification scratch is reused.
            multiply(limits.max_manifest_bytes, 72)?, multiply(limits.max_rows, 1024)?,
            gold_nodes.checked_mul(128).ok_or_else(invalid_options)?,
            multiply(extract.max_schemas, 512)?, multiply(extract.max_languages, 256)?,
            schema_shapes, compiler_staging, parser_stack, multiply(limits.max_output_bytes, 4)?, CLEANUP_BYTES])?;
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

pub(super) fn execute(command: &Command, limits: Limits, extract: native_extract::ExtractLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let mut plan = Plan::new(command, limits, extract)?;
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
    let dataset = native_extract::Dataset::parse(&manifest, &cases, limits, extract, plan.host.compiler, plan.host.source)?;
    let gates = command.comparison.as_ref().map(|comparison| {
        let bytes = read_file(comparison.gates.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
        native_extract::parse_gates(&bytes, limits).map_err(Failure::from)
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
            let report = engine.evaluate_int4_extraction(&candidate, dataset, key, namespace,
                options, limits, extract, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            publish_hosted(report, EXTRACT_PROTOCOL, command.operation(), limits, destination, output, &remaining)
        }
        (Some(comparison), Some(baseline), Some(gates)) => {
            let host_scope = comparison.host_scope.as_deref().ok_or_else(invalid_options)?.to_owned();
            let report = engine.qualify_int4_extraction(&candidate, baseline, dataset, key, namespace,
                host_scope, gates, options, limits, extract, plan.host, plan.kernel, cancellation).map_err(host_failure)?;
            publish_hosted(report, EXTRACT_PROTOCOL, command.operation(), limits, destination, output, &remaining)
        }
        _ => Err(invalid_options()),
    }
}

fn preflight_replay(dataset: &native_extract::Dataset, key: &CommitmentKey, namespace: &str,
    options: native_extract::EvalOptions, limits: Limits, paired: bool) -> Result<(), Failure> {
    if paired { options.validate_paired_rows(dataset.replay_len(), limits)?; }
    else { options.validate_rows(dataset.replay_len(), limits)?; }
    key.commit(CommitmentDomain::Configuration, namespace, "native_extract_assay_preflight_v1", &[])
        .map_err(|_| Failure::from(AssayError::Commitment))?;
    Ok(())
}

#[cfg(test)] mod tests;
