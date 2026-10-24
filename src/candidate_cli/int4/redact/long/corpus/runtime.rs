//! One owned reader/writer, resident native allocation and immutable secret scope.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    candidate_cli::batch::{CandidateWriter, completed},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits, RedactionPseudonyms,
        corpus::{CorpusLimits as HostCorpusLimits, RedactionCorpusConfig}},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::{BuiltInTask, redact::{detectors, pseudonym::Pseudonyms, grouped_int4::corpus::check_configuration}},
};

fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Batch,
    }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Batch(error) => error.fault.cancellation.map(cancelled).unwrap_or(CandidateError::Batch),
        HostedError::BatchSetup(error) => error.cancellation.map(cancelled).unwrap_or(CandidateError::Batch),
        HostedError::Int4(Int4HostError::SourceMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Batch),
        HostedError::Int4(Int4HostError::Redaction(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Batch),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Batch,
    }
}
fn key_scope(command: &CorpusArgs, input: &mut impl Read, request: &RedactionRequest)
    -> Result<Option<RedactionPseudonyms>, CandidateError> {
    let secret = command.document.redaction.key(input)?
        .map(|(key, namespace)| RedactionPseudonyms { key: Arc::new(key), namespace });
    {
        let context = secret.as_ref().map(|s| Pseudonyms::full256(&s.key, &s.namespace,
            request.actions.expected_key_commitment.as_deref())).transpose().map_err(|_| CandidateError::Identity)?;
        request.actions.check_key(context.as_ref()).map_err(|_| CandidateError::Arguments)?;
    }
    Ok(secret)
}
pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, command: CorpusArgs, _admitted: CorpusLimits, mut input: R, output: W,
) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let common = limits.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    // Read one binary key from inherited stdin and validate its commitment
    // BEFORE opening or reading the corpus. No row/flush rekeys this scope.
    let request = command.document.request(&args, limits.document);
    let secret = key_scope(&command, &mut input, &request)?;
    remaining()?;
    let options = command.document.redaction.source.options_path().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?,
            crate::candidate_cli::source::OPTIONS_BYTES)
    }).transpose()?;
    let ner = shared::ner_options(options.as_deref())?;
    drop(options);
    drop(detectors::detect("", &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    let (planner, vocabulary) = source::planner()?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::super::super::runtime::load_model(
        &engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let identity = source::identity(model.model_identity(), &planner, BuiltInTask::Ner)?;
    let batch = command.configuration(&args, limits, identity.clone(), ner);
    // Check the fixed complete-document recipe, actual model/group and whole
    // stream envelope before optional model snapshot publication or corpus IO.
    check_configuration(&planner, &batch, model.model_identity()).map_err(|_| CandidateError::Planning)?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-redact-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        redaction_mode: "complete_document", model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes, limits.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    let host = HostCorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: common.preparation_bytes, io_reserve_bytes: limits.io_bytes,
    };
    let summary = engine.batch_int4_redact_documents(&model, Arc::new(planner), Arc::new(vocabulary),
        RedactionCorpusConfig { batch, edit_reserve_bytes: limits.document.edit_bytes }, secret,
        host, common.kernel, reader, writer, cancellation).map_err(execution_error)?;
    // The host emitted the sole terminal frame. Earlier completed rows remain
    // valid, but any failed record still makes the process unsuccessful.
    completed(summary)
}
#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    redaction_mode: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}

#[cfg(test)]
mod tests;
