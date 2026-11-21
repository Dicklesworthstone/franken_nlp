//! Complete fixed-policy admission, one resident stream and typed delivery.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    MemoryClass, NlpEngine,
    batch::{BatchCode, BatchFault},
    candidate_cli::batch::CandidateWriter,
    hosted::{CancellationToken, HostedError, RunLimits,
        Int4SemanticObjectDocumentCorpusConfig, Int4SemanticObjectDocumentCorpusRequirements,
        int4_semantic_object_document_corpus_requirements},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::extract::semantic::grouped_int4::document::INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION,
    tokenizer::untrusted::UntrustedEncoding,
};
use super::super::runtime as document;

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, command: CorpusArgs, _admitted: extract_map::object::ObjectMapLimits,
    input: R, output: W) -> Result<(), Failure> {
    let started = Instant::now();
    let initial = command.validate(&args)?;
    let common = initial.map.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| document::cancelled(DecodeCancellationKind::Deadline));
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let caller_bytes = document::caller_preparation_bytes(&command.document, common)?;
    let _preparation = lease.reserve(MemoryClass::JobBuffers, caller_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let options = parse_options(&document::read_local(&command.document.semantic_options,
        OPTIONS_BYTES, &remaining)?)?;
    let mut limits = command.policy_limits(&args, initial, &options)?;
    let encoding = command.document.object.map.encoding.policy()?;
    let host = document::host_limits(&args, &command.document, limits.document, &options,
        limits.review, remaining()?);
    let quote = int4_semantic_object_document_corpus_requirements(args.group_width,
        limits.task, &options, &encoding, host, limits.work, limits.transport, limits.io_bytes,
        (common.compute.threads > 1).then_some(common.compute.threads)).map_err(host_failure)?;
    let common = admit_memory(&args, common, quote, sum(&[caller_bytes, limits.io_bytes])?)?;
    limits.document.map.common = common;
    let _io = lease.reserve(MemoryClass::JobBuffers, limits.io_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    // Exact declaration-based native, review, compiler and transport peaks
    // precede original schema, population, source weights and snapshot IO.
    let schema = document::read_local(command.document.object.map.extraction.schema_path(),
        crate::candidate_cli::extract::SCHEMA_BYTES, &remaining)?;
    let request = document::source_request(&command.document, &options, schema, limits.task,
        limits.document, &mut DeadlineControl { started, deadline })?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(
        &engine, &args, common, remaining()?, cancellation.clone(), host_failure)?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(host_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    remaining()?;
    let identity = model.model_identity();
    let content: String = identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", identity.matrix_recipe(), identity.group_width());
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-semantic-object-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "extract-semantic-document", execution: INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION,
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: identity.revision(),
        source_root_sha256: identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe, group_width: args.group_width,
        integer_kernel: common.kernel.kind().id(), snapshot,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        input_encoding: input_encoding::reported_policy(&encoding),
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes,
        limits.output_bytes)?;
    let config = Int4SemanticObjectDocumentCorpusConfig {
        extraction: request, encoding,
        host: document::host_limits(&args, &command.document, limits.document, &options,
            limits.review, remaining()?),
        options, work: limits.work, transport: limits.transport, io_reserve_bytes: limits.io_bytes,
    };
    let summary = if common.compute.threads > 1 {
        engine.batch_int4_semantic_object_documents_with_threads(&model, config, common.kernel,
            common.compute.threads, reader, writer, cancellation.clone())
    } else {
        engine.batch_int4_semantic_object_documents(&model, config, common.kernel,
            reader, writer, cancellation.clone())
    }.map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(document::cancelled(cause)); }
    remaining()?;
    completed(summary)
}
fn admit_memory(args: &Int4Args, mut common: Int4Limits,
    quote: Int4SemanticObjectDocumentCorpusRequirements, caller_bytes: u64) -> Result<Int4Limits, Failure> {
    if quote.native != common.native || quote.team_workspace_bytes != common.compute.workspace_bytes {
        return Err(CandidateError::Identity.into());
    }
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    let execution = quote.total_bytes().map_err(host_failure)?;
    common.minimum_memory_bytes = sum(&[caller_bytes, common.load.weight_payload_bytes, ALLOCATOR_BYTES,
        loading.max(saving).max(execution), CLEANUP_BYTES])?;
    if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Memory.into()); }
    Ok(common)
}
fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return document::cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "semantic_object_corpus_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "semantic_object_corpus_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "semantic_object_corpus_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "semantic_object_corpus_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "semantic_object_corpus_no_result"),
        Cancelled => (ErrorCode::Generic, "semantic_object_corpus_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "semantic_object_corpus_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}
fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::BatchSetup(fault) => batch_failure(fault),
        HostedError::Batch(error) => batch_failure(error.fault),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_failure(*error),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            document::cancelled(stop.kind)
        }
        error => document::host_error(error),
    }
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
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, execution: &'static str, same_model_correlated: bool, calibration: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'static str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
}

#[cfg(test)]
mod tests;
