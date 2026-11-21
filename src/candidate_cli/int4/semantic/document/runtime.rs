//! Own the complete extraction/review result through checked publication.
use super::*;
use std::time::{Duration, Instant};
use crate::{
    MemoryClass, NlpEngine,
    batch::extract::ExtractionBatchArgs,
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        Int4SemanticObjectDocumentHostLimits, Int4SemanticObjectDocumentRequirements,
        int4_semantic_object_document_requirements},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::extract::{object_reconcile::{RootObjectError, RootObjectSchema},
        semantic::{SemanticError, grouped_int4::{Int4SemanticError,
            document::Int4SemanticObjectDocumentError}}},
    tasks::judge::grouped_int4::{Int4JudgeError, faithfulness_map::Int4FaithfulnessMapError},
    tokenizer::untrusted::UntrustedEncoding,
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}

pub(super) fn execute(args: Int4Args, command: SemanticObjectArgs,
    _admitted: extract_map::object::ObjectMapLimits, input: &mut impl Read, output: &mut impl Write)
    -> Result<(), Failure> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let initial = limits.map.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    let engine = NlpEngine::builder().resource_config(compute::resource_config(initial))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Bounded fixed configuration, original source/schema construction and
    // caller-owned inputs coexist with the host's separate owned invocation.
    let caller_bytes = caller_preparation_bytes(&command, initial)?;
    let _preparation = lease.reserve(MemoryClass::JobBuffers, caller_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let options = parse_options(&read_local(&command.semantic_options, OPTIONS_BYTES, &remaining)?)?;
    let review = command.review_limits(&args, initial, &options)?;
    let encoding = command.object.map.encoding.policy()?;
    let task = command.object.map.extraction.budget(&args, initial);
    let mut host = host_limits(&args, &command, limits, &options, review, remaining()?);
    let quote = int4_semantic_object_document_requirements(args.group_width, task,
        &options, &encoding, host, (initial.compute.threads > 1).then_some(initial.compute.threads))
        .map_err(host_error)?;
    let envelope_bytes = options.limits.max_output_bytes.checked_add(4096)
        .and_then(|bytes| usize::try_from(bytes).ok()).ok_or(CandidateError::Memory)?;
    let staging_bytes = (envelope_bytes as u64).checked_mul(2).ok_or(CandidateError::Memory)?;
    let common = admit_memory(&args, initial, quote, sum(&[caller_bytes, staging_bytes])?)?;
    let _staging = lease.reserve(MemoryClass::JobBuffers, staging_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    // All declared simultaneous memory and fixed task policy are admitted
    // before private schema/source access, source conversion or snapshot IO.
    let schema = read_local(command.object.map.extraction.schema_path(),
        crate::candidate_cli::extract::SCHEMA_BYTES, &remaining)?;
    let request = source_request(&command, &options, schema, task, limits,
        &mut DeadlineControl { started, deadline })?;
    remaining()?;
    let source = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_local(&args.input, args.max_input_bytes, &remaining)? };
    if source.is_empty() { return Err(CandidateError::Input.into()); }
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(
        &engine, &args, common, remaining()?, cancellation.clone(), host_error)?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(host_error)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    host.native.run.max_elapsed = remaining()?;
    let result = if common.compute.threads > 1 {
        compute::Completion::Team(engine.extract_int4_semantic_object_document_with_threads(
            &model, source, request, options, encoding.clone(), host, common.kernel,
            common.compute.threads, cancellation.clone()).map_err(host_error)?)
    } else {
        compute::Completion::serial(engine.extract_int4_semantic_object_document(
            &model, source, request, options, encoding.clone(), host, common.kernel,
            cancellation.clone()).map_err(host_error)?)
    };
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    remaining()?;
    let identity = model.model_identity();
    let content: String = identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", identity.matrix_recipe(), identity.group_width());
    let response = Response {
        protocol: "fnlp-candidate-int4-semantic-object-document-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: "Nanbeige4.2-3B", source_revision: identity.revision(),
        source_root_sha256: identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe,
        group_width: identity.group_width(), integer_kernel: common.kernel.kind().id(),
        snapshot, input_encoding: input_encoding::reported_policy(&encoding), output: result.result(),
    };
    // Borrow the actual guarded complete result through serialization and
    // flush. The physical team has joined; no per-field prefix can escape.
    let response = compute::Response { response: &response, compute: result.compute() };
    publish(&response, envelope_bytes, output, &remaining, &cancellation)
}

fn source_request(command: &SemanticObjectArgs, options: &Int4SemanticOptions, schema: String,
    task: TaskBudget, limits: extract_map::object::ObjectMapLimits, control: &mut impl DecodeStepControl)
    -> Result<ExtractionBatchArgs, Failure> {
    let original = RootObjectSchema::derive(&schema, command.object.map.extraction.compiler(),
        limits.object, control).map_err(|error| match error {
            RootObjectError::Cancelled(cause) => cancelled(cause),
            _ => CandidateError::Planning.into(),
        })?;
    options.validate_schema(original.original_schema()).map_err(semantic_error)?;
    drop(original);
    command.object.request(schema, task, limits, control).map_err(|error| match error {
        CandidateError::Timeout => cancelled(DecodeCancellationKind::Deadline),
        error => error.into(),
    })
}
fn caller_preparation_bytes(command: &SemanticObjectArgs, common: Int4Limits) -> Result<u64, Failure> {
    // Original/partial schema derivation occurs before hosted preparation.
    // Keep its independent declaration-based charge alive through that phase;
    // a tiny result cap cannot hide a large optional schema or grammar bound.
    let schemas = RootObjectSchema::workspace_bytes(command.object.map.extraction.compiler())
        .map_err(|_| CandidateError::Memory)?;
    sum(&[common.preparation_bytes, OPTIONS_BYTES as u64 * 16, schemas]).map_err(Failure::from)
}
fn host_limits(args: &Int4Args, command: &SemanticObjectArgs,
    limits: extract_map::object::ObjectMapLimits, options: &Int4SemanticOptions,
    review: Int4FaithfulnessMapLimits, remaining: Duration) -> Int4SemanticObjectDocumentHostLimits {
    Int4SemanticObjectDocumentHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        compiler: command.object.map.extraction.compiler(), source: command.object.map.extraction.source_limits(),
        extraction: limits.mapping(), judge_planning: judge_limits(options), review,
        preparation_reserve_bytes: limits.map.common.preparation_bytes,
        reduction_reserve_bytes: limits.map.reduction_reserve_bytes,
    }
}
fn admit_memory(args: &Int4Args, mut common: Int4Limits,
    quote: Int4SemanticObjectDocumentRequirements, caller_bytes: u64) -> Result<Int4Limits, Failure> {
    if quote.native != common.native || quote.team_workspace_bytes != common.compute.workspace_bytes {
        return Err(CandidateError::Identity.into());
    }
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    let execution = quote.total_bytes().map_err(host_error)?;
    common.minimum_memory_bytes = sum(&[caller_bytes, common.load.weight_payload_bytes, ALLOCATOR_BYTES,
        loading.max(saving).max(execution), CLEANUP_BYTES])?;
    if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Memory.into()); }
    Ok(common)
}
fn read_local(path: &std::path::Path, cap: usize, remaining: &impl Fn() -> Result<Duration, Failure>)
    -> Result<String, Failure> {
    remaining()?;
    let text = read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, cap)?;
    remaining()?;
    Ok(text)
}
fn semantic_error(error: Int4SemanticError) -> Failure {
    error.cancellation().map(cancelled).unwrap_or_else(|| CandidateError::Planning.into())
}
fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "semantic_object_map_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "semantic_object_map_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "semantic_object_map_resources"),
        FailFast | LinkedExit => (ErrorCode::Generic, "semantic_object_map_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "semantic_object_map_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}
fn host_error(error: HostedError) -> Failure {
    match error {
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_error(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::Int4(Int4HostError::SemanticObjectDocument(error)) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            let (exit, code) = match error {
                Int4SemanticObjectDocumentError::WorkBudget | Int4SemanticObjectDocumentError::OutputBudget
                    | Int4SemanticObjectDocumentError::JudgeMap(Int4FaithfulnessMapError::Work
                        | Int4FaithfulnessMapError::Output)
                    | Int4SemanticObjectDocumentError::Semantic(Int4SemanticError::WorkBudget
                        | Int4SemanticError::Semantic(SemanticError::Limit(_))
                        | Int4SemanticError::Judge(Int4JudgeError::WorkBudget)) =>
                    (ErrorCode::BudgetOrTimeout, "semantic_object_map_budget"),
                Int4SemanticObjectDocumentError::Contract
                    | Int4SemanticObjectDocumentError::Semantic(Int4SemanticError::Contract) =>
                    (ErrorCode::Usage, "semantic_object_map_policy"),
                Int4SemanticObjectDocumentError::Semantic(Int4SemanticError::Semantic(SemanticError::AllocationRefused))
                    | Int4SemanticObjectDocumentError::JudgeMap(Int4FaithfulnessMapError::Allocation) =>
                    (ErrorCode::AdmissionOrResourceLimit, "semantic_object_map_resources"),
                Int4SemanticObjectDocumentError::Accounting => (ErrorCode::Generic, "semantic_object_map_accounting"),
                _ => (ErrorCode::StructuredTaskNoResult, "semantic_object_map_no_result"),
            };
            Failure { exit, code, cancellation: None }
        }
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Int4(Int4HostError::ObjectExtractionMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or_else(|| CandidateError::Planning.into()),
        HostedError::Int4(Int4HostError::Judge(error)) =>
            error.cancellation().map(cancelled).unwrap_or_else(|| CandidateError::Planning.into()),
        HostedError::Int4(Int4HostError::FaithfulnessMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or_else(|| CandidateError::Planning.into()),
        HostedError::Int4(Int4HostError::SemanticExtraction(error)) => semantic_error(error),
        HostedError::BatchSetup(fault) =>
            fault.cancellation.map(cancelled).unwrap_or_else(|| CandidateError::Planning.into()),
        HostedError::Reservation(_) => CandidateError::Memory.into(),
        HostedError::ModelIdentity | HostedError::ResourceDomain
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format
                | Int4SnapshotError::Integrity | Int4SnapshotError::Values)) => CandidateError::Identity.into(),
        HostedError::Limits(_) => CandidateError::Arguments.into(),
        _ => CandidateError::Execution.into(),
    }
}
fn publish<T: Serialize>(response: &T, cap: usize, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, Failure>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    publish_checked(response, cap, output, &|| {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ()).map_err(|_| CandidateError::Timeout)
    }).map_err(|error| match (error, cancellation.cause()) {
        (_, Some(cause)) => cancelled(cause),
        (CandidateError::Timeout, None) => cancelled(DecodeCancellationKind::Deadline),
        (error, None) => error.into(),
    })
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    same_model_correlated: bool, calibration: &'static str, model_id: &'static str,
    source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'static str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
