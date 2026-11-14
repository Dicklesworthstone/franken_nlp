//! Complete sampled sentiment publication after native drain and validation.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    grammar::mask::MaskOracleError,
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        SampledSentimentHostLimits},
    native_engine::{constrained::JsonDecodeError, constrained_int4::Int4JsonError,
        decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::sentiment::sampled::{SampledSentimentError, SampledSentimentPlanner,
        SampledSentimentVocabulary},
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}

fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "sampled_sentiment_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_resources"),
        FailFast | LinkedExit => (ErrorCode::Generic, "sampled_sentiment_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "sampled_sentiment_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}

fn task_failure(error: SampledSentimentError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    use SampledSentimentError::*;
    let (exit, code) = match error {
        Limits | WorkBudget | Allocation
            | Native(Int4JsonError::Decode(JsonDecodeError::AllocationRefused))
            | Native(Int4JsonError::Native(StrictInt4Error::Context
                | StrictInt4Error::Memory | StrictInt4Error::Allocation)) =>
            (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_resources"),
        OutputBudget | Native(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded(_)))
            | Native(Int4JsonError::Decode(JsonDecodeError::Mask(MaskOracleError::WorkBudgetExceeded { .. })))
            | Native(Int4JsonError::Native(StrictInt4Error::Work)) =>
            (ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget"),
        NoResult | Native(Int4JsonError::Decode(JsonDecodeError::NoLegalToken
            | JsonDecodeError::IndependentValidation)) =>
            (ErrorCode::StructuredTaskNoResult, "sampled_sentiment_no_result"),
        Identity | Tokenizer | Template | Native(Int4JsonError::ModelIdentity) =>
            (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "sampled_sentiment_identity"),
        Schema => (ErrorCode::SchemaOrRecipeCompile, "sampled_sentiment_schema"),
        _ => (ErrorCode::Generic, "sampled_sentiment_task_invariant"),
    };
    Failure { exit, code, cancellation: None }
}

fn admission_failure(_: CandidateError) -> Failure {
    Failure { exit: ErrorCode::AdmissionOrResourceLimit,
        code: "sampled_sentiment_plan_admission", cancellation: None }
}

fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::Int4(Int4HostError::SampledSentiment(error)) => task_failure(error),
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Int4(Int4HostError::Native(error)) => task_failure(error.into()),
        HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Reentrant
            | HostedError::MissingRuntimeContext | HostedError::SingleCoordinatorRequired => Failure {
                exit: ErrorCode::AdmissionOrResourceLimit, code: "sampled_sentiment_admission", cancellation: None },
        HostedError::ModelIdentity | HostedError::ResourceDomain
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format
                | Int4SnapshotError::Integrity | Int4SnapshotError::Values)) => Failure {
                exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion,
                code: "sampled_sentiment_model_identity", cancellation: None },
        // Panics and lost wrappers never become a successful sample or an
        // ordinary abstention, even if their nested operation was cancelled.
        _ => Failure::output("sampled_sentiment_runtime_or_completion"),
    }
}

pub(super) fn execute(args: Int4Args, command: SampledArgs, _admitted: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let options = command.options(&args)?;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(limits))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let mut request = parse_request(&text,
        command.budget(&args, limits, SentimentAxis::ALL.len())?, args.max_input_bytes)?;
    request.budget = command.budget(&args, limits, request.axes.len())?;
    // Byte encoding cannot fit a source longer than the complete prompt cap.
    // Refuse this obvious no-result before touching model weights.
    if request.document.len() > limits.max_prompt_tokens {
        return Err(admission_failure(CandidateError::Planning));
    }
    let samples = request.axes.len().checked_mul(command.samples).ok_or(CandidateError::Arguments)?;
    drop(text);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model_with_failure(&engine, &args, limits,
        remaining()?, cancellation.clone(), host_failure)?;
    remaining()?;
    let planner = SampledSentimentPlanner::for_model(model.model_identity(), options,
        request.budget, command.task_limits(&args)).map_err(task_failure)?;
    let prepared = planner.plan_with_control(&request, &mut DeadlineControl { started, deadline })
        .map_err(task_failure)?;
    command.admit_plan(prepared.required_context(), args.context_tokens, prepared.planned_work())
        .map_err(admission_failure)?;
    if prepared.sample_count() != samples || prepared.group_width() != args.group_width
        || prepared.preparation_requirement_bytes() > limits.preparation_bytes {
        return Err(admission_failure(CandidateError::Planning));
    }
    let identity = prepared.execution_identity().clone();
    let expected_work = prepared.planned_work();
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().map_err(task_failure)?);
    drop(planner);
    drop(request);
    remaining()?;
    let mut snapshot = limits.snapshot_pin.map(|_| SnapshotRecord::new("loaded", limits.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(host_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", limits.snapshot_need));
    }
    let host = SampledSentimentHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.preparation_bytes, max_model_work: command.work_ceiling(),
        max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        mask_limits: command.masks(), max_mask_node_visits: command.max_mask_node_visits,
    };
    let result = engine.execute_int4_sampled_sentiment(&model, prepared, vocabulary,
        host, limits.kernel, cancellation).map_err(host_failure)?;
    let used = result.result().model_work;
    if result.result().group_width != args.group_width
        || used.forward_positions > expected_work.forward_positions
        || used.projected_logits > expected_work.projected_logits
        || used.attention_pairs > expected_work.attention_pairs
        || !used.projections.fits(expected_work.projections) {
        return Err(Failure::output("sampled_sentiment_completion_work"));
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-sentiment-sampled-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled", model_id: "Nanbeige4.2-3B",
        source_revision: &identity.source_revision, source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: args.group_width,
        integer_kernel: limits.kernel.kind().id(), snapshot,
    };
    publish_checked(&Response { provenance: &provenance, output: result.result() },
        limits.result_bytes + 4096, output, &|| remaining().map(|_| ()))?;
    Ok(())
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
    task: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}
#[derive(Serialize)]
struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    provenance: &'a Provenance<'b>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
