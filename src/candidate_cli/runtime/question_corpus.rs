//! Owned complete-document QA populations with one checked artifact/deadline.
//! The host retains source, question, compiler and IO through physical join.
use super::*;
use super::source_tasks::{Session, SessionStageError};
use std::{io::{BufRead, BufReader}, sync::Arc};
use crate::{
    batch::{BatchCode, BatchFault, source::quantized::question::Int8QuestionDocumentBatchPlanner},
    candidate_cli::{batch::{CandidateWriter, IO_BUFFER_BYTES},
        question_corpus::{self as command, Admitted, QuestionCorpusCommand}, stream::Failure},
    hosted::{HostedError, corpus::{CorpusLimits, int8_question_document_corpus_requirements}},
    native_engine::{artifact_bridge::ArtifactBridgeError, decode::DecodeCancellationKind,
        strict_int8::{CurrentCandidateInt8Error, StrictInt8Error}},
    tasks::{BuiltInTask, source_planning::{SourcePlanningError,
        quantized::{Int8SourceError, long::Int8SourceMapError}}},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

pub(in crate::candidate_cli) fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "int8_question_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "int8_question_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "int8_question_resources"),
        FailFast | LinkedExit => (ErrorCode::Generic, "int8_question_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "int8_question_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}

pub(in crate::candidate_cli) fn setup_failure(error: CandidateError) -> Failure {
    if error == CandidateError::Timeout { return cancelled(DecodeCancellationKind::Deadline); }
    let (exit, code) = match error {
        CandidateError::Arguments => (ErrorCode::Usage, "int8_question_options"),
        CandidateError::Input => (ErrorCode::InputDecodeOrParse, "int8_question_input"),
        CandidateError::Planning | CandidateError::Memory | CandidateError::Runtime =>
            (ErrorCode::AdmissionOrResourceLimit, "int8_question_admission"),
        CandidateError::Identity | CandidateError::Model =>
            (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "int8_question_model_identity"),
        _ => (ErrorCode::Generic, "int8_question_setup_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}

pub(in crate::candidate_cli) fn session_failure(error: SessionStageError) -> Failure {
    match error {
        SessionStageError::Setup(error) => setup_failure(error),
        SessionStageError::Artifact(ArtifactBridgeError::Cancelled(cause)) => cancelled(cause),
        SessionStageError::Artifact(_) => setup_failure(CandidateError::Model),
        SessionStageError::Host(error) => host_failure(error),
    }
}

pub(in crate::candidate_cli) fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "int8_question_batch_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "int8_question_batch_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "int8_question_batch_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "int8_question_batch_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "int8_question_batch_execution"),
        _ => (ErrorCode::Generic, "int8_question_batch_delivery_or_invariant"),
    };
    Failure { exit, code, cancellation: None }
}

fn native_failure(error: StrictInt8Error) -> Failure {
    match error {
        StrictInt8Error::Cancelled(cause) => cancelled(cause),
        StrictInt8Error::Work => Failure { exit: ErrorCode::BudgetOrTimeout,
            code: "int8_question_native_work", cancellation: None },
        StrictInt8Error::Context | StrictInt8Error::Memory | StrictInt8Error::Allocation =>
            setup_failure(CandidateError::Memory),
        _ => Failure::output("int8_question_native_invariant"),
    }
}
fn source_failure(error: Int8SourceError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error {
        Int8SourceError::Planning(SourcePlanningError::InputBudget | SourcePlanningError::ContextBudget
            | SourcePlanningError::OutputBudget) => Failure { exit: ErrorCode::BudgetOrTimeout,
                code: "int8_question_source_budget", cancellation: None },
        Int8SourceError::Planning(SourcePlanningError::AllocationRefused) => setup_failure(CandidateError::Memory),
        _ => Failure::output("int8_question_source_invariant"),
    }
}
fn map_failure(error: Int8SourceMapError) -> Failure {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error {
        Int8SourceMapError::Source(error) => source_failure(error),
        Int8SourceMapError::InvalidLimits => setup_failure(CandidateError::Arguments),
        Int8SourceMapError::EmptySource => setup_failure(CandidateError::Input),
        Int8SourceMapError::WorkLimit => Failure { exit: ErrorCode::BudgetOrTimeout,
            code: "int8_question_document_budget", cancellation: None },
        Int8SourceMapError::Admission | Int8SourceMapError::Allocation => setup_failure(CandidateError::Memory),
        _ => Failure::output("int8_question_document_invariant"),
    }
}

pub(in crate::candidate_cli) fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error); failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_failure(*error),
        HostedError::BatchSetup(fault) => batch_failure(fault),
        HostedError::Batch(error) => batch_failure(error.fault),
        HostedError::Source(error) => source_failure(error),
        HostedError::SourceMap(error) => map_failure(error),
        HostedError::Native(error) => native_failure(error),
        HostedError::Model(CurrentCandidateInt8Error::Artifact(ArtifactBridgeError::Cancelled(cause)))
            | HostedError::Model(CurrentCandidateInt8Error::Engine(StrictInt8Error::Cancelled(cause))) => cancelled(cause),
        HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Spawn(_)
            | HostedError::Reentrant | HostedError::MissingRuntimeContext
            | HostedError::SingleCoordinatorRequired => setup_failure(CandidateError::Memory),
        HostedError::ModelIdentity | HostedError::ResourceDomain | HostedError::Model(_) => setup_failure(CandidateError::Identity),
        _ => Failure::output("int8_question_runtime_or_completion"),
    }
}

/// Even a refused stage owes its original invocation a terminal poll. Preserve
/// an already observed typed cause when cleanup crosses that same deadline.
pub(in crate::candidate_cli) fn checked_stage<T, E>(result: Result<T, E>,
    check: impl FnOnce() -> Result<(), CandidateError>, classify: impl FnOnce(E) -> Failure)
    -> Result<T, Failure> {
    let checked = check().map_err(setup_failure);
    match result {
        Ok(value) => { checked?; Ok(value) }
        Err(error) => {
            let failure = classify(error);
            if failure.cancellation.is_some() { return Err(failure); }
            checked?; Err(failure)
        }
    }
}

fn corpus_limits(admitted: &Admitted, elapsed: Duration) -> CorpusLimits {
    let document = &admitted.document;
    CorpusLimits { native: NativeLimits { context_tokens: document.common.context_tokens,
        allocator_reserve_bytes: 64 * MIB, run: RunLimits { max_elapsed: elapsed,
            max_checkpoints: document.common.max_checkpoints, cleanup_reserve_bytes: 65_536 } },
        transport: admitted.transport, preparation_reserve_bytes: document.shared.preparation_bytes,
        io_reserve_bytes: admitted.io_bytes }
}

/// Match the actual host quote, including prompt and team storage, before any
/// private stream or model IO. Transport/preparation is charged exactly once
/// inside the host; the caller's simultaneous preparation is a distinct owner.
pub(in crate::candidate_cli) fn admit(admitted: &Admitted) -> Result<u64, CandidateError> {
    let document = &admitted.document;
    let peak = command::validate_peak(admitted)?;
    let quote = int8_question_document_corpus_requirements(admitted.config,
        corpus_limits(admitted, Duration::from_secs(document.common.timeout_seconds)),
        document.reduction_reserve_bytes, document.prefill,
        (document.shared.compute.threads > 1).then_some(document.shared.compute.threads))
        .map_err(|_| CandidateError::Arguments)?;
    let mut expected = admitted.memory;
    expected.workspace_bytes = expected.workspace_bytes.checked_add(document.prefill
        .map(|limits| limits.validate()).transpose().map_err(|_| CandidateError::Arguments)?.unwrap_or(0))
        .ok_or(CandidateError::Arguments)?;
    if quote.document != expected || quote.team_workspace_bytes != document.shared.compute.workspace_bytes {
        return Err(CandidateError::Arguments);
    }
    quote.total_bytes().map_err(|_| CandidateError::Arguments)?;
    Ok(peak)
}

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, source_mode: &'static str, question_source: &'static str, answer_mode: &'static str,
    model_id: &'a str, source_revision: &'a str, source_root_sha256: &'a str,
    logical_model_sha256: &'a str, quant_recipe: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document_encoding_work_limits: Option<SafeBpeWorkLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: QuestionCorpusCommand, declared: Admitted, mut input: R, output: W) -> Result<(), Failure> {
    admit(&declared).map_err(setup_failure)?;
    // First owner: every private request, compiler/vocabulary, retained stream
    // and guarded output drops before this caller's preparation reservation.
    let session = Session::new_with_prefill(&declared.document.common,
        declared.document.shared, declared.document.prefill).map_err(setup_failure)?;
    let check = || session.remaining().map(|_| ());
    let raw_options = command.document.options.as_ref().map(|path|
        checked_stage(session.read(path, &mut input, crate::candidate_cli::source::OPTIONS_BYTES),
            &check, setup_failure)).transpose()?;
    let options = checked_stage(command::parse_options(raw_options.as_deref()), &check, setup_failure)?;
    drop(raw_options);
    let admitted = checked_stage(command.admit_options(options), &check, setup_failure)?;
    if admitted.memory != declared.memory || admitted.document.shared != declared.document.shared {
        return Err(setup_failure(CandidateError::Planning));
    }
    checked_stage(admit(&admitted), &check, setup_failure)?;
    let document = &admitted.document;
    let facts = checked_stage(session.facts_with_error_detail(&document.common), &check, session_failure)?;
    let (planner, vocabulary) = checked_stage(source_tasks::planner(), &check, setup_failure)?;
    let identity = checked_stage(source_tasks::source_identity(&facts, &planner, BuiltInTask::Answer), &check, setup_failure)?;
    let planner = checked_stage(Int8QuestionDocumentBatchPlanner::for_model(&facts, Arc::new(planner), identity,
        admitted.config), &check, batch_failure)?;
    let planner = match document.encoding_work {
        Some(total) => checked_stage(planner.with_map_document_encoding(document.encoding.clone(), total),
            &check, batch_failure)?,
        None => planner,
    };
    if planner.document_encoding() != &document.encoding
        || planner.map_document_encoding_limits() != document.encoding_work
        || checked_stage(planner.encoding_workspace_bytes(), &check, batch_failure)? > document.shared.preparation_bytes
        || planner.max_result_bytes() != admitted.memory.max_output_bytes || planner.max_output_tokens() != 0 {
        return Err(setup_failure(CandidateError::Planning));
    }
    let vocabulary = Arc::new(vocabulary);
    let provenance = Provenance { protocol: "fnlp-candidate-int8-answer-corpus-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", task: "answer",
        source_mode: "complete_document", question_source: "per_record",
        answer_mode: "independent_passage_answers",
        model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id,
        input_encoding: input_encoding::reported_policy(&document.encoding),
        document_encoding_work_limits: document.encoding_work,
        prefill_rows: document.prefill.map(|p| p.max_batch_rows),
        compute_threads: (document.shared.compute.threads > 1).then_some(document.shared.compute.threads) };
    let writer = checked_stage(CandidateWriter::new(output, &provenance,
        admitted.transport.max_output_line_bytes, admitted.output_bytes), &check, setup_failure)?;
    let reader: Box<dyn BufRead + Send> = if document.common.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        let file = checked_stage(crate::local_io::open_document(&document.common.input)
            .map_err(|_| CandidateError::Input), &check, setup_failure)?;
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, file))
    };
    let cancellation = CancellationToken::default();
    let model = checked_stage(session.load_with_error_detail(&document.common, document.shared,
        &facts, cancellation.clone()), &check, session_failure)?;
    let host = corpus_limits(&admitted, checked_stage(session.remaining(), &check, setup_failure)?);
    let width = document.shared.compute.threads;
    let reduction = document.reduction_reserve_bytes;
    let summary = checked_stage(match (width, document.prefill) {
        (1, None) => session.engine.batch_int8_question_documents(&model, planner, vocabulary, host,
            reduction, reader, writer, cancellation),
        (1, Some(prefill)) => session.engine.batch_int8_question_documents_layer_major(&model, planner,
            vocabulary, host, reduction, prefill, reader, writer, cancellation),
        (_, None) => session.engine.batch_int8_question_documents_with_threads(&model, planner,
            vocabulary, host, reduction, crate::native_engine::simd::config::process_kernel(),
            width, reader, writer, cancellation),
        (_, Some(prefill)) => session.engine.batch_int8_question_documents_layer_major_with_threads(&model,
            planner, vocabulary, host, reduction, crate::native_engine::simd::config::process_kernel(),
            width, prefill, reader, writer, cancellation),
    }, &check, host_failure)?;
    // The host already published the only terminal frame after physical join.
    if summary.failed > 0 { return Err(Failure::output("int8_question_corpus_items_failed")); }
    Ok(())
}

#[cfg(test)] mod tests;
