//! One resident INT8 extraction/review stream with complete pre-IO admission.
use super::*;
use std::io::{BufRead, BufReader};
use crate::{
    batch::{BatchCode, BatchFault, extract::semantic::int8::{
        Int8SemanticExtractionBatchLimits, Int8SemanticExtractionBatchPlanner}},
    candidate_cli::{batch::{CandidateWriter, IO_BUFFER_BYTES}, semantic::{self as semantic_cli,
        corpus::{self as command, BatchCommand, BatchEnvelope}}, stream::Failure},
    hosted::{corpus::{CorpusLimits, semantic::{
        Int8SemanticCorpusRequirements, int8_semantic_corpus_requirements}}, HostedError},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: BatchCommand, args: CandidateArgs, limits: Limits, mut input: R, output: W)
    -> Result<(), Failure> {
    let (_, actual_limits) = command.validate().map_err(semantic_cli::population_failure)?;
    let prefill = command.semantic.prefill().map_err(semantic_cli::population_failure)?;
    if actual_limits != limits || args.compute != command.semantic.compute
        || limits.compute != command.semantic.compute.admit_prefill(prefill).map_err(semantic_cli::population_failure)? {
        return Err(Failure::usage("semantic_batch_schedule_admission"));
    }
    // Declared before all private schema/options, factory and writer storage.
    // The caller's preparation charge survives physical host completion.
    let session = Session::new_with_prefill(&args, limits, prefill).map_err(semantic_cli::population_failure)?;
    let options = semantic_cli::parse_options(&session.read(&command.semantic.semantic_options,
        &mut input, semantic_cli::OPTIONS_BYTES).map_err(semantic_cli::population_failure)?)
        .map_err(semantic_cli::population_failure)?;
    let envelope = command.policy_limits(&args, limits, &options).map_err(semantic_cli::population_failure)?;
    let encoding = command.semantic.encoding_policy().map_err(semantic_cli::population_failure)?;
    let host = host_limits(&session, &args, limits, envelope).map_err(semantic_cli::population_failure)?;
    let quote = requirements(&command, limits, &options, envelope, host, prefill)?;
    admit_memory(&args, limits, quote, prefill).map_err(semantic_cli::population_failure)?;

    // Only the bounded options file is read before complete policy/run/owner
    // admission. No private population or model is opened to discover a limit.
    let schema = session.read(&command.semantic.schema, &mut input, extract::SCHEMA_BYTES)
        .map_err(semantic_cli::population_failure)?;
    let request = semantic_cli::fixed_request(&command.semantic, schema,
        command.semantic.host.task_budget(limits), &options).map_err(semantic_cli::population_failure)?;
    session.remaining().map_err(semantic_cli::population_failure)?;
    let facts = session.facts(&args).map_err(semantic_cli::population_failure)?;
    let planner = planner(&facts, &command.semantic, limits, request, options.clone(),
        envelope.native, envelope.encoding_work_limits)?;
    let actual = check_factory(&command, limits, &options, envelope, host, prefill, &planner)?;
    if actual != quote { return Err(Failure::usage("semantic_batch_quote_mismatch")); }
    admit_memory(&args, limits, actual, prefill).map_err(semantic_cli::population_failure)?;
    let vocabulary = super::super::extraction::vocabulary().map_err(semantic_cli::population_failure)?;
    session.remaining().map_err(semantic_cli::population_failure)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone()).map_err(semantic_cli::population_failure)?;
    let provenance = Provenance {
        protocol: command::PROTOCOL, schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", task: "extract-semantic", execution: INT8_SEMANTIC_EXECUTION,
        same_model_correlated: true, calibration: "experimental_uncalibrated_diagnostic_only",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|p| p.max_batch_rows),
        input_encoding: input_encoding::reported_policy(&encoding),
        invocation_encoding_work_limits: envelope.encoding_work_limits,
    };
    let writer = CandidateWriter::new(output, &provenance, envelope.transport.max_output_line_bytes,
        envelope.output_bytes).map_err(semantic_cli::population_failure)?;
    let reader: Box<dyn BufRead + Send> = if command.semantic.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&command.semantic.input).map_err(|_| Failure {
                exit: ErrorCode::InputDecodeOrParse, code: "semantic_batch_input", cancellation: None })?))
    };
    let host = host_limits(&session, &args, limits, envelope).map_err(semantic_cli::population_failure)?;
    let team = (limits.compute.threads > 1).then(||
        (crate::native_engine::simd::config::process_kernel(), limits.compute.threads));
    let summary = session.engine.corpus_int8_semantic_extract_scheduled(&model, planner, vocabulary,
        host, prefill, team, reader, writer, cancellation.clone()).map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(cancelled(cause)); }
    session.remaining().map_err(semantic_cli::population_failure)?;
    command::completed(summary)
}

fn host_limits(session: &Session, args: &CandidateArgs, limits: Limits, envelope: BatchEnvelope)
    -> Result<CorpusLimits, CandidateError> {
    Ok(CorpusLimits { native: session.native(args)?, transport: envelope.transport,
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes })
}

fn requirements(command: &BatchCommand, limits: Limits, options: &Int8SemanticOptions,
    envelope: BatchEnvelope, host: CorpusLimits, prefill: Option<Int8PrefillLimits>)
    -> Result<Int8SemanticCorpusRequirements, Failure> {
    int8_semantic_corpus_requirements(command.semantic.host.task_budget(limits), options,
        semantic_cli::judge_limits(options), &command.semantic.encoding_policy().map_err(semantic_cli::population_failure)?,
        envelope.encoding_work_limits, envelope.native, host, prefill,
        (limits.compute.threads > 1).then_some(limits.compute.threads)).map_err(host_failure)
}

fn planner(facts: &ArtifactIdentity, command: &SemanticCommand, limits: Limits,
    request: ExtractionBatchArgs, options: Int8SemanticOptions,
    work: Int8SemanticExtractionBatchLimits, total: Option<SafeBpeWorkLimits>)
    -> Result<Arc<Int8SemanticExtractionBatchPlanner>, Failure> {
    let encoding = command.encoding_policy().map_err(semantic_cli::population_failure)?;
    let extraction = super::super::extraction::planner(facts, &command.host, limits, Some(request))
        .map_err(semantic_cli::population_failure)?
        .with_document_encoding(encoding.clone()).map_err(batch_failure)?;
    let registry = pinned_controls::pinned().map_err(|_| semantic_cli::population_failure(CandidateError::Identity))?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or_else(|| semantic_cli::population_failure(CandidateError::Identity))?;
    let judge = Arc::new(JudgePlanner::pinned(controls, eos)
        .map_err(|_| semantic_cli::population_failure(CandidateError::Planning))?
        .with_document_encoding(encoding).map_err(|_| semantic_cli::population_failure(CandidateError::Planning))?);
    let mut identity = extraction.base_execution_identity().clone();
    identity.task_spec = "judge-v1".to_owned();
    identity.tokenizer_digest = judge.tokenizer_digest();
    identity.template_digest = *judge.template_digest();
    let planning = semantic_cli::judge_limits(&options);
    Int8SemanticExtractionBatchPlanner::new(extraction, judge, identity, planning, options, work, total)
        .map(Arc::new).map_err(batch_failure)
}

fn check_factory(command: &BatchCommand, limits: Limits, options: &Int8SemanticOptions,
    envelope: BatchEnvelope, host: CorpusLimits, prefill: Option<Int8PrefillLimits>,
    planner: &Int8SemanticExtractionBatchPlanner) -> Result<Int8SemanticCorpusRequirements, Failure> {
    let selected = command.semantic.encoding_policy().map_err(semantic_cli::population_failure)?;
    let actual = planner.limits(); let expected = envelope.native;
    let expected_workspace = match &selected {
        UntrustedEncoding::ByteFallback => 0,
        UntrustedEncoding::SafeBpeV1(call) => call.requirements()
            .map_err(|_| Failure::usage("semantic_batch_encoding_quote"))?.workspace_bytes
            .checked_add(1024).ok_or_else(|| Failure::usage("semantic_batch_encoding_quote"))?,
    };
    if planner.document_encoding() != &selected || planner.encoding_work_limits() != envelope.encoding_work_limits
        || planner.encoding_workspace_bytes().map_err(batch_failure)? != expected_workspace
        || planner.extraction_budget() != command.semantic.host.task_budget(limits)
        || planner.judge_budget() != options.specification.judge_budget
        || planner.semantic_limits() != options.limits
        || planner.max_result_bytes() != options.limits.max_output_bytes
        || actual.max_model_work != expected.max_model_work
        || actual.masks.max_visits_per_item != expected.masks.max_visits_per_item
        || actual.masks.max_visits_per_run != expected.masks.max_visits_per_run
        || actual.masks.per_mask.max_trie_node_visits != expected.masks.per_mask.max_trie_node_visits
        || actual.masks.per_mask.checkpoint_interval_nodes != expected.masks.per_mask.checkpoint_interval_nodes
        || canonjson::canonical_bytes(planner.options()).map_err(|_| Failure::usage("semantic_batch_options_binding"))?
            != canonjson::canonical_bytes(options).map_err(|_| Failure::usage("semantic_batch_options_binding"))? {
        return Err(Failure::usage("semantic_batch_factory_mismatch"));
    }
    int8_semantic_corpus_requirements(planner.extraction_budget(), planner.options(),
        planner.judge_planning_limits(), planner.document_encoding(), planner.encoding_work_limits(),
        planner.limits(), host, prefill, (limits.compute.threads > 1).then_some(limits.compute.threads))
        .map_err(host_failure)
}

pub(super) fn minimum_memory(args: &CandidateArgs, limits: Limits, quote: Int8SemanticCorpusRequirements,
    prefill: Option<Int8PrefillLimits>) -> Result<u64, CandidateError> {
    let native = Int8MemoryRequirement::for_context(args.context_tokens).map_err(|_| CandidateError::Arguments)?;
    let extra = prefill.map(Int8PrefillLimits::validate).transpose()
        .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
    let coordinator = native.rope_bytes.checked_add(native.scratch_payload_bound)
        .and_then(|n| n.checked_add(64 * MIB)).and_then(|n| n.checked_add(extra)).ok_or(CandidateError::Arguments)?;
    if quote.native != native || quote.native_workspace_bytes < coordinator
        || quote.team_workspace_bytes != limits.compute.workspace_bytes
        || limits.compute != args.compute.admit_prefill(prefill)?
        || quote.input_bytes < quote.preparation_bytes || quote.preparation_bytes < limits.preparation_bytes {
        return Err(CandidateError::Arguments);
    }
    let sum = |values: &[u64]| values.iter().try_fold(0_u64,
        |total, &bytes| total.checked_add(bytes).ok_or(CandidateError::Arguments));
    let loading = sum(&[limits.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[limits.weight_bytes, 192 * MIB,
        quote.total_bytes().map_err(|_| CandidateError::Arguments)?, 65_536])?;
    sum(&[limits.preparation_bytes, loading.max(execution)])
}
fn admit_memory(args: &CandidateArgs, limits: Limits, quote: Int8SemanticCorpusRequirements,
    prefill: Option<Int8PrefillLimits>) -> Result<(), CandidateError> {
    if minimum_memory(args, limits, quote, prefill)? > limits.memory_bytes { return Err(CandidateError::Memory); }
    Ok(())
}

pub(super) fn cancelled(cause: DecodeCancellationKind) -> Failure {
    use DecodeCancellationKind::*;
    let (exit, code) = match cause {
        Timeout | Deadline | PollQuota | CostBudget => (ErrorCode::BudgetOrTimeout, "semantic_population_budget"),
        User | ParentCancelled | Shutdown => (ErrorCode::Cancelled, "semantic_population_cancelled"),
        ResourceUnavailable => (ErrorCode::AdmissionOrResourceLimit, "semantic_population_resources"),
        FailFast | LinkedExit => (ErrorCode::Generic, "semantic_population_supervised_failure"),
        RaceLost => (ErrorCode::Generic, "semantic_population_root_race"),
    };
    Failure { exit, code, cancellation: Some(cause) }
}
pub(super) fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "semantic_population_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "semantic_population_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "semantic_population_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "semantic_population_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "semantic_population_no_result"),
        Cancelled => (ErrorCode::Generic, "semantic_population_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "semantic_population_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}
pub(super) fn host_failure(error: HostedError) -> Failure {
    match error {
        HostedError::BatchSetup(fault) => batch_failure(fault),
        HostedError::Batch(error) => batch_failure(error.fault),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_failure(*error),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = host_failure(*error); failure.cancellation = Some(stop.kind); return failure;
                }
            }
            cancelled(stop.kind)
        }
        HostedError::SemanticExtraction(error) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            let (exit, code) = match error {
                Int8SemanticError::WorkBudget | Int8SemanticError::Semantic(
                    crate::tasks::extract::semantic::SemanticError::Limit(_)) =>
                    (ErrorCode::BudgetOrTimeout, "semantic_population_budget"),
                Int8SemanticError::Contract => (ErrorCode::Usage, "semantic_population_policy"),
                Int8SemanticError::Accounting => (ErrorCode::Generic, "semantic_population_accounting"),
                _ => (ErrorCode::StructuredTaskNoResult, "semantic_population_no_result"),
            };
            Failure { exit, code, cancellation: None }
        }
        HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause)) => cancelled(cause),
        HostedError::Limits(_) | HostedError::Reservation(_) | HostedError::Reentrant
            | HostedError::MissingRuntimeContext | HostedError::SingleCoordinatorRequired => Failure {
                exit: ErrorCode::AdmissionOrResourceLimit, code: "semantic_population_admission", cancellation: None },
        HostedError::ModelIdentity | HostedError::ResourceDomain => Failure {
            exit: ErrorCode::ArtifactIntegrityOrFormatOrVersion, code: "semantic_population_model_identity", cancellation: None },
        _ => Failure::output("semantic_population_runtime_or_completion"),
    }
}

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, execution: &'static str, same_model_correlated: bool, calibration: &'static str,
    model_id: &'a str, source_revision: &'a str, source_root_sha256: &'a str,
    logical_model_sha256: &'a str, quant_recipe: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
}

#[cfg(test)] mod tests;
