//! One authenticated retained population on one resident two-stage INT8 runner.
use super::*;
use crate::{
    batch::extract::semantic::int8::Int8SemanticExtractionBatchLimits,
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs as job_cli,
        semantic::{self as semantic_cli, jobs::{self as command, JobCommand, JobEnvelope, storage_failure}},
        stream::Failure},
    hosted::corpus::{semantic::Int8SemanticCorpusRequirements, jobs::{HostedJobError,
        JobHostLimits, SourceJobRequest, semantic::{Int8SemanticJobRequirements,
            int8_semantic_job_requirements}}},
    jobs::{JobError, JobInputMemory, JobLimits, JobSecret, runner::{JobRunError,
        extract::semantic::{Int8SemanticExtractionJobConfig, Int8SemanticExtractionJobPlanner}}},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static>(command: JobCommand,
    args: CandidateArgs, limits: Limits, mut input: R, output: &mut impl Write) -> Result<(), Failure> {
    let (_, actual_limits) = command.validate()?;
    let prefill = command.semantic.prefill().map_err(semantic_cli::population_failure)?;
    if actual_limits != limits || args.compute != command.semantic.compute
        || limits.compute != command.semantic.compute.admit_prefill(prefill).map_err(semantic_cli::population_failure)? {
        return Err(Failure::usage("semantic_job_schedule_admission"));
    }
    // First declared, last dropped: options/schema/factory/output staging never
    // outlive their separate caller preparation reservation.
    let session = Session::new_with_prefill(&args, limits, prefill).map_err(semantic_cli::population_failure)?;
    let options = semantic_cli::parse_options(&session.read(&command.semantic.semantic_options,
        &mut input, semantic_cli::OPTIONS_BYTES).map_err(semantic_cli::population_failure)?)
        .map_err(semantic_cli::population_failure)?;
    let job = job_cli::parse_limits(&session.read(&command.storage.limits_file,
        &mut input, job_cli::LIMIT_BYTES).map_err(semantic_cli::population_failure)?).map_err(storage_failure)?;
    let envelope = command.lifetime_limits(&args, limits, &options, job)?;
    let encoding = command.semantic.encoding_policy().map_err(semantic_cli::population_failure)?;
    let input_memory = command.storage.input_retention.input_memory().map_err(storage_failure)?;
    let host = host_limits(&session, &args, envelope).map_err(semantic_cli::population_failure)?;
    let quote = requirements(&command, limits, &options, envelope, host, job, input_memory, prefill)?;
    admit_memory(&args, limits, quote, prefill).map_err(semantic_cli::population_failure)?;

    // No secret, private schema/population or model is opened before complete
    // source/review/recipe/stored-result/native/team/output peak admission.
    let key = JobSecret::read(&command.storage.key_file).map_err(job_failure)?;
    session.remaining().map_err(semantic_cli::population_failure)?;
    let schema = session.read(&command.semantic.schema, &mut input, extract::SCHEMA_BYTES)
        .map_err(semantic_cli::population_failure)?;
    let request = semantic_cli::fixed_request(&command.semantic, schema,
        command.semantic.host.task_budget(limits), &options).map_err(semantic_cli::population_failure)?;
    let schema_digest = Sha256Digest::of_bytes(request.schema.as_bytes());
    let facts = session.facts(&args).map_err(semantic_cli::population_failure)?;
    let factory = prepare(&facts, &command.semantic, limits, request, options.clone(),
        envelope.native, envelope.encoding_work_limits)?;
    let actual = check_factory(&command, limits, &options, schema_digest, envelope, host, job,
        input_memory, prefill, &factory)?;
    if actual != quote { return Err(Failure::usage("semantic_job_quote_mismatch")); }
    admit_memory(&args, limits, actual, prefill).map_err(semantic_cli::population_failure)?;
    let vocabulary = super::super::extraction::vocabulary().map_err(semantic_cli::population_failure)?;
    session.remaining().map_err(semantic_cli::population_failure)?;
    // Retained replay returns an empty owned reader; authentication restores
    // pending originals under the host lock and never falls back to stdin.
    let reader = job_cli::population_reader(command.mode, command.storage.input_retention,
        &command.semantic.input, input, IO_BUFFER_BYTES).map_err(storage_failure)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone()).map_err(semantic_cli::population_failure)?;
    let host = host_limits(&session, &args, envelope).map_err(semantic_cli::population_failure)?;
    let mode = command.mode; let job_id = command.storage.job_id;
    let materialize = command.storage.materialize;
    let request = SourceJobRequest { root: command.storage.job_dir, key, job_id, limits: job,
        mode: super::super::owned_jobs::open_mode(mode, command.storage.input_retention).map_err(storage_failure)?,
        materialize };
    let team = (limits.compute.threads > 1).then(||
        (crate::native_engine::simd::config::process_kernel(), limits.compute.threads));
    let progress = session.engine.job_int8_semantic_extract_scheduled(&model, factory, vocabulary,
        request, host, prefill, team, reader, cancellation.clone()).map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(super::corpus::cancelled(cause)); }
    session.remaining().map_err(semantic_cli::population_failure)?;
    let progress = super::super::owned_jobs::progress_report(progress, job_id, job, materialize,
        mode, "extract-semantic").map_err(storage_failure)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &progress };
    let response = Metadata { response: &response, protocol: command::PROTOCOL,
        execution: INT8_SEMANTIC_EXECUTION, same_model_correlated: true,
        calibration: "experimental_uncalibrated_diagnostic_only",
        compute_threads: team.map(|(_, width)| width), prefill_rows: prefill.map(|limits| limits.max_batch_rows),
        input_encoding: input_encoding::reported_policy(&encoding),
        invocation_encoding_work_limits: envelope.encoding_work_limits };
    publish_report(&response, output, &|| session.remaining(), &cancellation)
}

fn host_limits(session: &Session, args: &CandidateArgs, envelope: JobEnvelope) -> Result<JobHostLimits, CandidateError> {
    Ok(JobHostLimits { native: session.native(args)?, transport: envelope.transport,
        preparation_reserve_bytes: envelope.preparation_bytes, io_reserve_bytes: envelope.io_bytes,
        journal_reserve_bytes: envelope.journal_bytes, serialization_reserve_bytes: envelope.serialization_bytes })
}
fn requirements(command: &JobCommand, limits: Limits, options: &Int8SemanticOptions,
    envelope: JobEnvelope, host: JobHostLimits, job: JobLimits, input: JobInputMemory,
    prefill: Option<Int8PrefillLimits>) -> Result<Int8SemanticJobRequirements, Failure> {
    int8_semantic_job_requirements(command.semantic.host.task_budget(limits), options,
        semantic_cli::judge_limits(options), &command.semantic.encoding_policy().map_err(semantic_cli::population_failure)?,
        envelope.encoding_work_limits, envelope.native, host, job, input, prefill,
        (limits.compute.threads > 1).then_some(limits.compute.threads)).map_err(host_failure)
}

fn prepare(facts: &ArtifactIdentity, command: &SemanticCommand, limits: Limits,
    extraction: ExtractionBatchArgs, options: Int8SemanticOptions,
    work: Int8SemanticExtractionBatchLimits, encoding_work_limits: Option<SafeBpeWorkLimits>)
    -> Result<Int8SemanticExtractionJobPlanner, Failure> {
    if extraction.budget != command.host.task_budget(limits) {
        return Err(Failure::usage("semantic_job_extraction_budget"));
    }
    let registry = pinned_controls::pinned().map_err(|_| semantic_cli::population_failure(CandidateError::Identity))?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or_else(|| semantic_cli::population_failure(CandidateError::Identity))?;
    let mut identity = candidate_identity(facts).map_err(semantic_cli::population_failure)?;
    identity.task_spec = "extract-v1".to_owned();
    let config = Int8SemanticExtractionJobConfig {
        extraction, compiler_limits: extract::compiler(&command.host),
        source_limits: command.host.planning().source, judge_planning: semantic_cli::judge_limits(&options),
        options, encoding: command.encoding_policy().map_err(semantic_cli::population_failure)?,
        encoding_work_limits, work };
    Int8SemanticExtractionJobPlanner::pinned(controls, eos, identity, config).map_err(super::corpus::batch_failure)
}

fn check_factory(command: &JobCommand, limits: Limits, options: &Int8SemanticOptions,
    schema_digest: Sha256Digest, envelope: JobEnvelope, host: JobHostLimits, job: JobLimits,
    input: JobInputMemory, prefill: Option<Int8PrefillLimits>,
    factory: &Int8SemanticExtractionJobPlanner) -> Result<Int8SemanticJobRequirements, Failure> {
    factory.check_job_limits(job).map_err(super::corpus::batch_failure)?;
    let config = factory.config();
    let selected = command.semantic.encoding_policy().map_err(semantic_cli::population_failure)?;
    let expected_workspace = match &selected {
        UntrustedEncoding::ByteFallback => 0,
        UntrustedEncoding::SafeBpeV1(call) => call.requirements()
            .map_err(|_| Failure::usage("semantic_job_encoding_quote"))?.workspace_bytes
            .checked_add(1024).ok_or_else(|| Failure::usage("semantic_job_encoding_quote"))?,
    };
    let actual = config.work; let expected = envelope.native;
    let compiler = config.compiler_limits; let declared_compiler = extract::compiler(&command.semantic.host);
    let planning = config.judge_planning; let declared_planning = semantic_cli::judge_limits(options);
    let declared_source = command.semantic.host.planning().source;
    let grounding = if command.semantic.source_membership { ExtractionBatchGrounding::SourceMembership }
        else { ExtractionBatchGrounding::Structural };
    if config.extraction.budget != command.semantic.host.task_budget(limits)
        || Sha256Digest::of_bytes(config.extraction.schema.as_bytes()) != schema_digest
        || config.extraction.grounding != grounding
        || compiler.max_schema_bytes != declared_compiler.max_schema_bytes
        || compiler.max_string_bytes != declared_compiler.max_string_bytes
        || compiler.max_array_items != declared_compiler.max_array_items
        || compiler.max_output_bytes != declared_compiler.max_output_bytes
        || compiler.max_states != declared_compiler.max_states
        || compiler.max_transitions != declared_compiler.max_transitions
        || compiler.max_mask_bytes != declared_compiler.max_mask_bytes
        || planning.per_head.max_candidates != declared_planning.per_head.max_candidates
        || planning.per_head.max_total_tokens != declared_planning.per_head.max_total_tokens
        || planning.per_head.max_nodes != declared_planning.per_head.max_nodes
        || planning.per_head.max_depth != declared_planning.per_head.max_depth
        || planning.per_head.max_candidate_id_bytes != declared_planning.per_head.max_candidate_id_bytes
        || planning.per_head.max_projected_logits != declared_planning.per_head.max_projected_logits
        || planning.max_total_prompt_tokens != declared_planning.max_total_prompt_tokens
        || planning.max_total_candidate_tokens != declared_planning.max_total_candidate_tokens
        || planning.max_total_projected_logits != declared_planning.max_total_projected_logits
        || planning.max_output_bytes != declared_planning.max_output_bytes
        || canonjson::canonical_bytes(&config.source_limits).map_err(|_| Failure::usage("semantic_job_source_binding"))?
            != canonjson::canonical_bytes(&declared_source).map_err(|_| Failure::usage("semantic_job_source_binding"))?
        || factory.document_encoding() != &selected || &config.encoding != &selected
        || factory.encoding_work_limits() != envelope.encoding_work_limits
        || config.encoding_work_limits != envelope.encoding_work_limits
        || factory.encoding_workspace_bytes().map_err(super::corpus::batch_failure)? != expected_workspace
        || factory.max_result_bytes() != options.limits.max_output_bytes
        || actual.max_model_work != expected.max_model_work
        || actual.masks.max_visits_per_item != expected.masks.max_visits_per_item
        || actual.masks.max_visits_per_run != expected.masks.max_visits_per_run
        || actual.masks.per_mask.max_trie_node_visits != expected.masks.per_mask.max_trie_node_visits
        || actual.masks.per_mask.checkpoint_interval_nodes != expected.masks.per_mask.checkpoint_interval_nodes
        || canonjson::canonical_bytes(&config.options).map_err(|_| Failure::usage("semantic_job_options_binding"))?
            != canonjson::canonical_bytes(options).map_err(|_| Failure::usage("semantic_job_options_binding"))? {
        return Err(Failure::usage("semantic_job_factory_mismatch"));
    }
    int8_semantic_job_requirements(config.extraction.budget, &config.options, config.judge_planning,
        &config.encoding, config.encoding_work_limits, config.work, host, job, input, prefill,
        (limits.compute.threads > 1).then_some(limits.compute.threads)).map_err(host_failure)
}

fn minimum_memory(args: &CandidateArgs, limits: Limits, quote: Int8SemanticJobRequirements,
    prefill: Option<Int8PrefillLimits>) -> Result<u64, CandidateError> {
    // Population buffers and private recipe staging are already part of the
    // job quote. Reuse the same whole-process ownership floor without adding
    // or removing any individual host reservation.
    super::corpus::minimum_memory(args, limits, Int8SemanticCorpusRequirements {
        native: quote.native, native_workspace_bytes: quote.native_workspace_bytes,
        preparation_bytes: quote.preparation_bytes, input_bytes: quote.input_bytes,
        output_bytes: quote.output_bytes, team_workspace_bytes: quote.team_workspace_bytes,
    }, prefill)
}
fn admit_memory(args: &CandidateArgs, limits: Limits, quote: Int8SemanticJobRequirements,
    prefill: Option<Int8PrefillLimits>) -> Result<(), CandidateError> {
    if minimum_memory(args, limits, quote, prefill)? > limits.memory_bytes { return Err(CandidateError::Memory); }
    Ok(())
}
fn job_failure(error: JobError) -> Failure {
    match error {
        JobError::Cancelled(cause) => super::corpus::cancelled(cause),
        error => storage_failure(super::super::owned_jobs::job_failure(error)),
    }
}
fn host_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Host(error) => super::corpus::host_failure(error),
        HostedJobError::Job(JobRunError::Storage(error)) => job_failure(error),
        HostedJobError::Job(JobRunError::Processor(fault)) => super::corpus::batch_failure(fault),
        HostedJobError::Job(JobRunError::InvalidEnvelope) => Failure {
            exit: ErrorCode::InputDecodeOrParse, code: "semantic_job_envelope", cancellation: None },
        HostedJobError::Job(JobRunError::WorkContract) => Failure::output("semantic_job_work_contract"),
        HostedJobError::Job(JobRunError::Stopped) => Failure::output("semantic_job_missing_cancellation"),
    }
}
fn publish_report<T: Serialize>(response: &T, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>, cancellation: &CancellationToken)
    -> Result<(), Failure> {
    publish_checked(response, job_cli::REPORT_BYTES, output, &|| {
        if cancellation.cause().is_some() { return Err(CandidateError::Execution); }
        remaining().map(|_| ())
    }).map_err(|error| match (error, cancellation.cause()) {
        (CandidateError::Execution, Some(cause)) => super::corpus::cancelled(cause),
        (error, _) => semantic_cli::population_failure(error),
    })
}

#[derive(Serialize)]
struct Metadata<'a, T: Serialize> {
    #[serde(flatten)] response: &'a T,
    protocol: &'static str, execution: &'static str,
    same_model_correlated: bool, calibration: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")] compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")] prefill_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")] input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")] invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
}
#[cfg(test)] mod tests;
