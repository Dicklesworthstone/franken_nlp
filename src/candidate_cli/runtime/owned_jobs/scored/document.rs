//! Whole-document finite-scoring jobs use the existing authenticated lifecycle.
use super::*;
use crate::{
    candidate_cli::{jobs::scored::document::{self as settings, Mapping},
        scored_batch::Defaults, faithfulness_map::corpus as faithfulness},
    hosted::corpus::{ClassificationDocumentCorpusConfig, SentimentDocumentCorpusConfig,
        FaithfulnessDocumentCorpusConfig,
        jobs::{int8_classification_document_job_requirements,
            int8_sentiment_document_job_requirements, int8_faithfulness_document_job_requirements}},
    tasks::{classify::ClassificationPlanner, sentiment::{SentimentPlanner, SentimentOptions},
        judge::{FaithfulnessPolicy, JudgePlanner}},
    native_engine::lmhead::scoring::ScoringMode,
};

enum DocumentCorpus {
    Classify(Arc<ClassificationPlanner>, ClassificationDocumentCorpusConfig),
    Sentiment(Arc<SentimentPlanner>, SentimentDocumentCorpusConfig),
    Faithfulness(Arc<JudgePlanner>, FaithfulnessDocumentCorpusConfig),
}

fn configure(args: &ScoreJobArgs, facts: &ArtifactIdentity, limits: Limits,
    mapping: Mapping, defaults: Defaults, policy: Option<FaithfulnessPolicy>) -> Result<DocumentCorpus, Failure> {
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let encoding = args.encoding_policy()?;
    let task = args.task_budget(limits)?;
    let totals = args.encoding_work_limits()?;
    match (mapping, defaults, policy) {
        (Mapping::Classify { planning, mapping }, Defaults::Classify(defaults), None) => {
            let planner = Arc::new(ClassificationPlanner::pinned(controls, eos)
                .and_then(|planner| planner.with_document_encoding(encoding))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "classify-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let config = ClassificationDocumentCorpusConfig {
                identity, task_ceiling: task, planning, mapping, defaults,
                max_model_work: args.host.work_ceiling(), max_tokenizer_work: totals,
            };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(DocumentCorpus::Classify(planner, config))
        }
        (Mapping::Sentiment { planning, mapping }, Defaults::Sentiment { args: defaults, policy }, None) => {
            let planner = Arc::new(SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos, policy,
            }).and_then(|planner| planner.with_document_encoding(encoding))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "sentiment-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let config = SentimentDocumentCorpusConfig {
                identity, task_ceiling: task, planning, mapping, defaults: Some(defaults),
                max_model_work: args.host.work_ceiling(), max_tokenizer_work: totals,
            };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(DocumentCorpus::Sentiment(planner, config))
        }
        (Mapping::Faithfulness { planning, mapping }, Defaults::Judge(None), Some(policy)) => {
            let planner = Arc::new(JudgePlanner::pinned(controls, eos)
                .and_then(|planner| planner.with_document_encoding(encoding))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "judge-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let config = FaithfulnessDocumentCorpusConfig {
                identity, task_ceiling: task, planning, mapping, policy,
                max_model_work: args.host.work_ceiling(), max_tokenizer_work: totals,
            };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(DocumentCorpus::Faithfulness(planner, config))
        }
        _ => Err(Failure::usage("score_document_job_task_defaults")),
    }
}

fn sum(values: &[u64]) -> Result<u64, Failure> {
    values.iter().try_fold(0_u64, |total, &value| total.checked_add(value)
        .ok_or_else(|| Failure::usage("score_document_job_memory_arithmetic")))
}

/// Retained replay prices an index and one input record; original-input mode
/// prices the complete population. Both coexist with the whole-map frontier.
fn memory_bytes(args: &ScoreJobArgs, limits: Limits, mapping: &Mapping,
    host: JobHostLimits, lifetime: JobLimits) -> Result<u64, Failure> {
    let task = args.task_budget(limits)?;
    let encoding = args.encoding_policy()?;
    let input = args.input_retention.input_memory()?;
    let (native, input_bytes, workspace_bytes, output_bytes) = match mapping {
        Mapping::Classify { planning, mapping } => {
            let need = int8_classification_document_job_requirements(task, *planning, *mapping,
                &encoding, host, lifetime, input).map_err(host_failure)?;
            (need.native, need.input_bytes, need.workspace_bytes, need.output_reservation_bytes)
        }
        Mapping::Sentiment { planning, mapping } => {
            let need = int8_sentiment_document_job_requirements(task, *planning, *mapping,
                &encoding, host, lifetime, input).map_err(host_failure)?;
            (need.native, need.input_bytes, need.workspace_bytes, need.output_reservation_bytes)
        }
        Mapping::Faithfulness { planning, mapping } => {
            let need = int8_faithfulness_document_job_requirements(task, *planning, *mapping,
                &encoding, host, lifetime, input).map_err(host_failure)?;
            (need.native, need.input_bytes, need.workspace_bytes, need.output_reservation_bytes)
        }
    };
    let prefill = args.prefill_limits()?;
    if limits.compute != args.compute.admit_prefill(prefill)? {
        return Err(Failure::usage("score_job_schedule_admission"));
    }
    let extra = prefill.map(|prefill| prefill.validate()).transpose()
        .map_err(|_| Failure::usage("score_job_prefill_quote"))?.unwrap_or(0);
    let loading = sum(&[limits.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[limits.weight_bytes, 192 * MIB, input_bytes,
        native.kv_bytes, workspace_bytes, output_bytes, extra,
        limits.compute.workspace_bytes, 65_536])?;
    sum(&[limits.preparation_bytes, loading.max(execution)])
}

fn admit(args: &ScoreJobArgs, limits: Limits, mapping: &Mapping,
    host: JobHostLimits, lifetime: JobLimits) -> Result<u64, Failure> {
    let bytes = memory_bytes(args, limits, mapping, host, lifetime)?;
    if bytes > limits.memory_bytes {
        return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
            code: "score_document_job_population_memory" });
    }
    Ok(bytes)
}

#[derive(Serialize)]
struct DocumentReport<'a, T: Serialize> {
    #[serde(flatten)] response: &'a T,
    complete_document: bool,
}

pub(super) fn execute<R: Read + Send + 'static>(mode: RunMode, args: ScoreJobArgs,
    common: CandidateArgs, limits: Limits, input: R, output: &mut impl Write) -> Result<(), Failure> {
    settings::validate_mode(&args)?;
    args.input_retention.validate(mode, &args.host.input)?;
    let prefill = args.prefill_limits()?;
    let encoding = args.encoding_policy()?;
    let totals = args.encoding_work_limits()?;
    if common.input_encoding != args.encoding || common.compute != args.compute {
        return Err(Failure::usage("score_document_job_admission"));
    }
    // This first local charge owns config/default parsing and outlives every
    // transferred planner, reader and final metadata serialization value.
    let session = Session::new_with_prefill(&common, limits, prefill)?;
    let lifetime = command::parse_limits(&read_config(&session, &args.limits_file, command::LIMIT_BYTES)?)?;
    let path = args.document_config.as_ref()
        .ok_or_else(|| Failure::usage("score_document_job_configuration"))?;
    let mapping = settings::load(&read_config(&session, path, settings::CONFIG_BYTES)?,
        &args, limits, lifetime)?;
    let host = host_limits(&args, &common, limits, &session)?;
    admit(&args, limits, &mapping, host, lifetime)?;
    // Complete map, population, output, team and prompt-workspace admission
    // precedes secret, task defaults, source population and weight IO.
    let key = JobSecret::read(&args.key_file).map_err(job_failure)?;
    session.remaining()?;
    let raw_defaults = args.defaults.as_ref()
        .map(|path| read_config(&session, path, command::DEFAULT_BYTES)).transpose()?;
    let defaults = args.parse_defaults(raw_defaults.as_deref(), args.task_budget(limits)?)?;
    drop(raw_defaults);
    let policy = args.faithfulness_policy.as_ref().map(|path| {
        let text = read_config(&session, path, faithfulness::POLICY_BYTES)?;
        faithfulness::parse_policy(&text).map_err(Failure::from)
    }).transpose()?;
    let facts = session.facts(&common)?;
    let corpus = configure(&args, &facts, limits, mapping, defaults, policy)?;
    session.remaining()?;
    let reader = command::population_reader(mode, args.input_retention, &args.host.input, input, IO_BYTES)?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let host = host_limits(&args, &common, limits, &session)?;
    let job_id = args.job_id;
    let materialize = args.materialize;
    let request = SourceJobRequest { root: args.job_dir, key, job_id,
        limits: lifetime, mode: open_mode(mode, args.input_retention)?, materialize };
    let team = (limits.compute.threads > 1)
        .then(|| (crate::native_engine::simd::config::process_kernel(), limits.compute.threads));
    let progress = match corpus {
        DocumentCorpus::Classify(planner, config) => session.engine
            .job_int8_classification_documents_scheduled(&model, planner, config, request, host,
                prefill, team, reader, cancellation.clone()),
        DocumentCorpus::Sentiment(planner, config) => session.engine
            .job_int8_sentiment_documents_scheduled(&model, planner, config, request, host,
                prefill, team, reader, cancellation.clone()),
        DocumentCorpus::Faithfulness(planner, config) => session.engine
            .job_int8_faithfulness_documents_scheduled(&model, planner, config, request, host,
                prefill, team, reader, cancellation.clone()),
    }.map_err(host_failure)?;
    if let Some(cause) = cancellation.cause() { return Err(job_failure(JobError::Cancelled(cause))); }
    session.remaining()?;
    let report = progress_report(progress, job_id, lifetime, materialize, mode, &args.task)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &report };
    let response = EncodingReport { response: &response,
        input_encoding: input_encoding::reported_policy(&encoding),
        invocation_encoding_work_limits: totals, root_object_reconciliation: None };
    let response = ScheduledResponse { response: &response,
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|limits| limits.max_batch_rows) };
    let response = DocumentReport { response: &response, complete_document: true };
    // No private labels, claims, source text or chunk scores enter stdout. Publication
    // follows physical worker join and preserves terminal cancellation causes.
    publish_report(&response, output, &|| session.remaining(), &cancellation)
}

#[cfg(test)] mod tests;
