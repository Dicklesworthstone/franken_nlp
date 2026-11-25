//! Finite-score corpora on the existing hosted model and owned IO lifetime.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc};
use crate::tokenizer::untrusted::SafeBpeWorkLimits;
use crate::{
    batch::classify::quantized::Int8ClassificationBatchPlanner,
    candidate_cli::{scored_batch::{ScoreTask, ScoreBatchCommand, ScoreEnvelope, Defaults, DEFAULTS_BYTES},
        batch::{CandidateWriter, IO_BUFFER_BYTES, completed}},
    tasks::judge::JudgePlanner,
    hosted::corpus::{CorpusLimits, ClassificationCorpusConfig, SentimentCorpusConfig, JudgeCorpusConfig},
};

pub(in crate::candidate_cli) enum Corpus {
    Classify(Arc<ClassificationPlanner>, ClassificationCorpusConfig),
    Sentiment(Arc<SentimentPlanner>, SentimentCorpusConfig),
    Judge(Arc<JudgePlanner>, JudgeCorpusConfig),
}
fn configure(command: &ScoreBatchCommand, facts: &ArtifactIdentity, limits: Limits, defaults: Defaults)
    -> Result<Corpus, CandidateError> {
    let encoding = command.encoding_policy()?;
    let totals = command.encoding_work_limits()?;
    let corpus = configure_scored_with_encoding(command.kind()?, &command.host, facts, limits, defaults, &encoding)?;
    if let Corpus::Judge(planner, config) = &corpus {
        let actual = config.encoding_requirements(planner, totals).map_err(|_| CandidateError::Planning)?;
        let expected = (encoding, totals, command.encoding_workspace_bytes()?);
        if actual != expected { return Err(CandidateError::Arguments); }
    }
    Ok(corpus)
}
/// Identical pinned compiler and score-space selection for streams and jobs.
pub(in crate::candidate_cli) fn configure_scored(kind: ScoreTask, host: &ScoredArgs,
    facts: &ArtifactIdentity, limits: Limits, defaults: Defaults) -> Result<Corpus, CandidateError> {
    configure_scored_with_encoding(kind, host, facts, limits, defaults, &UntrustedEncoding::ByteFallback)
}
pub(in crate::candidate_cli) fn configure_scored_with_encoding(kind: ScoreTask, host: &ScoredArgs,
    facts: &ArtifactIdentity, limits: Limits, defaults: Defaults, encoding: &UntrustedEncoding)
    -> Result<Corpus, CandidateError> {
    encoding.validate().map_err(|_| CandidateError::Arguments)?;
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let ceiling = host.budget(limits);
    let work = host.work_ceiling();
    match (kind, defaults) {
        (ScoreTask::Classify, Defaults::Classify(defaults)) => {
            let planner = Arc::new(ClassificationPlanner::pinned(controls, eos)
                .and_then(|planner| planner.with_document_encoding(encoding.clone()))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "classify-v1".to_owned();
            identity.template_digest = *planner.template_digest(); identity.tokenizer_digest = planner.tokenizer_digest();
            let planning = host.classification_limits();
            // Fixed configuration checks only: no invented document stands in
            // for future records. Actual prompt plans compile per item.
            Int8ClassificationBatchPlanner::new_with_document_encoding(&planner, identity.clone(), ceiling, planning, defaults.clone())
                .map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Classify(planner, ClassificationCorpusConfig {
                identity, task_ceiling: ceiling, planning, defaults, max_model_work: work,
            }))
        }
        (ScoreTask::Sentiment, Defaults::Sentiment { args, policy }) => {
            let planner = Arc::new(SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos, policy,
            }).and_then(|planner| planner.with_document_encoding(encoding.clone()))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "sentiment-v1".to_owned();
            identity.template_digest = *planner.template_digest(); identity.tokenizer_digest = planner.tokenizer_digest();
            let config = SentimentCorpusConfig { identity, task_ceiling: ceiling,
                planning: host.sentiment_limits(), defaults: Some(args),
                max_item_work: work, max_model_work: work };
            config.validate_with_document_encoding(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Sentiment(planner, config))
        }
        (ScoreTask::Judge, Defaults::Judge(defaults)) => {
            let planner = Arc::new(JudgePlanner::pinned(controls, eos)
                .and_then(|planner| planner.with_document_encoding(encoding.clone()))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "judge-v1".to_owned();
            identity.template_digest = *planner.template_digest(); identity.tokenizer_digest = planner.tokenizer_digest();
            let config = JudgeCorpusConfig { identity, task_ceiling: ceiling,
                planning: crate::candidate_cli::judge::planning_limits(host, ceiling), defaults, max_model_work: work };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Judge(planner, config))
        }
        _ => Err(CandidateError::Arguments),
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    model_id: &'a str, source_revision: &'a str, source_root_sha256: &'a str,
    logical_model_sha256: &'a str, quant_recipe: &'a str, task: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding_work_limits: Option<SafeBpeWorkLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
}
pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: ScoreBatchCommand, args: CandidateArgs, limits: Limits, envelope: ScoreEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    // Session is the first owning local: preparation authority outlives every
    // setting, planner, source metadata, input buffer and output staging value.
    let prefill = command.prefill_limits()?;
    command.check_scheduling_memory(limits, envelope, prefill)?;
    if args.compute != command.compute { return Err(CandidateError::Arguments); }
    let encoding = command.encoding_policy()?;
    let encoding_work_limits = command.encoding_work_limits()?;
    let session = Session::new_with_prefill(&args, limits, prefill)?;
    let raw_defaults = command.defaults.as_ref()
        .map(|path| session.read(path, &mut input, DEFAULTS_BYTES)).transpose()?;
    let defaults = command.parse_defaults(raw_defaults.as_deref(), command.host.budget(limits))?;
    let facts = session.facts(&args)?;
    let corpus = configure(&command, &facts, limits, defaults)?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id, task: &command.task,
        input_encoding: input_encoding::reported_policy(&encoding),
        input_encoding_work_limits: encoding_work_limits,
        compute_threads: (limits.compute.threads > 1).then_some(limits.compute.threads),
        prefill_rows: prefill.map(|limits| limits.max_batch_rows) };
    let writer = CandidateWriter::new(output, &provenance, envelope.transport.max_output_line_bytes, envelope.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, File::open(&args.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let corpus_limits = CorpusLimits { native: session.native(&args)?, transport: envelope.transport,
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: envelope.io_bytes };
    let width = limits.compute.threads;
    let summary = match corpus {
        Corpus::Judge(planner, config) if encoding != UntrustedEncoding::ByteFallback => {
            let team = (width > 1).then(|| (crate::native_engine::simd::config::process_kernel(), width));
            session.engine.batch_int8_judge_scheduled(&model, planner, config, corpus_limits,
                encoding_work_limits, prefill, team, reader, writer, cancellation)
        }
        corpus => if width == 1 {
            match corpus {
                Corpus::Classify(planner, config) => match prefill {
                    Some(prefill) => session.engine.batch_int8_classify_layer_major(&model, planner, config,
                        corpus_limits, prefill, reader, writer, cancellation),
                    None => session.engine.batch_int8_classify(&model, planner, config,
                        corpus_limits, reader, writer, cancellation),
                },
                Corpus::Sentiment(planner, config) => match prefill {
                    Some(prefill) => session.engine.batch_int8_sentiment_layer_major(&model, planner, config,
                        corpus_limits, prefill, reader, writer, cancellation),
                    None => session.engine.batch_int8_sentiment(&model, planner, config,
                        corpus_limits, reader, writer, cancellation),
                },
                Corpus::Judge(planner, config) => match prefill {
                    Some(prefill) => session.engine.batch_int8_judge_layer_major(&model, planner, config,
                        corpus_limits, prefill, reader, writer, cancellation),
                    None => session.engine.batch_int8_judge(&model, planner, config,
                        corpus_limits, reader, writer, cancellation),
                },
            }
        } else {
            let kernel = crate::native_engine::simd::config::process_kernel();
            match corpus {
                Corpus::Classify(planner, config) => match prefill {
                    Some(prefill) => session.engine.batch_int8_classify_layer_major_with_threads(&model,
                        planner, config, corpus_limits, kernel, width, prefill, reader, writer, cancellation),
                    None => session.engine.batch_int8_classify_with_threads(&model,
                        planner, config, corpus_limits, kernel, width, reader, writer, cancellation),
                },
                Corpus::Sentiment(planner, config) => match prefill {
                    Some(prefill) => session.engine.batch_int8_sentiment_layer_major_with_threads(&model,
                        planner, config, corpus_limits, kernel, width, prefill, reader, writer, cancellation),
                    None => session.engine.batch_int8_sentiment_with_threads(&model,
                        planner, config, corpus_limits, kernel, width, reader, writer, cancellation),
                },
                Corpus::Judge(planner, config) => match prefill {
                    Some(prefill) => session.engine.batch_int8_judge_layer_major_with_threads(&model,
                        planner, config, corpus_limits, kernel, width, prefill, reader, writer, cancellation),
                    None => session.engine.batch_int8_judge_with_threads(&model,
                        planner, config, corpus_limits, kernel, width, reader, writer, cancellation),
                },
            }
        },
    }.map_err(|_| CandidateError::Batch)?;
    // Earlier complete records may exist. Never append a second terminal event,
    // retry a poisoned sink or conflate EOF with every document succeeding.
    completed(summary)
}

#[cfg(test)] mod tests;

#[cfg(test)] mod judge_tests;
