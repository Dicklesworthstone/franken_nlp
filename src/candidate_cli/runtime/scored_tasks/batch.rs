//! Finite-score corpora on the existing hosted model and owned IO lifetime.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc};
use crate::{
    batch::classify::quantized::Int8ClassificationBatchPlanner,
    candidate_cli::{scored::Kind, scored_batch::{ScoreBatchCommand, ScoreEnvelope, Defaults, DEFAULTS_BYTES},
        batch::{CandidateWriter, IO_BUFFER_BYTES, completed}},
    hosted::corpus::{CorpusLimits, ClassificationCorpusConfig, SentimentCorpusConfig},
};

enum Corpus {
    Classify(Arc<ClassificationPlanner>, ClassificationCorpusConfig),
    Sentiment(Arc<SentimentPlanner>, SentimentCorpusConfig),
}
fn configure(command: &ScoreBatchCommand, facts: &ArtifactIdentity, limits: Limits, defaults: Defaults)
    -> Result<Corpus, CandidateError> {
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let ceiling = command.host.budget(limits);
    let work = command.host.work_ceiling();
    match (command.kind()?, defaults) {
        (Kind::Classify, Defaults::Classify(defaults)) => {
            let planner = Arc::new(ClassificationPlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "classify-v1".to_owned();
            identity.template_digest = *planner.template_digest(); identity.tokenizer_digest = planner.tokenizer_digest();
            let planning = command.host.classification_limits();
            // Fixed configuration checks only: no invented document stands in
            // for future records. Actual prompt plans compile per item.
            Int8ClassificationBatchPlanner::new(&planner, identity.clone(), ceiling, planning, defaults.clone())
                .map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Classify(planner, ClassificationCorpusConfig {
                identity, task_ceiling: ceiling, planning, defaults, max_model_work: work,
            }))
        }
        (Kind::Sentiment, Defaults::Sentiment { args, policy }) => {
            let planner = Arc::new(SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos, policy,
            }).map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "sentiment-v1".to_owned();
            identity.template_digest = *planner.template_digest(); identity.tokenizer_digest = planner.tokenizer_digest();
            let config = SentimentCorpusConfig { identity, task_ceiling: ceiling,
                planning: command.host.sentiment_limits(), defaults: Some(args),
                max_item_work: work, max_model_work: work };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Sentiment(planner, config))
        }
        _ => Err(CandidateError::Arguments),
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    model_id: &'a str, source_revision: &'a str, source_root_sha256: &'a str,
    logical_model_sha256: &'a str, quant_recipe: &'a str, task: &'a str,
}
pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: ScoreBatchCommand, args: CandidateArgs, limits: Limits, envelope: ScoreEnvelope,
    mut input: R, output: W,
) -> Result<(), CandidateError> {
    // Session is declared first: preparation authority outlives every local
    // setting, planner, source metadata, input buffer and output staging value.
    let session = Session::new(&args, limits)?;
    let raw_defaults = command.defaults.as_ref()
        .map(|path| session.read(path, &mut input, DEFAULTS_BYTES)).transpose()?;
    let defaults = command.parse_defaults(raw_defaults.as_deref(), command.host.budget(limits))?;
    let facts = session.facts(&args)?;
    let corpus = configure(&command, &facts, limits, defaults)?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision, source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id, task: &command.task };
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
    let summary = match corpus {
        Corpus::Classify(planner, config) => session.engine.batch_int8_classify(&model, planner, config,
            corpus_limits, reader, writer, cancellation),
        Corpus::Sentiment(planner, config) => session.engine.batch_int8_sentiment(&model, planner, config,
            corpus_limits, reader, writer, cancellation),
    }.map_err(|_| CandidateError::Batch)?;
    // Earlier complete records may exist. Never append a second terminal event,
    // retry a poisoned sink or conflate EOF with every document succeeding.
    completed(summary)
}

#[cfg(test)] mod tests;
