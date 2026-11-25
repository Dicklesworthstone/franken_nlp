//! Owned INT8 document populations with one resident model and whole-run limits.
use super::*;
use super::source_tasks::Session;
use std::{io::{BufRead, BufReader}, sync::Arc};
use crate::{
    candidate_cli::{
        scored_documents::{Admitted, DocumentCorpusCommand, Mapping},
        scored_batch::{Defaults, DEFAULTS_BYTES},
        batch::{CandidateWriter, IO_BUFFER_BYTES, completed},
    },
    hosted::{HostedError, corpus::{
        CorpusLimits, ClassificationDocumentCorpusConfig, SentimentDocumentCorpusConfig,
        int8_classification_document_corpus_requirements, int8_sentiment_document_corpus_requirements,
    }},
    tasks::{
        classify::ClassificationPlanner,
        sentiment::{SentimentPlanner, SentimentOptions},
    },
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

fn corpus_limits(admitted: &Admitted, elapsed: Duration) -> CorpusLimits {
    CorpusLimits {
        native: NativeLimits { context_tokens: admitted.common.context_tokens,
            allocator_reserve_bytes: 64 * MIB, run: RunLimits {
                max_elapsed: elapsed, max_checkpoints: admitted.common.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } },
        transport: admitted.transport,
        preparation_reserve_bytes: admitted.limits.preparation_bytes,
        io_reserve_bytes: admitted.io_bytes,
    }
}

fn sum(values: &[u64]) -> Result<u64, CandidateError> {
    values.iter().try_fold(0_u64, |total, &value|
        total.checked_add(value).ok_or(CandidateError::Arguments))
}

/// Pure complete-population capacity admission runs before defaults, corpus or
/// model IO. The transferred map frontier and transport coexist with CLI setup;
/// packed prompts and persistent worker payloads are separately charged.
pub(in crate::candidate_cli) fn admit(admitted: &Admitted) -> Result<u64, CandidateError> {
    if admitted.limits.compute != admitted.common.compute.admit_prefill(admitted.prefill)? {
        return Err(CandidateError::Arguments);
    }
    let corpus = corpus_limits(admitted, Duration::from_secs(admitted.common.timeout_seconds));
    let (native, input_bytes, workspace_bytes, output_bytes) = match &admitted.mapping {
        Mapping::Classify { planning, mapping } => {
            let need = int8_classification_document_corpus_requirements(admitted.task, *planning,
                *mapping, &admitted.encoding, corpus).map_err(|_| CandidateError::Arguments)?;
            (need.native, need.input_bytes, need.workspace_bytes, need.output_reservation_bytes)
        }
        Mapping::Sentiment { planning, mapping } => {
            let need = int8_sentiment_document_corpus_requirements(admitted.task, *planning,
                *mapping, &admitted.encoding, corpus).map_err(|_| CandidateError::Arguments)?;
            (need.native, need.input_bytes, need.workspace_bytes, need.output_reservation_bytes)
        }
    };
    let prefill_bytes = admitted.prefill.map(|prefill| prefill.validate()).transpose()
        .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
    let loading = sum(&[admitted.limits.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[admitted.limits.weight_bytes, 192 * MIB, input_bytes,
        native.kv_bytes, workspace_bytes, output_bytes, prefill_bytes,
        admitted.limits.compute.workspace_bytes, 65_536])?;
    let peak = sum(&[admitted.limits.preparation_bytes, loading.max(execution)])?;
    if peak > admitted.limits.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(peak)
}

enum Corpus {
    Classify(Arc<ClassificationPlanner>, ClassificationDocumentCorpusConfig),
    Sentiment(Arc<SentimentPlanner>, SentimentDocumentCorpusConfig),
}

fn configure(admitted: &Admitted, facts: &ArtifactIdentity, defaults: Defaults)
    -> Result<Corpus, CandidateError> {
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    match (&admitted.mapping, defaults) {
        (Mapping::Classify { planning, mapping }, Defaults::Classify(defaults)) => {
            let planner = Arc::new(ClassificationPlanner::pinned(controls, eos)
                .and_then(|planner| planner.with_document_encoding(admitted.encoding.clone()))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "classify-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let config = ClassificationDocumentCorpusConfig {
                identity, task_ceiling: admitted.task, planning: *planning, mapping: *mapping,
                defaults, max_model_work: mapping.max_model_work,
                max_tokenizer_work: admitted.max_tokenizer_work,
            };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Classify(planner, config))
        }
        (Mapping::Sentiment { planning, mapping }, Defaults::Sentiment { args, policy }) => {
            let planner = Arc::new(SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos, policy,
            }).and_then(|planner| planner.with_document_encoding(admitted.encoding.clone()))
                .map_err(|_| CandidateError::Planning)?);
            identity.task_spec = "sentiment-v1".to_owned();
            identity.template_digest = *planner.template_digest();
            identity.tokenizer_digest = planner.tokenizer_digest();
            let config = SentimentDocumentCorpusConfig {
                identity, task_ceiling: admitted.task, planning: *planning, mapping: *mapping,
                defaults: Some(args), max_model_work: mapping.max_model_work,
                max_tokenizer_work: admitted.max_tokenizer_work,
            };
            config.validate(&planner).map_err(|_| CandidateError::Planning)?;
            Ok(Corpus::Sentiment(planner, config))
        }
        _ => Err(CandidateError::Arguments),
    }
}

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str,
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    model_id: &'a str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    logical_model_sha256: &'a str,
    quant_recipe: &'a str,
    task: &'static str,
    complete_document: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: DocumentCorpusCommand, admitted: Admitted, mut input: R, output: W,
) -> Result<(), CandidateError> {
    admit(&admitted)?;
    let session = Session::new_with_prefill(&admitted.common, admitted.limits, admitted.prefill)?;
    let raw_defaults = command.transport.defaults.as_ref()
        .map(|path| session.read(path, &mut input, DEFAULTS_BYTES)).transpose()?;
    let defaults = command.parse_defaults(raw_defaults.as_deref(), admitted.task)?;
    drop(raw_defaults);
    let facts = session.facts(&admitted.common)?;
    let corpus = configure(&admitted, &facts, defaults)?;
    session.remaining()?;
    let provenance = Provenance {
        protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id,
        task: admitted.mapping.task_name(), complete_document: true,
        input_encoding: input_encoding::reported_policy(&admitted.encoding),
        invocation_encoding_work_limits: admitted.max_tokenizer_work,
        compute_threads: (admitted.limits.compute.threads > 1).then_some(admitted.limits.compute.threads),
        prefill_rows: admitted.prefill.map(|prefill| prefill.max_batch_rows),
    };
    let writer = CandidateWriter::new(output, &provenance,
        admitted.transport.max_output_line_bytes, admitted.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if admitted.common.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            File::open(&admitted.common.input).map_err(|_| CandidateError::Input)?))
    };
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&admitted.common, admitted.limits, &facts, cancellation.clone())?;
    let limits = corpus_limits(&admitted, session.remaining()?);
    let width = admitted.limits.compute.threads;
    let summary = match corpus {
        Corpus::Classify(planner, config) => if width == 1 {
            match admitted.prefill {
                None => session.engine.batch_int8_classification_documents(&model, planner, config,
                    limits, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_classification_documents_layer_major(&model,
                    planner, config, limits, prefill, reader, writer, cancellation),
            }
        } else {
            let kernel = crate::native_engine::simd::config::process_kernel();
            match admitted.prefill {
                None => session.engine.batch_int8_classification_documents_with_threads(&model,
                    planner, config, limits, kernel, width, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_classification_documents_layer_major_with_threads(
                    &model, planner, config, limits, kernel, width, prefill, reader, writer, cancellation),
            }
        },
        Corpus::Sentiment(planner, config) => if width == 1 {
            match admitted.prefill {
                None => session.engine.batch_int8_sentiment_documents(&model, planner, config,
                    limits, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_sentiment_documents_layer_major(&model,
                    planner, config, limits, prefill, reader, writer, cancellation),
            }
        } else {
            let kernel = crate::native_engine::simd::config::process_kernel();
            match admitted.prefill {
                None => session.engine.batch_int8_sentiment_documents_with_threads(&model,
                    planner, config, limits, kernel, width, reader, writer, cancellation),
                Some(prefill) => session.engine.batch_int8_sentiment_documents_layer_major_with_threads(
                    &model, planner, config, limits, kernel, width, prefill, reader, writer, cancellation),
            }
        },
    }.map_err(host_error)?;
    // The hosted path owns run_complete, including post-join publication.
    // Failed records remain a nonzero process result without a second footer.
    completed(summary)
}

fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn host_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Batch,
    }
}

#[cfg(test)]
mod tests;
