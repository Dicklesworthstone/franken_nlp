//! Immutable whole-document settings for retained INT8 finite-scoring jobs.
//! Native work remains bounded by both the invocation and the durable journal.
use super::*;
use crate::{
    candidate_cli::scored_documents::Mapping,
    jobs::JobWork,
    tasks::{
        classify::{ClassificationAggregationPolicy, quantized::long::Int8ClassificationMapLimits},
        sentiment::{SentimentAggregationPolicy, quantized::long::Int8SentimentMapLimits},
        mapreduce::ChunkLimits,
    },
};

pub(in crate::candidate_cli) const CONFIG_BYTES: usize = 16 * 1024;
const MAX_RESULT_BYTES: u64 = 64 * MIB;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    schema_version: u32,
    max_chunks: usize,
    max_chunk_bytes: usize,
    max_tokenizer_calls: usize,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    /// Classified coverage for classification, estimated coverage for sentiment.
    minimum_coverage_ppm: u32,
}

pub(in crate::candidate_cli) fn validate_mode(args: &ScoreJobArgs) -> Result<(), Failure> {
    if let Some(path) = &args.document_config {
        if path.as_os_str().is_empty() || path.as_os_str() == "-"
            || !matches!(args.kind()?, ScoreTask::Classify | ScoreTask::Sentiment) {
            return Err(Failure::usage("score_document_job_options"));
        }
    }
    Ok(())
}

pub(in crate::candidate_cli) fn load(text: &str, args: &ScoreJobArgs,
    limits: Limits, lifetime: JobLimits) -> Result<Mapping, Failure> {
    validate_mode(args)?;
    if args.document_config.is_none() || text.is_empty() || text.len() > CONFIG_BYTES {
        return Err(Failure::usage("score_document_job_configuration"));
    }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 4, max_string_bytes: 1024,
    }).map_err(|_| Failure::usage("score_document_job_configuration"))?;
    let settings: Settings = serde_json::from_value(value)
        .map_err(|_| Failure::usage("score_document_job_configuration"))?;
    lifetime.validate().map_err(|_| Failure::usage("job_limits"))?;
    let task = args.task_budget(limits)?;
    task.validate().map_err(|_| Failure::usage("score_document_job_budget"))?;
    if settings.schema_version != 1 || !(1..=256).contains(&settings.max_chunks)
        || !(4..=args.host.max_input_bytes).contains(&settings.max_chunk_bytes)
        || !(1..=1_000_000).contains(&settings.max_tokenizer_calls)
        || !(1..=MAX_RESULT_BYTES).contains(&settings.max_result_bytes)
        || !(1..=MAX_RESULT_BYTES).contains(&settings.max_retained_result_bytes)
        || settings.max_result_bytes < task.max_output_bytes
        || settings.max_retained_result_bytes < task.max_output_bytes
        || settings.max_result_bytes > lifetime.max_result_bytes as u64
        || settings.minimum_coverage_ppm > 1_000_000 {
        return Err(Failure::usage("score_document_job_limits"));
    }
    let work = args.host.work_ceiling();
    if !(JobWork { model: work, mask_node_visits: 0 }).fits(lifetime.max_work) {
        return Err(Failure::usage("score_document_job_lifetime_work"));
    }
    let chunks = ChunkLimits {
        max_input_bytes: args.host.max_input_bytes,
        max_chunk_bytes: settings.max_chunk_bytes,
        max_chunk_tokens: task.max_input_tokens as usize,
        context_tokens: args.host.context_tokens,
        reserved_tokens: task.max_output_tokens as usize,
        max_chunks: settings.max_chunks,
        max_tokenizer_calls: settings.max_tokenizer_calls,
    };
    chunks.effective_token_limit().map_err(|_| Failure::usage("score_document_job_context"))?;
    // Never multiply a ceiling by the chunk, input-line or attempt count.
    // The sealed complete plan supplies the actual persisted attempt debit.
    let mapping = match args.kind()? {
        ScoreTask::Classify => {
            let mapping = Int8ClassificationMapLimits {
                chunks, max_model_work: work,
                max_retained_result_bytes: settings.max_retained_result_bytes,
                max_result_bytes: settings.max_result_bytes,
                aggregation: ClassificationAggregationPolicy {
                    minimum_classified_coverage_ppm: settings.minimum_coverage_ppm },
            };
            mapping.validate(task).map_err(|_| Failure::usage("score_document_job_limits"))?;
            Mapping::Classify { planning: args.host.classification_limits(), mapping }
        }
        ScoreTask::Sentiment => {
            let mapping = Int8SentimentMapLimits {
                chunks, max_model_work: work,
                max_retained_result_bytes: settings.max_retained_result_bytes,
                max_result_bytes: settings.max_result_bytes,
                aggregation: SentimentAggregationPolicy {
                    minimum_estimated_coverage_ppm: settings.minimum_coverage_ppm },
            };
            mapping.validate(task).map_err(|_| Failure::usage("score_document_job_limits"))?;
            Mapping::Sentiment { planning: args.host.sentiment_limits(), mapping }
        }
        ScoreTask::Judge => return Err(Failure::usage("score_document_job_options")),
    };
    // The CLI owns strict config/default parsing beside the host's separate
    // complete-map frontier. This is a reservation estimate, not an RSS cap.
    let floor = settings.max_result_bytes.checked_mul(2)
        .and_then(|bytes| bytes.checked_add(256 * MIB))
        .and_then(|bytes| bytes.checked_add((args.host.max_input_bytes as u64).checked_mul(32)?))
        .and_then(|bytes| bytes.checked_add(args.encoding_workspace_bytes().ok()?))
        .ok_or_else(|| Failure::usage("score_document_job_preparation"))?;
    if floor > limits.preparation_bytes {
        return Err(Failure::usage("score_document_job_preparation"));
    }
    Ok(mapping)
}

#[cfg(test)] pub(in crate::candidate_cli) mod tests;
