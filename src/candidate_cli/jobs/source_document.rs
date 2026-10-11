//! Closed settings for durable NER/keyphrase/cited-summary document maps.
//! Lifetime JobLimits remains authority; chunk count never multiplies it.
use super::*;
use crate::{
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::Int8SourceMapLimits},
    batch::source::{SourceMaskBudget, quantized::{Int8SourceBatchLimits, validate_limits}},
};

pub(in crate::candidate_cli) const CONFIG_BYTES: usize = 16 * 1024;
const MAX_RESULT: usize = 64 * 1024 * 1024;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct SourceDocumentSettings {
    schema_version: u32,
    max_chunks: usize,
    max_chunk_bytes: usize,
    max_tokenizer_calls: usize,
    max_result_bytes: usize,
    max_live_value_bytes: usize,
    max_total_value_bytes: usize,
    reduction_reserve_mib: u64,
}
fn parse(text: &str) -> Result<SourceDocumentSettings, Failure> {
    if text.is_empty() || text.len() > CONFIG_BYTES { return Err(Failure::usage("source_document_job_configuration")); }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits { max_depth: 4, max_string_bytes: 4096 })
        .map_err(|_| Failure::usage("source_document_job_configuration"))?;
    serde_json::from_value(value).map_err(|_| Failure::usage("source_document_job_configuration"))
}
pub(in crate::candidate_cli) fn validate_mode(args: &JobArgs) -> Result<(), Failure> {
    if let Some(path) = &args.source_document_config {
        if path.as_os_str().is_empty() || path.as_os_str() == "-"
            || !matches!(args.task.as_str(), "ner" | "keyphrases" | "summarize")
            || args.document_config.is_some() || args.schema.is_some() || args.source_membership {
            return Err(Failure::usage("source_document_job_options"));
        }
    }
    Ok(())
}

pub(in crate::candidate_cli) fn load(text: &str, args: &JobArgs, budget: TaskBudget,
    lifetime: JobLimits, limits: Limits) -> Result<(Int8SourceMapLimits, u64), Failure> {
    validate_mode(args)?;
    if args.source_document_config.is_none() { return Err(Failure::usage("source_document_job_configuration_missing")); }
    lifetime.validate().map_err(|_| Failure::usage("job_limits"))?;
    budget.validate().map_err(|_| Failure::usage("source_document_job_budget"))?;
    let ceiling = args.host.task_budget(limits);
    if budget.max_input_tokens > ceiling.max_input_tokens || budget.max_output_tokens > ceiling.max_output_tokens
        || budget.max_output_bytes > ceiling.max_output_bytes || budget.max_grammar_states > ceiling.max_grammar_states
        || budget.max_kv_bytes != ceiling.max_kv_bytes {
        return Err(Failure::usage("source_document_job_budget"));
    }
    let d = parse(text)?; let host = &args.host;
    if d.schema_version != 1 || !(1..=256).contains(&d.max_chunks)
        || !(4..=MAX_INPUT_BYTES).contains(&d.max_chunk_bytes) || d.max_chunk_bytes > host.max_input_bytes
        || !(1..=1_000_000).contains(&d.max_tokenizer_calls)
        || [d.max_result_bytes, d.max_live_value_bytes, d.max_total_value_bytes].iter().any(|&n| !(1..=MAX_RESULT).contains(&n))
        || (d.max_result_bytes as u64) < budget.max_output_bytes || d.max_result_bytes > lifetime.max_result_bytes
        || d.max_result_bytes > d.max_live_value_bytes || d.max_result_bytes > d.max_total_value_bytes {
        return Err(Failure::usage("source_document_job_limits"));
    }
    let reserve = d.reduction_reserve_mib.checked_mul(MIB).filter(|&n| n != 0 && n <= limits.memory_bytes)
        .ok_or_else(|| Failure::usage("source_document_job_memory"))?;
    // Necessary preparation floor for the complete set of retained source
    // grammars/prompts and their identity snapshots, not one context's plan.
    // The host additionally charges maximum output/reduction staging and native
    // allocations. No arithmetic estimate is a measured RSS bound or permit.
    let per = (host.context_tokens as u64).checked_mul(32)
        .and_then(|n| n.checked_add(u64::from(host.max_grammar_states).checked_mul(512)?))
        .and_then(|n| n.checked_add(2 * DEFAULT_BYTES as u64))
        .ok_or_else(|| Failure::usage("source_document_job_memory"))?;
    let floor = per.checked_mul(d.max_chunks as u64).and_then(|n| n.checked_add(256 * MIB))
        .and_then(|n| n.checked_add((host.max_input_bytes as u64).checked_mul(64)?))
        .and_then(|n| n.checked_add((d.max_result_bytes as u64).checked_mul(2)?))
        .ok_or_else(|| Failure::usage("source_document_job_memory"))?;
    if floor > limits.preparation_bytes { return Err(Failure::usage("source_document_job_preparation")); }
    let masks = host.max_mask_node_visits.checked_mul(d.max_chunks as u64)
        .ok_or_else(|| Failure::usage("source_document_job_work"))?
        .min(lifetime.max_work.mask_node_visits);
    // Deliberately DO NOT scale native lifetime work by max_chunks, item count,
    // attempts or epochs. The real planner sums the actual full map, and the
    // journal durably charges that exact sum before execution. A later attempt
    // still needs its own debit from the unchanged lifetime allowance.
    validate_limits(Int8SourceBatchLimits { max_model_work: lifetime.max_work.model,
        masks: SourceMaskBudget { per_mask: host.masks(), max_visits_per_item: host.max_mask_node_visits,
            max_visits_per_run: masks } }).map_err(|_| Failure::usage("source_document_job_work"))?;
    let mapping = Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: host.max_input_bytes, max_chunk_bytes: d.max_chunk_bytes,
            max_chunk_tokens: host.context_tokens, context_tokens: host.context_tokens,
            reserved_tokens: budget.max_output_tokens as usize, max_chunks: d.max_chunks,
            max_tokenizer_calls: d.max_tokenizer_calls },
        reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8, max_reduction_levels: 8,
            max_task_calls: 1024, max_value_bytes: d.max_result_bytes, max_live_value_bytes: d.max_live_value_bytes,
            max_total_value_bytes: d.max_total_value_bytes, max_result_bytes: d.max_result_bytes },
        max_model_work: lifetime.max_work.model, mask_limits: host.masks(),
        mask_visits_per_chunk: host.max_mask_node_visits, max_mask_visits: masks,
    };
    mapping.chunks.effective_token_limit().map_err(|_| Failure::usage("source_document_job_context"))?;
    Ok((mapping, reserve))
}

#[cfg(test)] mod tests;
