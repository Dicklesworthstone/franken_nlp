//! Closed document-job settings; no executable recipe or inferred lifetime budget.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, quantized::long::{Int8ExtractionMapLimits,
        stream::ExtractionDocumentBatchConfig}},
    jobs::JobWork,
    native_engine::{constrained_int8, strict_int8::Int8Work},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits}, source_planning::quantized::long::Int8SourceMapLimits},
    validation::grounded_fields::GroundingBudget,
};

pub(in crate::candidate_cli) const DOCUMENT_BYTES: usize = 16 * 1024;
const MAX_RESULT: usize = 64 * 1024 * 1024;

/// Every field is required and unknown fields are refused. A resume freezes
/// semantic settings alongside the exact shared schema and JobLimits; modeled
/// reduction headroom remains a separately admitted host-memory reservation.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct DocumentSettings {
    schema_version: u32,
    max_chunks: usize,
    max_chunk_bytes: usize,
    max_tokenizer_calls: usize,
    max_result_bytes: usize,
    max_live_value_bytes: usize,
    max_total_value_bytes: usize,
    verification: GroundingBudget,
    reduction_reserve_mib: u64,
}
fn parse(text: &str) -> Result<DocumentSettings, Failure> {
    if text.is_empty() || text.len() > DOCUMENT_BYTES { return Err(Failure::usage("document_job_configuration")); }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits { max_depth: 8, max_string_bytes: 4096 })
        .map_err(|_| Failure::usage("document_job_configuration"))?;
    serde_json::from_value(value).map_err(|_| Failure::usage("document_job_configuration"))
}
pub(in crate::candidate_cli) fn load(text: &str, args: &JobArgs, request: ExtractionBatchArgs,
    lifetime: JobLimits, limits: Limits) -> Result<(ExtractionDocumentBatchConfig, u64), Failure> {
    if args.task != "extract" || args.schema.is_none() || args.defaults.is_some() || args.document_config.is_none() {
        return Err(Failure::usage("document_job_requires_fixed_schema"));
    }
    lifetime.validate().map_err(|_| Failure::usage("job_limits"))?;
    let d = parse(text)?; let host = &args.host; let v = d.verification;
    if d.schema_version != 1 || !(1..=256).contains(&d.max_chunks)
        || !(4..=MAX_INPUT_BYTES).contains(&d.max_chunk_bytes) || d.max_chunk_bytes > host.max_input_bytes
        || !(1..=1_000_000).contains(&d.max_tokenizer_calls)
        || [d.max_result_bytes, d.max_live_value_bytes, d.max_total_value_bytes].iter().any(|&n| !(1..=MAX_RESULT).contains(&n))
        || d.max_result_bytes < host.max_result_bytes || d.max_result_bytes > lifetime.max_result_bytes
        || d.max_result_bytes > d.max_live_value_bytes || d.max_result_bytes > d.max_total_value_bytes
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || !(1..=1_000_000_000_000).contains(&v.max_scan_steps) {
        return Err(Failure::usage("document_job_limits"));
    }
    let reserve = d.reduction_reserve_mib.checked_mul(MIB)
        .filter(|&n| n != 0 && n <= limits.memory_bytes).ok_or_else(|| Failure::usage("document_job_memory"))?;
    // Retained schema/prompt/grammar plans, plus two complete result stages.
    // The actual host additionally reserves native map values, coordinate scan
    // vectors, journal/IO, model/KV/scratch and output guards. This is a necessary
    // modeled floor, not an RSS measurement or an admission permit.
    let per = (host.context_tokens as u64).checked_mul(32)
        .and_then(|n| n.checked_add(u64::from(host.max_grammar_states).checked_mul(512)?))
        .and_then(|n| n.checked_add(2 * crate::candidate_cli::extract::SCHEMA_BYTES as u64))
        .ok_or_else(|| Failure::usage("document_job_memory"))?;
    let floor = per.checked_mul(d.max_chunks as u64).and_then(|n| n.checked_add(256 * MIB))
        .and_then(|n| n.checked_add((host.max_input_bytes as u64).checked_mul(64)?))
        .and_then(|n| n.checked_add((d.max_result_bytes as u64).checked_mul(2)?))
        .ok_or_else(|| Failure::usage("document_job_memory"))?;
    if floor > limits.preparation_bytes { return Err(Failure::usage("document_job_preparation")); }
    let prompt = host.context_tokens.checked_sub(host.max_new_tokens).filter(|&n| n != 0)
        .ok_or_else(|| Failure::usage("document_job_context"))?;
    let per_document = scale(constrained_int8::planned_work(prompt, host.max_new_tokens)
        .map_err(|_| Failure::usage("document_job_work"))?, d.max_chunks as u64)?;
    let masks = host.max_mask_node_visits.checked_mul(d.max_chunks as u64)
        .ok_or_else(|| Failure::usage("document_job_work"))?;
    let debit = JobWork { model: per_document, mask_node_visits: masks };
    if !debit.fits(lifetime.max_work) { return Err(Failure::usage("document_job_lifetime_work")); }
    let count = usize::try_from(lifetime.max_attempts).map_err(|_| Failure::usage("document_job_attempts"))?;
    // Lifetime authority derives from DURABLE attempts, not input count. The
    // planner and runner recheck this relationship before storage or execution.
    let verification = GroundingBudget {
        max_fields: v.max_fields.checked_mul(count).ok_or_else(|| Failure::usage("document_job_attempts"))?,
        max_matches: v.max_matches.checked_mul(count).ok_or_else(|| Failure::usage("document_job_attempts"))?,
        max_scan_steps: v.max_scan_steps.checked_mul(lifetime.max_attempts).ok_or_else(|| Failure::usage("document_job_attempts"))?,
    };
    let config = ExtractionDocumentBatchConfig { request,
        limits: Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: host.max_input_bytes, max_chunk_bytes: d.max_chunk_bytes,
                max_chunk_tokens: host.context_tokens, context_tokens: host.context_tokens,
                reserved_tokens: host.max_new_tokens, max_chunks: d.max_chunks, max_tokenizer_calls: d.max_tokenizer_calls },
            reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8, max_reduction_levels: 8,
                max_task_calls: 1024, max_value_bytes: d.max_result_bytes, max_live_value_bytes: d.max_live_value_bytes,
                max_total_value_bytes: d.max_total_value_bytes, max_result_bytes: d.max_result_bytes },
            max_model_work: per_document, mask_limits: host.masks(), mask_visits_per_chunk: host.max_mask_node_visits,
            max_mask_visits: masks }, verification: v },
        max_model_work: lifetime.max_work.model, max_mask_visits: lifetime.max_work.mask_node_visits,
        max_verification: verification };
    config.validate().map_err(|_| Failure::usage("document_job_limits"))?;
    Ok((config, reserve))
}
fn scale(mut work: Int8Work, count: u64) -> Result<Int8Work, Failure> {
    if count == 0 { return Err(Failure::usage("document_job_work")); }
    for n in [&mut work.forward_positions, &mut work.projected_logits, &mut work.attention_pairs,
        &mut work.projections.dot_products, &mut work.projections.multiply_accumulates] {
        *n = n.checked_mul(count).ok_or_else(|| Failure::usage("document_job_work"))?;
    }
    Ok(work)
}

#[cfg(test)] mod tests;
