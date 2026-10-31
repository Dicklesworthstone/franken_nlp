//! Complete-document INT4 extraction with explicit chunk and reduction budgets.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, grouped_int4::long::{Int4ExtractionMappingLimits, Int4ExtractionMapLimits,
        cohort::{Int4ExtractionMapCohortMemoryLimits, Int4ExtractionMapCohortRequirement}}},
    native_engine::{portable_int8::ProjectionWork, strict_int4::Int4Work},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MapArgs { pub extraction: extract::ExtractionArgs, mapping: MappingArgs }
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct MappingArgs {
    /// Explicit shared-layer extraction over at most N source chunks (1..=64).
    /// The full document is still prepared and validated before inference.
    #[arg(long, value_name = "N")]
    cohort_chunks: Option<usize>,
    /// Fail if complete input requires more chunks; never truncate the source.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional chunk byte cap; actual prompt/schema context fit is checked separately.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete document result, separate from each chunk's --max-result-bytes.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_live_value_bytes: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_value_bytes: usize,
    /// Additional modeled reduction/allocator reserve, not an RSS guarantee.
    #[arg(long, default_value_t = 64)]
    reduction_reserve_mib: u64,
    /// Nonrenewable document mask allowance across every chunk.
    #[arg(long, default_value_t = 128_000_000_000)]
    max_total_mask_node_visits: u64,
    #[arg(long, default_value_t = 262_144)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 10_000_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_attention_pairs: u64,
    /// Actual per-group dot products, not an INT8 projection estimate.
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_multiply_accumulates: u64,
    /// Independent source-field verification budget across the complete document.
    #[arg(long, default_value_t = 4096)]
    max_extraction_fields: usize,
    /// Total original-coordinate occurrence spans across every chunk.
    #[arg(long, default_value_t = 16384)]
    max_extraction_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_extraction_scan_steps: u64,
}
pub(super) fn definition() -> clap::Command {
    MappingArgs::augment_args(extract::definition().name("extract-map")
        .about("Extract one exact schema over every chunk of a complete document using grouped INT4")
        .long_about("Losslessly partition one complete UTF-8 document with the pinned source encoder and actual schema/template prompt capacity. Compile and admit ALL chunks before the first native forward. Execution is serial unless --cohort-chunks N explicitly groups up to N independent chunks through shared native INT4 decoder layers; N must be 1..64 and cannot exceed --max-chunks. The complete window's native, decoder, output and reduction storage is admitted before IO, with exact native contexts for each actual window and the final partial window. Every chunk keeps its original task budget, grammar, source language and causal KV. Output retains each independent exact-schema JSON string and chunk byte/scalar lineage; --source-membership independently verifies all compatible verbatim occurrences within each chunk and lifts them to original-document coordinates. It does not find occurrences in other chunks from which the value was not extracted. Required fields must be satisfiable in each chunk. There is no merged global object, unconstrained repair, overlapping-window analysis or single-context equivalence claim. Work, mask, field/span verification, tokenization and reduction budgets are for the whole document and never renew per chunk or cohort. No partial document success is published. Source-conversion and digest-pinned snapshot routes share the same checked path. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("One complete UTF-8 document; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576").help("Maximum bytes of the COMPLETE original document"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoint allowance for hosted document preparation, chunks and reduction")))
}
#[derive(Clone, Copy, Debug)]
pub(super) struct MapLimits {
    pub common: Int4Limits,
    pub mapping: Int4ExtractionMapLimits,
    pub reduction_reserve_bytes: u64,
    pub max_result_bytes: usize,
    pub workspace_bytes: u64,
    pub cohort_chunks: Option<usize>,
    pub cohort: Option<Int4ExtractionMapCohortRequirement>,
}
impl MapArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { extraction: extract::ExtractionArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        let m = self.mapping;
        let mut common = self.extraction.validate(args)?;
        if args.max_input_bytes < 4 || !(1..=256).contains(&m.max_chunks)
            || m.cohort_chunks.is_some_and(|n| !(1..=64).contains(&n) || n > m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || [m.max_map_result_bytes, m.max_live_value_bytes, m.max_total_value_bytes]
                .iter().any(|&n| !(1..=64 * 1024 * 1024).contains(&n))
            || m.max_map_result_bytes > m.max_live_value_bytes || m.max_map_result_bytes > m.max_total_value_bytes
            || common.result_bytes > m.max_map_result_bytes
            || m.max_total_mask_node_visits < self.extraction.mask_visits()
            || m.max_total_mask_node_visits > 1_000_000_000_000_000
            || !(1..=16 * 1024 * 1024).contains(&m.max_forward_positions)
            || m.max_projected_logits == 0 || m.max_attention_pairs == 0
            || m.max_dot_products == 0 || m.max_multiply_accumulates == 0
            || !(1..=1_000_000).contains(&m.max_extraction_fields)
            || !(1..=1_000_000).contains(&m.max_extraction_evidence_spans)
            || !(1..=1_000_000_000_000).contains(&m.max_extraction_scan_steps) {
            return Err(CandidateError::Arguments);
        }
        let reduction_reserve_bytes = mib(m.reduction_reserve_mib)?;
        let mapping = self.mapping_limits(args)?;
        let budget = self.extraction.budget(args, common);
        // Every prompt, grammar and schema stays retained until whole-document
        // execution completes. The complete result has its own staging charge.
        let per_chunk = sum(&[args.context_tokens as u64 * 32,
            u64::from(budget.max_grammar_states) * 512,
            crate::candidate_cli::extract::SCHEMA_BYTES as u64 * 2])?;
        let preparation_floor = sum(&[times(per_chunk, m.max_chunks as u64)?, 256 * MIB,
            args.max_input_bytes as u64 * 64, (m.max_map_result_bytes as u64 + 4096) * 2])?;
        if common.preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        let workspace_bytes = workspace_bytes(budget, mapping, m.max_chunks, reduction_reserve_bytes)?;
        let tokens = times(m.max_chunks as u64, u64::from(budget.max_output_tokens))?;
        let output = sum(&[m.max_map_result_bytes as u64 * 4, times(tokens, 8)?, 4096])?;
        // Additional to the single-task quote: the hosted original/schema and
        // preparation capture, full reduction frontier, and complete output.
        // Over-counting the small per-chunk output reserve is conservative.
        let input = sum(&[args.max_input_bytes as u64 + 1,
            crate::candidate_cli::extract::SCHEMA_BYTES as u64 + 1, common.preparation_bytes])?;
        let cohort = m.cohort_chunks.map(|max_chunks| Int4ExtractionMapCohortMemoryLimits {
            max_chunks, context_tokens: args.context_tokens,
            max_output_tokens_per_chunk: u64::from(budget.max_output_tokens),
            max_result_bytes_per_chunk: budget.max_output_bytes,
        }.requirements(args.group_width).map_err(|_| CandidateError::Arguments)).transpose()?;
        if let Some(quote) = cohort {
            // The hosted cohort wrapper carries bounded physical statistics
            // beside the unchanged complete document. Its output_claim prices
            // the additional 4096 bytes with the same four-payload multiplier.
            let output = sum(&[output, 4096 * 4])?;
            let execution = sum(&[quote.native.kv_bytes, quote.native.rope_bytes,
                quote.native.scratch_payload_bound, ALLOCATOR_BYTES, quote.workspace_bytes,
                input, workspace_bytes, output])?;
            common.minimum_memory_bytes = grouped_memory_floor(args, common, execution)?;
            // common.native remains one chunk: grouping changes scheduling,
            // never the source/schema plan's original per-chunk TaskBudget.
        } else {
            common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, input, workspace_bytes, output])?;
        }
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, mapping, reduction_reserve_bytes, max_result_bytes: m.max_map_result_bytes,
            workspace_bytes, cohort_chunks: m.cohort_chunks, cohort })
    }
    fn mapping_limits(&self, args: &Int4Args) -> Result<Int4ExtractionMapLimits, CandidateError> {
        let m = self.mapping;
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.context_tokens).min(args.max_input_bytes),
            max_chunk_tokens: args.context_tokens, context_tokens: args.context_tokens,
            reserved_tokens: args.max_new_tokens, max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        Ok(Int4ExtractionMapLimits { mapping: Int4ExtractionMappingLimits {
            chunks, reduction: ExecutionLimits { map_batch_chunks: m.cohort_chunks.unwrap_or(1), reduce_fan_in: 8,
                max_reduction_levels: 8, max_task_calls: 1024, max_value_bytes: m.max_map_result_bytes,
                max_live_value_bytes: m.max_live_value_bytes, max_total_value_bytes: m.max_total_value_bytes,
                max_result_bytes: m.max_map_result_bytes },
            max_model_work: Int4Work { forward_positions: m.max_forward_positions,
                projected_logits: m.max_projected_logits, attention_pairs: m.max_attention_pairs,
                projections: ProjectionWork { dot_products: m.max_dot_products,
                    multiply_accumulates: m.max_multiply_accumulates } },
            mask_limits: self.extraction.masks(), mask_visits_per_chunk: self.extraction.mask_visits(),
            max_mask_visits: m.max_total_mask_node_visits,
        }, verification: GroundingBudget { max_fields: m.max_extraction_fields,
            max_matches: m.max_extraction_evidence_spans, max_scan_steps: m.max_extraction_scan_steps } })
    }
    pub(super) fn request(&self, schema: String, budget: TaskBudget) -> Result<ExtractionBatchArgs, CandidateError> {
        // The source language is compiled independently for each REAL chunk.
        // A whole-document source automaton or placeholder would prove less.
        self.extraction.shared_defaults(schema, budget)
    }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn grouped_memory_floor(args: &Int4Args, common: Int4Limits, execution: u64) -> Result<u64, CandidateError> {
    let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    sum(&[common.preparation_bytes, resident, loading.max(saving).max(execution), CLEANUP_BYTES])
}
fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
fn workspace_bytes(budget: TaskBudget, limits: Int4ExtractionMapLimits, chunks: usize, reserve: u64)
    -> Result<u64, CandidateError> {
    if chunks == 0 || chunks > limits.mapping.chunks.max_chunks || chunks > 256 || reserve == 0 {
        return Err(CandidateError::Arguments);
    }
    let native = times(budget.max_output_bytes, chunks as u64)?;
    let staged = times(sum(&[native, limits.mapping.reduction.max_live_value_bytes as u64])?, 4)?;
    let tokens = times(times(u64::from(budget.max_output_tokens), chunks as u64)?, 8)?;
    let fields = times(limits.verification.max_fields as u64, std::mem::size_of::<SourceFieldEvidence>() as u64)?;
    let spans = times(times(limits.verification.max_matches as u64,
        std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
    sum(&[staged, tokens, fields, spans, reserve])
}
