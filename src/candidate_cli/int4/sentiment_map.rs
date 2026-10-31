//! Complete-document dimensional sentiment under one finite native budget.
use super::*;
use crate::tasks::{ir::TaskBudget, mapreduce::ChunkLimits,
    sentiment::{SentimentAggregationPolicy, SentimentLimits, SentimentRequest,
        grouped_int4::long::{Int4SentimentMapLimits,
            memory::{Int4SentimentMapMemoryLimits, Int4SentimentMapMemoryRequirement}}}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MapArgs {
    scoring: classify::ScoringArgs,
    mapping: MappingArgs,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct MappingArgs {
    /// Refuse a document needing more chunks; never truncate the source.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional byte ceiling; every axis also checks its exact prompt fit.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete semantic document result; framing has a separate fixed allowance.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Complete retained native chunk results, never renewed per chunk.
    #[arg(long, default_value_t = 67_108_864)]
    max_retained_result_bytes: u64,
    /// Required original-scalar coverage by estimated chunks; 1000000 requires all.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_estimated_coverage_ppm: u32,
    /// Execute up to N complete chunks together; N times the requested axes must be <=64.
    #[arg(long, value_name = "N")]
    cohort_chunks: Option<usize>,
}

pub(super) fn definition() -> clap::Command {
    MappingArgs::augment_args(classify::definition().name("sentiment-map")
        .about("Score every chunk of a complete document and report descriptive dimension aggregates")
        .long_about("Read the same strict {document, axes?, policy?} JSON request as sentiment, then losslessly partition the complete original source using the pinned encoder and every requested axis's exact prompt capacity. Preflight all chunks and all five whole-document native work ceilings before the first forward. Preserve complete per-chunk distributions and byte/scalar lineage. The named reduction weights chunks by original Unicode scalar counts, includes abstained chunks in descriptive moments, and publishes an estimate only when both the explicit estimated-coverage threshold and existing peak/entropy policy pass. This is an uncalibrated chunk aggregation, not a full-context model probability or psychological measurement. Serial execution is the default; --cohort-chunks explicitly shares native decoder layers across complete independent chunks, with at most 64 total axis heads. Actual geometry is checked before model IO and never silently clamped or retried. All-chunk plans, native workspaces, retained results and complete publication are memory-admitted before input IO. Source conversion and independently pinned snapshots share this route. A failed chunk publishes no partial document. No sampling, focus presets, prompt overrides, thinking, tools, --prefill-rows or --cohort-heads. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("One complete typed sentiment JSON object; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Maximum bytes of the complete JSON request, including the exact source document"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native result per chunk; --max-map-result-bytes bounds the document"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("262144")
            .help("Whole-document native forward ceiling across all chunks and axes"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("1000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for the complete hosted plan, all chunks and reduction")))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MapLimits {
    pub common: Int4Limits,
    mapping: Int4SentimentMapLimits,
    planning: SentimentLimits,
    memory: Int4SentimentMapMemoryRequirement,
    cli_preparation_bytes: u64,
    envelope_bytes: usize,
}
impl MapArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        let mut common = self.scoring.validate(args)?;
        let m = self.mapping;
        if args.max_input_bytes < 4 || !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|bytes| !(4..=MAX_INPUT_BYTES).contains(&bytes))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
            || m.max_map_result_bytes > m.max_retained_result_bytes
            || common.result_bytes as u64 > m.max_retained_result_bytes
            || m.cohort_chunks.is_some_and(|chunks| !(1..=64).contains(&chunks) || chunks > m.max_chunks) {
            return Err(CandidateError::Arguments);
        }
        let aggregation = SentimentAggregationPolicy {
            minimum_estimated_coverage_ppm: m.minimum_estimated_coverage_ppm };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let budget = self.budget(common)?;
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.max_input_bytes),
            max_chunk_tokens: common.max_prompt_tokens,
            context_tokens: args.context_tokens, reserved_tokens: budget.max_output_tokens as usize,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = Int4SentimentMapLimits { chunks, max_model_work: self.scoring.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes, aggregation };
        let planning = self.scoring.sentiment_limits(args);
        // Before request IO, price the largest valid geometry for the four
        // available axes. Actual axes*cohort_chunks must still fit 64 later.
        let memory = Int4SentimentMapMemoryLimits { context_tokens: args.context_tokens,
            preparation_reserve_bytes: common.preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES,
            max_axes: 4, cohort_chunks: m.cohort_chunks }
            .requirements(args.group_width, budget, planning, mapping).map_err(|_| CandidateError::Arguments)?;
        let envelope_bytes = memory.max_output_bytes.checked_add(4096)
            .and_then(|bytes| usize::try_from(bytes).ok()).ok_or(CandidateError::Arguments)?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes,
            (envelope_bytes as u64).checked_mul(2).ok_or(CandidateError::Arguments)?])?;
        let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
        let execution = sum(&[memory.native.kv_bytes, memory.input_bytes,
            memory.workspace_bytes, memory.output_reservation_bytes])?;
        common.minimum_memory_bytes = sum(&[cli_preparation_bytes, resident,
            loading.max(saving).max(execution), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, mapping, planning, memory, cli_preparation_bytes, envelope_bytes })
    }
    fn budget(self, common: Int4Limits) -> Result<TaskBudget, CandidateError> {
        let mut budget = self.scoring.budget(common);
        // One complete chunk can contain four independent axes. Freeze this
        // task authority for both serial and cohort execution; the actual
        // resident engine geometry is priced separately by the pure quote.
        budget.max_kv_bytes = common.native.kv_bytes.checked_mul(4).ok_or(CandidateError::Arguments)?;
        Ok(budget)
    }
    fn admit_request(self, request: &SentimentRequest) -> Result<(), CandidateError> {
        if request.axes.is_empty() || request.axes.len() > 4
            || self.mapping.cohort_chunks.is_some_and(|chunks|
                chunks.checked_mul(request.axes.len()).is_none_or(|heads| heads > 64)) {
            return Err(CandidateError::Planning);
        }
        Ok(())
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
