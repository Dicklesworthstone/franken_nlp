//! Complete-document dimensional sentiment under one finite native budget.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
use crate::tasks::{ir::TaskBudget, mapreduce::ChunkLimits,
    sentiment::{SentimentAggregationPolicy, SentimentLimits, SentimentRequest,
        grouped_int4::long::{Int4SentimentMapLimits,
            memory::{Int4SentimentMapMemoryLimits, Int4SentimentMapMemoryRequirement}}}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
pub(super) mod corpus;
pub(super) mod job;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MapArgs {
    scoring: classify::ScoringArgs,
    mapping: MappingArgs,
    encoding: MapEncodingArgs,
}
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MapEncodingArgs {
    /// Cumulative bytes across all attempted chunks and actual plans;
    /// safe-bpe-v1 only, at most 1048576. Source byte limits are separate.
    #[arg(long, value_name = "BYTES")]
    max_tokenizer_input_bytes: Option<usize>,
    /// Cumulative encoded tokens across all probes and final chunk plans;
    /// safe-bpe-v1 only, at most 1048576. Native context limits are separate.
    #[arg(long, value_name = "COUNT")]
    max_tokenizer_output_tokens: Option<usize>,
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

/// One document owns a bounded compact-tokenizer allowance. Population
/// callers additionally retain that allowance through exact reconstruction.
pub(super) fn one_off_definition() -> clap::Command {
    MapEncodingArgs::augment_args(input_encoding::definition(definition()))
        .after_long_help("--input-encoding safe-bpe-v1 chooses chunks using actual compact source-token counts. Every attempted chunk, including oversized probes, and every actual chunk plan spends one shared tokenizer allowance. Each encoded chunk is reused across its sentiment axes. --max-tokenizer-input-bytes and --max-tokenizer-output-tokens default to 1048576 each; pair checks and examined pair bytes use their explicit shared flags. Corpus and retained-job documents share that allowance across preparation AND exact execution reconstruction; neither pass renews it. The result's input_encoding.work records the completed planning pass, not the summed two-pass cost. Each new document or explicit resumed attempt receives its own bounded tokenizer ledger; native work and durable attempts retain their existing lifetime accounting. Exhaustion refuses the whole document before any native forward. Context, source bytes, native work, output and memory retain separate ceilings. The exact policy and all four limits authenticate retained-job resume. This selects a different prompt token sequence and makes no model-quality or tokenizer-parity claim. Compact maps accept pinned sources or snapshot reload but refuse --save-snapshot, whose separate pre-save planning would repeat tokenizer work. Byte-fallback output and job recipes are unchanged when the option is omitted.")
}

/// Keep numerical teams opt-in on the one-document route; definitions reused
/// by corpora and retained jobs do not acquire an unwired compute flag.
pub(super) fn threaded_definition() -> clap::Command {
    let command = one_off_definition();
    let encoding_help = command.get_after_long_help().map(|help| help.to_string()).unwrap_or_default();
    compute::definition(command).after_long_help(format!(
        "{encoding_help}\n\n--compute-threads N keeps one admitted output-row team across every chunk and sentiment axis. It supports byte-fallback and safe-bpe-v1 with the same complete-document tokenizer and native work ceilings. Actual projection counters are reported separately in compute. Cannot combine with --cohort-chunks; corpus and durable-job routes retain their existing schedules."))
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
            mapping: MappingArgs::from_arg_matches(matches)?,
            encoding: if matches.try_get_one::<String>("input_encoding").ok().flatten().is_some() {
                MapEncodingArgs::from_arg_matches(matches)?
            } else { MapEncodingArgs::default() } })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        if args.compute.threads() != 1 { return Err(CandidateError::Arguments); }
        self.validate_document(args, 4)
    }
    /// Corpus callers can narrow every document's declared axis capability.
    /// One-off maps continue to admit their original four-axis request space.
    fn validate_with_max_axes(self, args: &Int4Args, max_axes: usize) -> Result<MapLimits, CandidateError> {
        if args.compute.threads() != 1 {
            return Err(CandidateError::Arguments);
        }
        self.validate_document(args, max_axes)
    }
    fn validate_with_threads(self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        if args.compute.threads() == 1 { self.validate(args) }
        else { self.validate_document(args, 4) }
    }
    fn input_encoding(self, args: &Int4Args) -> Result<UntrustedEncoding, CandidateError> {
        if args.input_encoding.is_default() && self.encoding != MapEncodingArgs::default() {
            return Err(CandidateError::Arguments);
        }
        args.input_encoding.policy(
            self.encoding.max_tokenizer_input_bytes.unwrap_or(SafeBpeLimits::MAX_INPUT_BYTES),
            self.encoding.max_tokenizer_output_tokens.unwrap_or(SafeBpeLimits::MAX_OUTPUT_TOKENS))
    }
    fn validate_document(self, args: &Int4Args, max_axes: usize) -> Result<MapLimits, CandidateError> {
        if !(1..=4).contains(&max_axes)
            || args.prefill()?.is_some() || self.scoring.cohort_head_limit().is_some()
            || (args.compute.threads() != 1 && self.mapping.cohort_chunks.is_some()) {
            return Err(CandidateError::Arguments);
        }
        let encoding = self.input_encoding(args)?;
        if encoding != UntrustedEncoding::ByteFallback && args.save_snapshot.is_some() {
            return Err(CandidateError::Arguments);
        }
        // Shared scorer admission prices fixed storage and native authority.
        // This map's independently validated policy is charged below with its
        // actual cumulative ceilings, not the one-off source/context defaults.
        let mut storage_args = args.clone();
        storage_args.input_encoding = input_encoding::InputEncodingArgs::default();
        let mut common = self.scoring.validate_one_off_with_prefill(&storage_args)?;
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
        let budget = self.budget_with_max_axes(common, max_axes)?;
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
        // Before request IO, price the largest valid declared axis geometry.
        // Actual axes*cohort_chunks must still fit 64 later.
        let memory = Int4SentimentMapMemoryLimits { context_tokens: args.context_tokens,
            preparation_reserve_bytes: common.preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES,
            max_axes, cohort_chunks: m.cohort_chunks }
            .requirements(args.group_width, budget, planning, mapping)
            .and_then(|required| required.with_input_encoding(&encoding)).map_err(|_| CandidateError::Arguments)?;
        let envelope_bytes = memory.max_output_bytes.checked_add(4096)
            .and_then(|bytes| usize::try_from(bytes).ok()).ok_or(CandidateError::Arguments)?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes,
            (envelope_bytes as u64).checked_mul(2).ok_or(CandidateError::Arguments)?])?;
        let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
        let execution = sum(&[memory.native.kv_bytes, memory.input_bytes,
            memory.workspace_bytes, memory.output_reservation_bytes, common.compute.workspace_bytes])?;
        common.minimum_memory_bytes = sum(&[cli_preparation_bytes, resident,
            loading.max(saving).max(execution), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, mapping, planning, memory, cli_preparation_bytes, envelope_bytes })
    }
    fn budget(self, common: Int4Limits) -> Result<TaskBudget, CandidateError> {
        self.budget_with_max_axes(common, 4)
    }
    fn budget_with_max_axes(self, common: Int4Limits, max_axes: usize) -> Result<TaskBudget, CandidateError> {
        if !(1..=4).contains(&max_axes) { return Err(CandidateError::Arguments); }
        let mut budget = self.scoring.budget(common);
        // One complete chunk can contain every declared axis. Freeze this
        // task authority for both serial and cohort execution; the actual
        // resident engine geometry is priced separately by the pure quote.
        budget.max_kv_bytes = common.native.kv_bytes.checked_mul(max_axes as u64).ok_or(CandidateError::Arguments)?;
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
        let limits = self.validate_with_threads(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
