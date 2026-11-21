//! Complete-source independent sampled sentiment with one bounded map budget.
use super::*;
use crate::{
    tasks::{mapreduce::ChunkLimits, sentiment::sampled::long::{SampledSentimentMapLimits,
        memory::{SampledSentimentMapMemoryLimits, SampledSentimentMapMemoryRequirement}}},
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
pub(in crate::candidate_cli::int4) mod corpus;
pub(in crate::candidate_cli::int4) mod job;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct MapArgs {
    sampled: SampledArgs,
    mapping: MappingArgs,
    encoding: EncodingArgs,
}

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct MappingArgs {
    /// Refuse an over-limit document; every original source byte must be covered.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional source-byte ceiling; all axis prompts must also fit context.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    /// Bound all token-count probes, including rejected oversized candidates.
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Whole-document result, including all chunks, explanations and replicates.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Cumulative canonical bytes of retained complete chunk bundles.
    #[arg(long, default_value_t = 67_108_864)]
    max_retained_result_bytes: u64,
}

#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EncodingArgs {
    /// Shared bytes across every chunk probe and actual plan; safe-bpe-v1 only.
    #[arg(long)]
    max_tokenizer_input_bytes: Option<usize>,
    /// Shared encoded tokens across probes and plans; safe-bpe-v1 only.
    #[arg(long)]
    max_tokenizer_output_tokens: Option<usize>,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let command = super::definition()
            .name("sentiment-sampled-map")
            .about("Sample every source chunk and report explicit document score replicates")
            .long_about("Read one complete {document,axes?} JSON request. Deterministically partition the exact original UTF-8 source into nonoverlapping chunks that fit every requested axis prompt. Generate the complete independent score/explanation sample bundle for every chunk. All chunks, cold native work and mask allowances are preflighted before the first forward; one failed chunk returns no document result. Each document replicate pairs the same sample ordinal across chunks and weights integer scores by original Unicode scalar count, with a checked exact numerator. Mean, median, range and population standard deviation describe the resulting document replicates. They are not calibrated confidence, human ratings or single-context equivalents. Results retain every complete chunk bundle and original byte/scalar spans. The explicit seed, whole-document native/mask ceilings and one tokenizer-work ledger span the complete map. --reuse-prefill remains private to the current chunk and axis. --compute-threads 2..64 keeps one numerical team across the whole map; alternatively --prefill-rows 1..64 uses serial layer-major prompts. Byte fallback is the default; explicit safe-bpe-v1 changes prompt tokens under a shared bounded encoding allowance. Uses local source conversion or an independently pinned snapshot. Snapshot saving, cohorts, tools, thinking and per-chunk policy overrides are unsupported. No partial document is published.")
            .mut_arg("input", |arg| arg.help("One complete typed sentiment JSON object; '-' reads stdin; not NDJSON"))
            .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
                .help("Maximum bytes of the complete JSON request, including the original source"))
            .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle per chunk; --max-map-result-bytes caps the document"))
            .mut_arg("max_forward_positions", |arg| arg.default_value("1048576")
                .help("Whole-document cold forward ceiling across every chunk, axis and sample"))
            .mut_arg("max_projected_logits", |arg| arg.default_value("100000000000"))
            .mut_arg("max_attention_pairs", |arg| arg.default_value("10000000000000"))
            .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
            .mut_arg("max_multiply_accumulates", |arg| arg.default_value("100000000000000000"))
            .mut_arg("max_mask_node_visits", |arg| arg.default_value("1000000000000")
                .help("One nonrenewable mask allowance for every chunk, axis, sample and EOS"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for whole-document planning, native work and reduction"))
            .mut_arg("save_snapshot", |arg| arg.help("Unsupported on complete sampled maps; use source conversion or a pinned existing snapshot"));
    EncodingArgs::augment_args(MappingArgs::augment_args(
        compute::definition(input_encoding::definition(command))))
}

#[derive(Clone, Copy, Debug)]
struct MapLimits {
    common: Int4Limits,
    mapping: SampledSentimentMapLimits,
    planning: SampledSentimentLimits,
    memory: SampledSentimentMapMemoryRequirement,
    cli_preparation_bytes: u64,
    envelope_bytes: usize,
}

impl MapArgs {
    #[cfg(any(feature = "asupersync-runtime", test))]
    fn execution(self) -> &'static str {
        use crate::tasks::sentiment::sampled::long::{SAMPLED_SENTIMENT_MAP_EXECUTION,
            SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION, SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION,
            SAMPLED_SENTIMENT_MAP_SPARSE_REUSED_PREFILL_EXECUTION};
        match (self.sampled.max_selected_rows.is_some(), self.sampled.reuse_prefill) {
            (false, false) => SAMPLED_SENTIMENT_MAP_EXECUTION,
            (false, true) => SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION,
            (true, false) => SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION,
            (true, true) => SAMPLED_SENTIMENT_MAP_SPARSE_REUSED_PREFILL_EXECUTION,
        }
    }

    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<Self, clap::Error> {
        Ok(Self { sampled: SampledArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?,
            encoding: EncodingArgs::from_arg_matches(matches)? })
    }

    fn input_encoding(self, args: &Int4Args) -> Result<UntrustedEncoding, CandidateError> {
        if args.input_encoding.is_default() && self.encoding != EncodingArgs::default() {
            return Err(CandidateError::Arguments);
        }
        args.input_encoding.policy(
            self.encoding.max_tokenizer_input_bytes.unwrap_or(SafeBpeLimits::MAX_INPUT_BYTES),
            self.encoding.max_tokenizer_output_tokens.unwrap_or(SafeBpeLimits::MAX_OUTPUT_TOKENS))
    }

    fn validate(self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        if args.save_snapshot.is_some() { return Err(CandidateError::Arguments); }
        let encoding = self.input_encoding(args)?;
        // The map owns one aggregate encoding allowance. Common storage uses
        // byte defaults here; the complete policy workspace is quoted below.
        let mut storage = args.clone();
        storage.input_encoding = input_encoding::InputEncodingArgs::default();
        let mut common = self.sampled.validate(&storage)?;
        let m = self.mapping;
        if !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|bytes| !(4..=MAX_INPUT_BYTES).contains(&bytes))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
            || self.sampled.max_result_bytes as u64 > m.max_retained_result_bytes {
            return Err(CandidateError::Arguments);
        }
        let task = self.sampled.budget(args, common, SentimentAxis::ALL.len())?;
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.max_input_bytes),
            max_chunk_tokens: common.max_prompt_tokens, context_tokens: args.context_tokens,
            reserved_tokens: args.max_new_tokens - 1,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = SampledSentimentMapLimits { chunks,
            max_model_work: self.sampled.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes };
        let planning = self.sampled.task_limits(args);
        let options = self.sampled.options(args)?;
        let memory = SampledSentimentMapMemoryLimits { context_tokens: args.context_tokens,
            preparation_reserve_bytes: common.preparation_bytes,
            allocator_reserve_bytes: ALLOCATOR_BYTES, max_axes: SentimentAxis::ALL.len(),
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)? }
            .requirements(args.group_width, task, planning, mapping, &options, &encoding)
            .map_err(|_| CandidateError::Arguments)?;
        let envelope_bytes = memory.max_output_bytes.checked_add(4096)
            .and_then(|bytes| usize::try_from(bytes).ok()).ok_or(CandidateError::Arguments)?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes,
            (envelope_bytes as u64).checked_mul(2).ok_or(CandidateError::Arguments)?])?;
        let prefill = common.prefill.map(Int4PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let execution = sum(&[memory.input_bytes, memory.native.kv_bytes,
            memory.workspace_bytes, memory.output_reservation_bytes, prefill, common.compute.workspace_bytes])?;
        common.minimum_memory_bytes = sum(&[cli_preparation_bytes, resident,
            loading.max(execution), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, mapping, planning, memory, cli_preparation_bytes, envelope_bytes })
    }

    pub(in crate::candidate_cli::int4) fn run(self, args: Int4Args, input: &mut impl Read,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let _ = writeln!(diagnostics, "fnlp candidate sampled sentiment map: {}", failure.code);
                failure.exit.as_process_exit()
            }
        }
    }

    fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output);
            Err(Failure::usage("sampled_sentiment_map_profile_unavailable")) }
    }
}
