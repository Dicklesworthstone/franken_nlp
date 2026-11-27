//! Complete-document INT8 samples share one source, work and output contract.
use super::*;
use crate::{
    tasks::{mapreduce::ChunkLimits, sentiment::sampled::quantized::long::{
        Int8SampledSentimentMapLimits,
        memory::{Int8SampledSentimentMapMemoryLimits, Int8SampledSentimentMapMemoryRequirement},
    }},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(in crate::candidate_cli) struct SampledSentimentMapCommand {
    pub(in crate::candidate_cli) sampled: SampledSentimentCommand,
    pub(in crate::candidate_cli) mapping: MappingArgs,
    encoding: input_encoding::DocumentInputEncodingArgs,
}

#[derive(Args)]
pub(in crate::candidate_cli) struct MappingArgs {
    /// Refuse a document requiring more chunks; never omit its tail.
    #[arg(long, default_value_t = 64)]
    pub(in crate::candidate_cli) max_chunks: usize,
    /// Additional source-byte ceiling; every axis prompt must also fit context.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    /// Shared token-count probes, including refused oversized candidates.
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete document JSON, including every sample, explanation and span.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Cumulative canonical bytes of all retained complete chunk bundles.
    #[arg(long, default_value_t = 67_108_864)]
    max_retained_result_bytes: u64,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    let command = SampledSentimentCommand::augment_args(
        clap::Command::new("sentiment-sampled-map")
            .about("Sample every INT8 source chunk and aggregate complete-document score replicates")
            .long_about("Read one complete {document,axes?} JSON request. Losslessly partition the original UTF-8 source into nonoverlapping chunks that fit every requested axis prompt. Generate the entire independent score/explanation sample bundle for every chunk. Each document replicate pairs the same sample ordinal across chunks and weights scores by original Unicode scalar count, retaining a checked exact numerator. Mean, median, range and population standard deviation describe those replicates; they are not calibrated confidence, human ratings or single-context equivalents. All chunks and the complete native/mask/output budget are preflighted before the first forward. A failed chunk returns no completed document. --seed is mandatory. --compute-threads keeps one numerical team across all chunks, axes and samples; --prefill-rows uses bounded layer-major prompts. No model discovery, download, thinking or tools."));
    compute::definition(input_encoding::document_definition(MappingArgs::augment_args(command)))
        .mut_arg("input", |arg| arg.help("One complete sentiment JSON object; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle per chunk; document output has its own cap"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("1048576")
            .help("Whole-document forward ceiling across every chunk, axis and sample"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("100000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("10000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("100000000000000000"))
        .mut_arg("max_mask_node_visits", |arg| arg.default_value("1000000000000")
            .help("One nonrenewable mask allowance for the complete document, including every EOS"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints shared by complete-document planning, inference and reduction"))
        .after_long_help("Byte fallback is the default. Explicit --input-encoding safe-bpe-v1 preserves source bytes and excludes template-control tokens with ONE tokenizer-work ledger spanning accepted and rejected sizing probes and all chunk plans. Tokenizer, native, mask, retained-result and document-output limits never renew per chunk. The output retains every complete chunk bundle and original byte/scalar spans, plus the exact encoding policy and work. A different prompt encoding may change model output. One end-to-end cooperative deadline covers private input, model loading, preparation, execution, reduction and publication; blocking IO is not preempted. This candidate route has no numerical, task-quality, performance or release certification.")
}

pub(in crate::candidate_cli) struct MapLimits {
    pub common: CandidateArgs,
    pub shared: Limits,
    pub task: TaskBudget,
    pub mapping: Int8SampledSentimentMapLimits,
    pub memory: Int8SampledSentimentMapMemoryRequirement,
    pub encoding: UntrustedEncoding,
    pub prefill: Option<Int8PrefillLimits>,
    pub envelope_bytes: usize,
    pub minimum_memory_bytes: u64,
}

impl SampledSentimentMapCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        // The map owns the complete-document encoder. The shared sample args
        // retain byte defaults only while quoting common storage; they never
        // independently encode or receive a fresh per-chunk allowance.
        let mut sampled = SampledSentimentCommand::from_arg_matches(matches)?;
        sampled.compute = compute::Args::from_matches(matches)?;
        Ok(Self { sampled, mapping: MappingArgs::from_arg_matches(matches)?,
            encoding: input_encoding::DocumentInputEncodingArgs::from_matches(matches)? })
    }

    pub(in crate::candidate_cli) fn validate(&self) -> Result<MapLimits, CandidateError> {
        let encoding = self.encoding.policy()?;
        let (common, shared) = self.sampled.common()?;
        let prefill = self.sampled.prefill()?;
        let m = &self.mapping;
        if !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
            || self.sampled.max_result_bytes as u64 > m.max_retained_result_bytes {
            return Err(CandidateError::Arguments);
        }
        let task = self.sampled.budget(shared, SentimentAxis::ALL.len())?;
        let mapping = Int8SampledSentimentMapLimits {
            chunks: ChunkLimits { max_input_bytes: self.sampled.max_input_bytes,
                max_chunk_bytes: m.max_chunk_bytes.unwrap_or(self.sampled.max_input_bytes),
                max_chunk_tokens: shared.max_prompt_tokens,
                context_tokens: self.sampled.context_tokens,
                reserved_tokens: self.sampled.max_new_tokens - 1,
                max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls },
            max_model_work: self.sampled.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes,
        };
        let memory = Int8SampledSentimentMapMemoryLimits {
            context_tokens: self.sampled.context_tokens,
            preparation_reserve_bytes: shared.preparation_bytes,
            allocator_reserve_bytes: 64 * MIB,
            max_axes: SentimentAxis::ALL.len(),
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        }.requirements(task, self.sampled.task_limits(), mapping, &self.sampled.options()?, &encoding)
            .map_err(|_| CandidateError::Arguments)?;
        let envelope_bytes = memory.max_output_bytes.checked_add(4096)
            .and_then(|n| usize::try_from(n).ok()).ok_or(CandidateError::Arguments)?;
        // Session owns the external request and serialization independently of
        // the hosted compiler, retained plans and guarded result ownership.
        let staging = sum(&[128 * MIB, self.sampled.max_input_bytes as u64 * 32,
            (envelope_bytes as u64).checked_mul(2).ok_or(CandidateError::Arguments)?])?;
        if staging > shared.preparation_bytes { return Err(CandidateError::Arguments); }
        let extra = prefill.map(|p| p.validate()).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        let execution = sum(&[memory.native.kv_bytes, memory.workspace_bytes, memory.input_bytes,
            memory.output_reservation_bytes, extra, shared.compute.workspace_bytes,
            shared.weight_bytes, 192 * MIB, 65_536])?;
        let loading = sum(&[shared.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
        let minimum_memory_bytes = sum(&[shared.preparation_bytes, execution.max(loading)])?;
        if minimum_memory_bytes > shared.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, shared, task, mapping, memory, encoding, prefill,
            envelope_bytes, minimum_memory_bytes })
    }

    pub(in crate::candidate_cli) fn execute(self, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), Failure> {
        let admitted = self.validate().map_err(|_| Failure::usage("int8_sampled_sentiment_map_options"))?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::map::execute(self, admitted, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, admitted, input, output);
            Err(Failure::usage("int8_sampled_sentiment_map_profile_unavailable")) }
    }

    pub(in crate::candidate_cli) fn run(self, input: &mut impl Read, output: &mut impl Write,
        diagnostics: &mut impl Write) -> ExitCode {
        match self.execute(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_failure(failure, diagnostics),
        }
    }
}

fn sum(values: &[u64]) -> Result<u64, CandidateError> {
    values.iter().try_fold(0_u64, |sum, &value| sum.checked_add(value).ok_or(CandidateError::Arguments))
}

#[cfg(test)] mod tests;
