//! Complete INT8 sentiment documents use the existing strict request wire.
use super::scored::{ScoredArgs, prefill::ScoringPrefillArgs};
use super::*;
use crate::{
    tasks::{
        ir::TaskBudget,
        mapreduce::ChunkLimits,
        sentiment::{
            SentimentAggregationPolicy,
            quantized::long::{
                Int8SentimentMapLimits,
                memory::{Int8SentimentMapMemoryLimits, Int8SentimentMapMemoryRequirement},
            },
        },
    },
    tokenizer::untrusted::UntrustedEncoding,
};

pub(crate) struct SentimentMapCommand {
    pub(super) args: ScoredArgs,
    pub(super) mapping: MappingArgs,
    pub(super) prefill: ScoringPrefillArgs,
    pub(super) encoding: input_encoding::DocumentInputEncodingArgs,
}
#[derive(Args)]
pub(super) struct MappingArgs {
    /// Refuse sources requiring more chunks; never omit a tail.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Independent source-byte cap; actual prompts must also fit context.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete document JSON, including every original chunk and aggregate.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Cumulative retained chunk-result bytes across the whole document.
    #[arg(long, default_value_t = 16_777_216)]
    max_retained_result_bytes: u64,
    /// Original scalar coverage by non-abstained chunks, per affect dimension.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_estimated_coverage_ppm: u32,
}

pub(super) fn definition() -> clap::Command {
    let command = clap::Command::new("sentiment-map")
            .about("Score every chunk of a complete INT8 document and aggregate independent affect dimensions")
            .long_about("Read one strict {document, axes?, policy?} JSON object. Losslessly partition the original source, plan every selected dimension of every chunk, then reuse one admitted native INT8 cache. Every five-bin candidate includes EOS and a full-vocabulary denominator. Whole-document model and tokenizer work never renew per chunk. Output retains all original byte/scalar spans and chunk distributions alongside an original-scalar-weighted mixture, including abstained chunks. Coverage and the original peak/entropy policy govern each aggregate estimate. This descriptive mixture is uncalibrated and is not equivalent to a single full-context model posterior. Any failed chunk, late cancellation or exhausted output allowance returns no completed document. --prefill-rows explicitly uses bounded layer-major prompts without changing the candidate language. No sampling, thinking, tools or caller-supplied chunk scores.");
    let command = ScoringPrefillArgs::augment_args(ScoredArgs::augment_args(command));
    input_encoding::document_definition(MappingArgs::augment_args(command))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576"))
        .mut_arg("max_result_bytes", |arg| arg.default_value("65536")
            .help("Complete result ceiling for each native chunk; document output has its own cap"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("262144")
            .help("Whole-document forward allowance across every source chunk and affect dimension"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("1000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("10000000000000000"))
        .after_long_help("Compact input is explicit: --input-encoding safe-bpe-v1 preserves exact source bytes while excluding template-control tokens. One bounded tokenizer ledger covers every successful or rejected sizing probe and every final chunk/dimension plan. --max-tokenizer-input-bytes, --max-tokenizer-output-tokens, --max-tokenizer-pair-checks and --max-tokenizer-pair-bytes apply to that whole preparation, separately from native context and model work. There is no per-chunk reset or fallback. The complete result records the selected policy and actual tokenizer work. Compact prompts make no output-parity, quality or speed claim. The cooperative deadline spans input, model loading, planning, inference, reduction and publication; blocking IO is not preempted.")
}

pub(super) struct MapLimits {
    pub common: CandidateArgs,
    pub shared: Limits,
    pub task: TaskBudget,
    pub mapping: Int8SentimentMapLimits,
    pub memory: Int8SentimentMapMemoryRequirement,
    pub encoding: UntrustedEncoding,
    pub envelope_bytes: usize,
}
impl SentimentMapCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self {
            args: ScoredArgs::from_arg_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?,
            prefill: ScoringPrefillArgs::from_arg_matches(matches)?,
            encoding: input_encoding::DocumentInputEncodingArgs::from_matches(matches)?,
        })
    }
    pub(super) fn validate(&self) -> Result<MapLimits, CandidateError> {
        let prefill = self.prefill.limits()?;
        let encoding = self.encoding.policy()?;
        let (common, shared) = self.args.common()?;
        let m = &self.mapping;
        if !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes
                .is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
        {
            return Err(CandidateError::Arguments);
        }
        let mut task = self.args.budget(shared);
        // The semantic task covers every axis; serial physical execution
        // still reserves only one native cache and reuses it between heads.
        task.max_kv_bytes = task
            .max_kv_bytes
            .checked_mul(4)
            .ok_or(CandidateError::Arguments)?;
        let mapping = Int8SentimentMapLimits {
            chunks: ChunkLimits {
                max_input_bytes: self.args.max_input_bytes,
                max_chunk_bytes: m.max_chunk_bytes.unwrap_or(self.args.max_input_bytes),
                max_chunk_tokens: task.max_input_tokens as usize,
                context_tokens: self.args.context_tokens,
                reserved_tokens: self.args.max_candidate_tokens,
                max_chunks: m.max_chunks,
                max_tokenizer_calls: m.max_tokenizer_calls,
            },
            max_model_work: self.args.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes,
            aggregation: SentimentAggregationPolicy {
                minimum_estimated_coverage_ppm: m.minimum_estimated_coverage_ppm,
            },
        };
        let memory = Int8SentimentMapMemoryLimits {
            context_tokens: self.args.context_tokens,
            preparation_reserve_bytes: shared.preparation_bytes,
            allocator_reserve_bytes: 64 * MIB,
            max_axes: 4,
        }
        .requirements(task, self.args.sentiment_limits(), mapping)
        .and_then(|need| need.with_input_encoding(&encoding))
        .map_err(|_| CandidateError::Arguments)?;
        let envelope_bytes =
            usize::try_from(memory.max_output_bytes).map_err(|_| CandidateError::Arguments)?;
        // Session owns JSON parsing, the shared compiler and final external
        // serialization. The host independently owns every retained map plan.
        let cli_staging = (envelope_bytes as u64)
            .checked_mul(2)
            .and_then(|n| n.checked_add(128 * MIB + self.args.max_input_bytes as u64 * 32))
            .ok_or(CandidateError::Arguments)?;
        if cli_staging > shared.preparation_bytes {
            return Err(CandidateError::Arguments);
        }
        let extra = prefill
            .map(|limits| limits.validate())
            .transpose()
            .map_err(|_| CandidateError::Arguments)?
            .unwrap_or(0);
        let execution = checked_sum(&[
            memory.native.kv_bytes,
            memory.workspace_bytes,
            memory.input_bytes,
            memory.output_reservation_bytes,
            extra,
            65_536,
        ])?;
        let loading = 128 * MIB + 64 * MIB + 65_536;
        let minimum = checked_sum(&[
            shared.preparation_bytes,
            shared.weight_bytes,
            64 * MIB,
            execution.max(loading),
        ])?;
        if minimum > shared.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(MapLimits {
            common,
            shared,
            task,
            mapping,
            memory,
            encoding,
            envelope_bytes,
        })
    }
    pub(super) fn execute(
        self,
        input: &mut impl Read,
        output: &mut impl Write,
    ) -> Result<(), CandidateError> {
        let admitted = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        {
            runtime::sentiment_map::execute(self, admitted, input, output)
        }
        #[cfg(not(feature = "asupersync-runtime"))]
        {
            let _ = (self, admitted, input, output);
            Err(CandidateError::Unavailable)
        }
    }
}
fn checked_sum(values: &[u64]) -> Result<u64, CandidateError> {
    values.iter().try_fold(0_u64, |sum, &value| {
        sum.checked_add(value).ok_or(CandidateError::Arguments)
    })
}

#[cfg(test)]
mod tests;
