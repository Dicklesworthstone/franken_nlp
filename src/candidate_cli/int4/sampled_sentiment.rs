//! Explicit independent score/explanation samples, separate from distributions.
use super::*;
use crate::{
    candidate_cli::stream::Failure,
    grammar::{CompileLimits, mask::MaskWorkLimits},
    native_engine::{constrained_int4::json_sampling_workspace_bytes,
        portable_int8::ProjectionWork, strict_int4::Int4Work},
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, SentimentRequest,
        sampled::{SampledSentimentLimits, SampledSentimentOptions}}},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

const UNSUPPORTED: &[&str] = &[
    "max_output_bytes", "logprobs", "prefill_rows", "min_new_tokens", "stop_suffixes",
    "banned_token_ids", "repetition_penalty_milli", "presence_penalty_milli",
    "frequency_penalty_milli", "logit_bias",
];

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SampledArgs {
    /// Independent addressed generations per axis. Every sample must complete.
    #[arg(long, default_value_t = 4)]
    samples: usize,
    /// Maximum Unicode scalar count in each generated explanation.
    #[arg(long, default_value_t = 512)]
    max_justification_chars: usize,
    /// Complete JSON score/explanation bytes per sample, separate from report size.
    #[arg(long, default_value_t = 8192)]
    max_sample_json_bytes: usize,
    /// Complete aggregate result, including every sample and explanation.
    #[arg(long, default_value_t = 1_048_576)]
    max_result_bytes: usize,
    #[arg(long, default_value_t = 4096)]
    max_grammar_states: u32,
    #[arg(long, default_value_t = 2_000_000)]
    mask_step_node_visits: u64,
    /// One nonrenewable mask allowance across every axis, sample and EOS.
    #[arg(long, default_value_t = 32_000_000_000)]
    max_mask_node_visits: u64,
    #[arg(long, default_value_t = 131_072)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 1_000_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    max_attention_pairs: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)]
    max_multiply_accumulates: u64,
}

pub(super) fn definition() -> clap::Command {
    let mut command = Int4Args::augment_args(clap::Command::new("sentiment-sampled")
        .about("Generate independently sampled affect scores and explanations")
        .long_about("For each requested valence, arousal, dominance or approach axis, generate --samples independent addressed JSON scores in [-100,100] with bounded explanations. --seed is required. Every generation is constrained and independently validated; one failure returns no bundle. Results retain all samples and descriptive aggregates, not calibrated confidence or psychological measurements. Input is {document,axes?}; omitted axes selects all four. The complete prompt is preserved with control-free byte encoding. Work, masks and deadlines are shared across the whole bundle. Uses explicitly selected local sources or a digest-pinned snapshot. This candidate has no quality certification, sampling of distribution-mode labels, parse-repair retry, thinking, tools, cohorts or prompt schedule override."))
        .mut_arg("input", |arg| arg.help("Bounded sentiment JSON object; '-' reads stdin"))
        .mut_arg("seed", |arg| arg.required(true))
        .mut_arg("max_new_tokens", |arg| arg.default_value("256")
            .help("Per-sample constrained tokens INCLUDING terminal EOS"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"));
    for &id in UNSUPPORTED { command = command.mut_arg(id, |arg| arg.hide(true)); }
    SampledArgs::augment_args(command)
}

impl SampledArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if UNSUPPORTED.iter().any(|id| matches.value_source(id)
            == Some(clap::parser::ValueSource::CommandLine)) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "sampled sentiment accepts its finite JSON sampler, not free-text controls or schedules"));
        }
        Self::from_arg_matches(matches)
    }

    pub(super) fn options(self, args: &Int4Args) -> Result<SampledSentimentOptions, CandidateError> {
        let options = SampledSentimentOptions {
            effective_seed: parse_seed(args.seed.as_deref().ok_or(CandidateError::Arguments)?)?,
            samples: self.samples,
            temperature_milli: args.temperature_milli.unwrap_or(1000),
            top_k: args.top_k,
            top_p_ppm: args.top_p_ppm.unwrap_or(1_000_000),
            max_sample_tokens: args.max_new_tokens,
            max_justification_chars: self.max_justification_chars,
        };
        options.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(options)
    }

    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        self.options(args)?;
        self.task_limits(args).validate().map_err(|_| CandidateError::Arguments)?;
        if !(1..=32).contains(&self.samples) || !(1..=1024).contains(&args.max_new_tokens)
            || args.max_new_tokens >= args.context_tokens
            || !(1..=4096).contains(&self.max_justification_chars)
            || !(32..=65_536).contains(&self.max_sample_json_bytes)
            || !(1..=MAX_CONTENT_BYTES).contains(&self.max_result_bytes)
            || !(1..=65_536).contains(&self.max_grammar_states)
            || !(1..=1_000_000_000).contains(&self.mask_step_node_visits)
            || !(1..=1_000_000_000_000).contains(&self.max_mask_node_visits)
            || !(1..=16 * 1024 * 1024).contains(&self.max_forward_positions)
            || self.max_projected_logits == 0 || self.max_attention_pairs == 0
            || self.max_dot_products == 0 || self.max_multiply_accumulates == 0
            || args.prefill()?.is_some() || args.compute.threads() != 1
            || !args.input_encoding.is_default() || args.logprobs || args.max_output_bytes != 65_536 {
            return Err(CandidateError::Arguments);
        }
        // Defend internal construction as well as the parser. Sampling is the
        // sole opt-in generation policy; bans, biases, penalties and stops are
        // not silently accepted or applied to this finite output language.
        let configured = args.options(166_101)?;
        let mut allowed = GenerationOptions::greedy(args.max_new_tokens, 65_536, 166_101);
        allowed.sampling = configured.sampling.clone();
        if configured != allowed { return Err(CandidateError::Arguments); }
        let total_tokens = self.total_tokens(args, SentimentAxis::ALL.len())?;
        let sampler = json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?;
        let extra = sum(&[512 * MIB, self.max_forward_positions * 32,
            self.max_result_bytes as u64 * 16, u64::from(self.max_grammar_states) * 64,
            self.max_sample_json_bytes as u64 * 16])?;
        let mut limits = args.validate_storage(self.max_result_bytes, total_tokens,
            args.context_tokens - args.max_new_tokens + 1, sampler, extra)?;
        // Caller preparation remains charged while the host owns the sealed
        // plans, vocabulary and output. These are separate simultaneous owners.
        limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, limits.preparation_bytes])?;
        if limits.minimum_memory_bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(limits)
    }

    fn total_tokens(self, args: &Int4Args, axes: usize) -> Result<usize, CandidateError> {
        self.samples.checked_mul(axes).and_then(|count| count.checked_mul(args.max_new_tokens))
            .filter(|&tokens| tokens != 0 && tokens <= u32::MAX as usize).ok_or(CandidateError::Arguments)
    }
    pub(super) fn budget(self, args: &Int4Args, limits: Int4Limits, axes: usize)
        -> Result<TaskBudget, CandidateError> {
        Ok(TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: self.total_tokens(args, axes)? as u32,
            max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: self.max_grammar_states, max_kv_bytes: limits.native.kv_bytes })
    }
    pub(super) fn task_limits(self, args: &Int4Args) -> SampledSentimentLimits {
        SampledSentimentLimits { max_input_bytes: args.max_input_bytes,
            max_total_prompt_tokens: self.max_forward_positions as usize,
            max_sample_json_bytes: self.max_sample_json_bytes,
            max_result_bytes: self.max_result_bytes as u64,
            compile: CompileLimits { max_output_bytes: self.max_sample_json_bytes,
                max_states: self.max_grammar_states as usize, ..CompileLimits::default() } }
    }
    pub(super) fn work_ceiling(self) -> Int4Work {
        Int4Work { forward_positions: self.max_forward_positions,
            projected_logits: self.max_projected_logits, attention_pairs: self.max_attention_pairs,
            projections: ProjectionWork { dot_products: self.max_dot_products,
                multiply_accumulates: self.max_multiply_accumulates } }
    }
    pub(super) fn admit_plan(self, context: usize, capacity: usize, work: Int4Work)
        -> Result<(), CandidateError> {
        let cap = self.work_ceiling();
        if context == 0 || context > capacity || work.forward_positions > cap.forward_positions
            || work.projected_logits > cap.projected_logits || work.attention_pairs > cap.attention_pairs
            || !work.projections.fits(cap.projections) { return Err(CandidateError::Planning); }
        Ok(())
    }
    pub(super) fn masks(self) -> MaskWorkLimits {
        MaskWorkLimits { max_trie_node_visits: self.mask_step_node_visits as usize,
            checkpoint_interval_nodes: 256 }
    }
    pub(super) fn run(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write,
        diagnostics: &mut impl Write) -> ExitCode {
        match self.execute(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let _ = writeln!(diagnostics, "fnlp candidate sampled sentiment: {}", failure.code);
                failure.exit.as_process_exit()
            }
        }
    }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(Failure::usage("sampled_sentiment_profile_unavailable")) }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    document: String,
    #[serde(default = "all_axes")]
    axes: Vec<SentimentAxis>,
}
fn all_axes() -> Vec<SentimentAxis> { SentimentAxis::ALL.to_vec() }

fn parse_request(text: &str, budget: TaskBudget, cap: usize) -> Result<SentimentRequest, CandidateError> {
    if cap == 0 || cap > MAX_INPUT_BYTES || text.len() > cap { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 8, max_string_bytes: cap,
    }).map_err(|_| CandidateError::Input)?;
    let input: Input = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    if input.document.is_empty() || input.axes.is_empty() || input.axes.len() > SentimentAxis::ALL.len()
        || input.axes.iter().collect::<std::collections::BTreeSet<_>>().len() != input.axes.len() {
        return Err(CandidateError::Input);
    }
    Ok(SentimentRequest { document: input.document, axes: input.axes, budget })
}
