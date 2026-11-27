//! Independent INT8 score/justification samples with complete-bundle admission.
use super::*;
use crate::{
    candidate_cli::stream::Failure,
    grammar::{CompileLimits, mask::MaskWorkLimits},
    native_engine::{constrained_int8::json_sampling_workspace_bytes,
        portable_int8::ProjectionWork, strict_int8::{Int8Work, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, SentimentRequest,
        sampled::{SampledSentimentOptions, SampledSentimentLimits}}},
};

#[cfg(feature = "asupersync-runtime")]
use super::runtime::sampled_sentiment as runtime;
#[cfg(test)]
mod tests;

#[derive(Args)]
pub(super) struct SampledSentimentCommand {
    /// Sentiment JSON object {document,axes?}; '-' reads stdin.
    #[arg(default_value = "-")]
    input: PathBuf,
    /// Explicit local current-candidate INT8 artifact; no discovery or download.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,
    /// Process memory ledger in MiB; not an operating-system RSS cap.
    #[arg(long)]
    memory_mib: u64,
    #[arg(long, default_value_t = 2048)]
    context_tokens: usize,
    /// Maximum tokens per independent sample INCLUDING scored terminal EOS.
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 65_536)]
    max_input_bytes: usize,
    #[arg(long, default_value_t = 6144)]
    max_weight_mib: u64,
    /// Caller and transferred plan/vocabulary ownership are charged separately.
    #[arg(long, default_value_t = 1024)]
    preparation_mib: u64,
    /// One cooperative invocation deadline, including preparation and delivery.
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
    /// Native checkpoints shared by every axis and sample, never renewed.
    #[arg(long, default_value_t = 1_000_000_000)]
    max_checkpoints: u64,
    /// Exactly 64 lowercase hexadecimal digits. No implicit entropy is drawn.
    #[arg(long, value_name = "HEX")]
    seed: String,
    #[arg(long, default_value_t = 1000)]
    temperature_milli: u32,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 1_000_000)]
    top_p_ppm: u32,
    /// Independent addressed generations per axis; all must complete.
    #[arg(long, default_value_t = 4)]
    samples: usize,
    #[arg(long, default_value_t = 512)]
    max_justification_chars: usize,
    #[arg(long, default_value_t = 8192)]
    max_sample_json_bytes: usize,
    /// Whole result including every sample and explanation, before provenance.
    #[arg(long, default_value_t = 1_048_576)]
    max_result_bytes: usize,
    #[arg(long, default_value_t = 4096)]
    max_grammar_states: u32,
    #[arg(long, default_value_t = 2_000_000)]
    mask_step_node_visits: u64,
    /// One nonrenewable mask allowance across all samples, including EOS.
    #[arg(long, default_value_t = 32_000_000_000)]
    pub(in crate::candidate_cli) max_mask_node_visits: u64,
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
    /// Optional layer-major prompt morsels, 1..=64 tokens. Each sample keeps fresh KV.
    #[arg(long, value_name = "ROWS")]
    prefill_rows: Option<usize>,
    #[arg(skip)]
    encoding: input_encoding::InputEncodingArgs,
}

pub(super) fn definition() -> clap::Command {
    input_encoding::definition(SampledSentimentCommand::augment_args(
        clap::Command::new("sentiment-sampled")
            .about("Generate independent INT8 affect scores with bounded explanations")
            .long_about("For each requested valence, arousal, dominance or approach axis, generate independently addressed JSON scores in [-100,100] with bounded explanations. --seed is mandatory. Input is {document,axes?}; omitted axes selects all four. Every sample must complete and pass independent schema/content checks before one complete result is published. Results retain all scores, explanations and descriptive aggregates; generated variation is not calibrated confidence and explanations are unverified. The complete axis-by-sample work, mask, sampler, preparation and output envelope is admitted before execution. --input-encoding safe-bpe-v1 explicitly compacts lossless untrusted input under finite tokenizer limits; it may change model output. --prefill-rows changes only bounded prompt scheduling. Each sample starts with empty logical KV. This explicit local candidate has no model-quality, performance or release certification. Distribution-mode sentiment remains a separate command.")))
}

impl SampledSentimentCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        Ok(command)
    }
    pub(in crate::candidate_cli) fn options(&self) -> Result<SampledSentimentOptions, CandidateError> {
        let options = SampledSentimentOptions { effective_seed: parse_seed(&self.seed)?,
            samples: self.samples, temperature_milli: self.temperature_milli,
            top_k: self.top_k, top_p_ppm: self.top_p_ppm,
            max_sample_tokens: self.max_new_tokens,
            max_justification_chars: self.max_justification_chars };
        options.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(options)
    }
    pub(in crate::candidate_cli) fn prefill(&self) -> Result<Option<Int8PrefillLimits>, CandidateError> {
        self.prefill_rows.map(|rows| {
            let bytes = Int8PrefillLimits::required_extra_scratch_bytes(rows)
                .map_err(|_| CandidateError::Arguments)?;
            Ok(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes })
        }).transpose()
    }
    fn total_tokens(&self, axes: usize) -> Result<usize, CandidateError> {
        if axes == 0 || axes > SentimentAxis::ALL.len() { return Err(CandidateError::Arguments); }
        self.samples.checked_mul(axes).and_then(|n| n.checked_mul(self.max_new_tokens))
            .filter(|&n| n != 0 && n <= u32::MAX as usize).ok_or(CandidateError::Arguments)
    }
    pub(in crate::candidate_cli) fn task_limits(&self) -> SampledSentimentLimits {
        SampledSentimentLimits { max_input_bytes: self.max_input_bytes,
            max_total_prompt_tokens: self.max_forward_positions as usize,
            max_sample_json_bytes: self.max_sample_json_bytes,
            max_result_bytes: self.max_result_bytes as u64,
            compile: CompileLimits { max_output_bytes: self.max_sample_json_bytes,
                max_states: self.max_grammar_states as usize, ..CompileLimits::default() } }
    }
    pub(in crate::candidate_cli) fn budget(&self, limits: Limits, axes: usize) -> Result<TaskBudget, CandidateError> {
        Ok(TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: self.total_tokens(axes)? as u32,
            max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: self.max_grammar_states, max_kv_bytes: limits.kv_bytes })
    }
    pub(in crate::candidate_cli) fn masks(&self) -> MaskWorkLimits {
        MaskWorkLimits { max_trie_node_visits: self.mask_step_node_visits as usize,
            checkpoint_interval_nodes: 256 }
    }
    pub(in crate::candidate_cli) fn work_ceiling(&self) -> Int8Work {
        Int8Work { forward_positions: self.max_forward_positions,
            projected_logits: self.max_projected_logits, attention_pairs: self.max_attention_pairs,
            projections: ProjectionWork { dot_products: self.max_dot_products,
                multiply_accumulates: self.max_multiply_accumulates } }
    }
    pub(in crate::candidate_cli) fn admit_plan(&self, context: usize, work: Int8Work) -> Result<(), CandidateError> {
        let cap = self.work_ceiling();
        if context == 0 || context > self.context_tokens
            || work.forward_positions > cap.forward_positions
            || work.projected_logits > cap.projected_logits || work.attention_pairs > cap.attention_pairs
            || !work.projections.fits(cap.projections) { return Err(CandidateError::Planning); }
        Ok(())
    }
    fn common(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        self.options()?;
        if self.max_new_tokens >= self.context_tokens
            || !(1..=MAX_CONTENT_BYTES).contains(&self.max_result_bytes)
            || !(1..=65_536).contains(&self.max_grammar_states)
            || !(1..=1_000_000_000).contains(&self.mask_step_node_visits)
            || !(1..=1_000_000_000_000).contains(&self.max_mask_node_visits)
            || !(1..=16 * 1024 * 1024).contains(&self.max_forward_positions)
            || self.max_projected_logits == 0 || self.max_attention_pairs == 0
            || self.max_dot_products == 0 || self.max_multiply_accumulates == 0 {
            return Err(CandidateError::Arguments);
        }
        self.task_limits().validate().map_err(|_| CandidateError::Arguments)?;
        let prefill = self.prefill()?;
        // Reuse the existing host, checked model loader and input policy; no
        // unrelated free-generation processor is exposed by this command.
        let common = CandidateArgs { input: self.input.clone(), model: self.model.clone(),
            memory_mib: self.memory_mib, context_tokens: self.context_tokens,
            max_new_tokens: self.max_new_tokens, max_output_bytes: self.max_result_bytes,
            max_input_bytes: self.max_input_bytes, max_weight_mib: self.max_weight_mib,
            preparation_mib: self.preparation_mib, timeout_seconds: self.timeout_seconds,
            max_checkpoints: self.max_checkpoints, seed: Some(self.seed.clone()),
            temperature_milli: Some(self.temperature_milli), top_k: self.top_k,
            top_p_ppm: Some(self.top_p_ppm), logprobs: false,
            policy: generation_policy::GenerationPolicyArgs::default(),
            compute: compute::Args::default(), input_encoding: self.encoding.clone() };
        let mut limits = common.validate()?;
        limits.result_bytes = self.max_result_bytes;
        limits.compute = common.compute.admit_prefill(prefill)?;
        let encoder = self.encoding.workspace_bytes(self.max_input_bytes, limits.max_prompt_tokens)?;
        let preparation_floor = [512 * MIB, self.max_forward_positions.checked_mul(32)
            .ok_or(CandidateError::Arguments)?, self.max_input_bytes as u64 * 32,
            self.max_result_bytes as u64 * 16, u64::from(self.max_grammar_states) * 64,
            self.max_sample_json_bytes as u64 * 16, encoder]
            .into_iter().try_fold(0_u64, |total, bytes| total.checked_add(bytes))
            .ok_or(CandidateError::Arguments)?;
        if preparation_floor > limits.preparation_bytes { return Err(CandidateError::Arguments); }
        let native = Int8MemoryRequirement::for_context(self.context_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        let sampler = json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?;
        // Price both simultaneous preparation owners, full resident KV and
        // sampler, whole-bundle output and declared prompt workspace before IO.
        let peak = compute::request_peak_bytes(limits, native, self.max_result_bytes as u64,
            self.total_tokens(SentimentAxis::ALL.len())? as u64, sampler, limits.preparation_bytes)?
            .checked_add(prefill.map(|p| p.max_extra_scratch_bytes).unwrap_or(0))
            .ok_or(CandidateError::Arguments)?;
        if peak > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok((common, limits))
    }
    pub(super) fn run(self, input: &mut impl Read, output: &mut impl Write,
        diagnostics: &mut impl Write) -> ExitCode {
        match self.execute(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_failure(failure, diagnostics),
        }
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
        let (common, limits) = self.common()
            .map_err(|_| Failure::usage("int8_sampled_sentiment_options"))?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(self, common, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, input, output);
            Err(Failure::usage("int8_sampled_sentiment_profile_unavailable")) }
    }
}

fn report_failure(failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    // The stable code and fixed cancellation enum retain the native terminal
    // reason without exposing parser, path, prompt or generated diagnostics.
    match failure.cancellation {
        Some(cause) => {
            let _ = writeln!(diagnostics, "fnlp candidate INT8 sampled sentiment: {} ({cause:?})", failure.code);
        }
        None => { let _ = writeln!(diagnostics, "fnlp candidate INT8 sampled sentiment: {}", failure.code); }
    }
    failure.exit.as_process_exit()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    document: String,
    #[serde(default = "all_axes")]
    axes: Vec<SentimentAxis>,
}
fn all_axes() -> Vec<SentimentAxis> { SentimentAxis::ALL.to_vec() }
pub(in crate::candidate_cli) fn request(text: &str, budget: TaskBudget, cap: usize)
    -> Result<SentimentRequest, CandidateError> {
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
