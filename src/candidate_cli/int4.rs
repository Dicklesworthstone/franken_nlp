//! Explicit source-candidate INT4 text/chat. Never an INT8 artifact alias.
use super::*;
use crate::native_engine::strict_int4::{Int4MemoryRequirement, Int4ModelLoadRequirement};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

const ALLOCATOR_BYTES: u64 = 64 * MIB;
const CLEANUP_BYTES: u64 = 65_536;

#[derive(Args)]
struct Int4Args {
    /// UTF-8 prompt or JSON message array; '-' reads stdin.
    #[arg(default_value = "-")]
    input: PathBuf,
    /// Explicit pinned local HF source directory, not a .fnlpq artifact.
    #[arg(long, value_name = "DIR")]
    source_dir: PathBuf,
    /// Pinned ten-file conversion-source manifest, independently checked by the loader.
    #[arg(long, value_name = "FILE")]
    source_manifest: PathBuf,
    /// Process memory-ledger ceiling in MiB; not an operating-system RSS cap.
    #[arg(long, value_name = "MIB")]
    memory_mib: u64,
    /// Contiguous group width; the exact partition enters model identity and work.
    #[arg(long, default_value_t = 128)]
    group_width: usize,
    #[arg(long, default_value_t = 2048)]
    context_tokens: usize,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 65_536)]
    max_input_bytes: usize,
    /// Generated text bytes; complete JSON/result staging is additionally admitted.
    #[arg(long, default_value_t = 65_536)]
    max_output_bytes: usize,
    /// Maximum complete weight-plus-conversion-scratch payload, not per tensor.
    #[arg(long, default_value_t = 6144)]
    max_load_mib: u64,
    /// Source metadata/snapshots/hash-workspace allowance, at least 128 MiB.
    #[arg(long, default_value_t = 128)]
    source_preparation_mib: u64,
    /// Input, pinned tokenizer, task planning and publication staging allowance.
    #[arg(long, default_value_t = 256)]
    preparation_mib: u64,
    /// Cooperative total deadline, including reads, source verification and conversion.
    /// Blocking IO is not preempted; the remaining budget is checked between stages.
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000_000)]
    load_checkpoints: u64,
    #[arg(long, default_value_t = 1_000_000_000)]
    max_checkpoints: u64,
    /// Exactly 64 lowercase hexadecimal digits; absent means greedy.
    #[arg(long, value_name = "HEX")]
    seed: Option<String>,
    #[arg(long, requires = "seed")]
    temperature_milli: Option<u32>,
    #[arg(long, requires = "seed")]
    top_k: Option<usize>,
    #[arg(long, requires = "seed")]
    top_p_ppm: Option<u32>,
    #[arg(long)]
    logprobs: bool,
    #[command(flatten)]
    policy: generation_policy::GenerationPolicyArgs,
}

pub(super) struct Int4Command { task: Task, args: Int4Args }

pub(super) fn definition() -> clap::Command {
    let task = |name: &'static str, about: &'static str| Int4Args::augment_args(clap::Command::new(name).about(about))
        // This scheduling option belongs to INT8 only. Keep the parser's field
        // available to the shared policy type, but never advertise or ignore it.
        .mut_arg("prefill_rows", |arg| arg.hide(true));
    clap::Command::new("source-int4")
        .about("Convert an explicit pinned source in memory and run candidate INT4 text/chat")
        .long_about("Non-certified grouped INT4 source-candidate inference on the existing resource host. Requires explicit --source-dir, --source-manifest and --memory-mib. Uses scalar sequential prefill/decode; no fallback, download, active catalog, persistent conversion, thinking mode or tools. Source verification/conversion repeats for each invocation. Emits one completed JSON object, not provisional token events.")
        .subcommand_required(true)
        .subcommand(task("generate", "Generate from a bounded UTF-8 prompt using the pinned template"))
        .subcommand(task("chat", "Complete a bounded JSON system/user/assistant transcript"))
}
impl Int4Command {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "source-int4 task required"))?;
        let task = match name { "generate" => Task::Generate, "chat" => Task::Chat,
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "source-int4 task refused")) };
        Ok(Self { task, args: Int4Args::from_arg_matches(matches)? })
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let limits = self.args.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(self.task, self.args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

#[derive(Clone, Copy, Debug)]
struct Int4Limits {
    memory_bytes: u64, preparation_bytes: u64, source_preparation_bytes: u64,
    max_load_bytes: u64, result_bytes: usize, max_prompt_tokens: usize,
    load: Int4ModelLoadRequirement, native: Int4MemoryRequirement,
    /// Lower bound for these declared simultaneous reservations. Host admission
    /// independently enforces all runtime/domain obligations; this is not RSS.
    minimum_memory_bytes: u64,
}
fn sum(values: &[u64]) -> Result<u64, CandidateError> {
    values.iter().try_fold(0_u64, |total, value| total.checked_add(*value).ok_or(CandidateError::Arguments))
}
fn mib(value: u64) -> Result<u64, CandidateError> {
    value.checked_mul(MIB).filter(|&bytes| bytes != 0).ok_or(CandidateError::Arguments)
}
impl Int4Args {
    fn validate(&self) -> Result<Int4Limits, CandidateError> {
        if self.input.as_os_str().is_empty() || self.source_dir.as_os_str().is_empty()
            || self.source_manifest.as_os_str().is_empty()
            || self.max_input_bytes == 0 || self.max_input_bytes > MAX_INPUT_BYTES
            || self.max_output_bytes == 0 || self.max_output_bytes > MAX_CONTENT_BYTES
            || self.max_new_tokens == 0 || self.max_new_tokens > 1024 || self.max_new_tokens > self.context_tokens
            || self.timeout_seconds == 0 || self.timeout_seconds > 86_400
            || !(2..=1_000_000_000_000).contains(&self.load_checkpoints)
            || !(2..=1_000_000_000_000).contains(&self.max_checkpoints)
            || self.policy.prefill()?.is_some() {
            return Err(CandidateError::Arguments);
        }
        let load = Int4ModelLoadRequirement::for_group_width(self.group_width).map_err(|_| CandidateError::Arguments)?;
        let native = Int4MemoryRequirement::for_context(self.context_tokens, self.group_width)
            .map_err(|_| CandidateError::Arguments)?;
        let memory_bytes = mib(self.memory_mib)?;
        let preparation_bytes = mib(self.preparation_mib)?;
        let source_preparation_bytes = mib(self.source_preparation_mib)?;
        let max_load_bytes = mib(self.max_load_mib)?;
        if source_preparation_bytes < 128 * MIB
            || load.peak_payload_bytes().map_err(|_| CandidateError::Arguments)? > max_load_bytes {
            return Err(CandidateError::Arguments);
        }
        let result_bytes = self.max_output_bytes.checked_mul(8)
            .and_then(|n| n.checked_add(self.max_new_tokens.checked_mul(64)?))
            .and_then(|n| n.checked_add(65_536)).ok_or(CandidateError::Arguments)?;
        let preparation_floor = sum(&[128 * MIB, self.max_input_bytes as u64 * 32,
            (result_bytes as u64 + 4096) * 2])?;
        if preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        let resident = sum(&[load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let load_stage = sum(&[source_preparation_bytes, load.scratch_payload_bytes])?;
        // Matches the hosted output charge: four complete result payloads,
        // token/score vectors, and fixed bookkeeping. The CLI additionally
        // retains its own separately charged serialization staging.
        let output_charge = sum(&[result_bytes as u64 * 4, self.max_new_tokens as u64 * 8, 4096])?;
        let execute_stage = sum(&[native.kv_bytes, native.rope_bytes, native.scratch_payload_bound,
            SAMPLER_BYTES, ALLOCATOR_BYTES, output_charge])?;
        let minimum_memory_bytes = sum(&[preparation_bytes, resident,
            load_stage.max(execute_stage), CLEANUP_BYTES])?;
        if minimum_memory_bytes > memory_bytes { return Err(CandidateError::Arguments); }
        let limits = Int4Limits { memory_bytes, preparation_bytes, source_preparation_bytes, max_load_bytes,
            result_bytes, max_prompt_tokens: self.context_tokens - self.max_new_tokens + 1,
            load, native, minimum_memory_bytes };
        let options = self.options(166_101)?;
        if options.banned_token_ids.contains(&166_101) { return Err(CandidateError::Arguments); }
        options.validate(self.generation_limits(limits)).map_err(|_| CandidateError::Arguments)?;
        Ok(limits)
    }
    fn generation_limits(&self, limits: Int4Limits) -> GenerationLimits {
        GenerationLimits { max_prompt_tokens: limits.max_prompt_tokens, max_new_tokens: self.max_new_tokens,
            max_output_bytes: self.max_output_bytes, max_sampler_bytes: SAMPLER_BYTES }
    }
    fn options(&self, eos: u32) -> Result<GenerationOptions, CandidateError> {
        if self.policy.prefill()?.is_some() { return Err(CandidateError::Arguments); }
        let mut options = GenerationOptions::greedy(self.max_new_tokens, self.max_output_bytes, eos);
        self.policy.apply(&mut options)?;
        options.capture_logprobs = self.logprobs;
        options.sampling = match &self.seed {
            Some(seed) => GenerationSampling::Seeded { effective_seed: parse_seed(seed)?,
                temperature_milli: self.temperature_milli.unwrap_or(1000), top_k: self.top_k,
                top_p_ppm: self.top_p_ppm.unwrap_or(1_000_000) },
            None => {
                if self.temperature_milli.is_some() || self.top_k.is_some() || self.top_p_ppm.is_some() {
                    return Err(CandidateError::Arguments);
                }
                GenerationSampling::Greedy
            }
        };
        Ok(options)
    }
}

// Intentionally no Debug: these values contain private user text.
enum TextInput { Generate(String), Chat(Vec<ChatMessage>) }
fn parse_input(task: Task, text: String, cap: usize) -> Result<TextInput, CandidateError> {
    if text.len() > cap { return Err(CandidateError::Input); }
    if task == Task::Generate { return Ok(TextInput::Generate(text)); }
    let messages = parse_messages(&text, cap)?;
    // Refuse malformed role order BEFORE the expensive source conversion, not
    // just during the later model-bound token compiler. Never repair history.
    let mut expected = ChatRole::User;
    for (index, message) in messages.iter().enumerate() {
        if index == 0 && message.role == ChatRole::System { continue; }
        if message.role != expected { return Err(CandidateError::Input); }
        expected = if expected == ChatRole::User { ChatRole::Assistant } else { ChatRole::User };
    }
    Ok(TextInput::Chat(messages))
}
