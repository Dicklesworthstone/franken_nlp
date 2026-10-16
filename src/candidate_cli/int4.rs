//! Explicit source-candidate INT4 tasks. Never an INT8 artifact alias.
use super::*;
use crate::native_engine::{int4::PackedInt4Kernel, simd::Int8KernelKind,
    strict_int4::{Int4MemoryRequirement, Int4ModelLoadRequirement, prefill::Int4PrefillLimits,
        weights::snapshot::{Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements}}};

mod classify;
mod prefill;
mod batch;
#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod snapshot_tests;

const ALLOCATOR_BYTES: u64 = 64 * MIB;
const CLEANUP_BYTES: u64 = 65_536;

#[derive(Args)]
struct Int4Args {
    /// UTF-8 prompt or JSON message array; '-' reads stdin.
    #[arg(default_value = "-")]
    input: PathBuf,
    /// Explicit pinned local HF source directory, not a .fnlpq artifact.
    #[arg(long, value_name = "DIR", required_unless_present = "snapshot", conflicts_with = "snapshot")]
    source_dir: Option<PathBuf>,
    /// Pinned ten-file conversion-source manifest, independently checked by the loader.
    #[arg(long, value_name = "FILE", required_unless_present = "snapshot", conflicts_with = "snapshot")]
    source_manifest: Option<PathBuf>,
    /// Reload a local candidate snapshot instead of reading or converting sources.
    #[arg(long, value_name = "FILE", requires = "expected_model_sha256",
        conflicts_with_all = ["source_dir", "source_manifest", "save_snapshot"])]
    snapshot: Option<PathBuf>,
    /// Independent model-content pin from a trusted conversion; never read from the snapshot itself.
    #[arg(long, value_name = "HEX", requires = "snapshot")]
    expected_model_sha256: Option<String>,
    /// Save the converted model to a NEW local snapshot. Never overwrites an existing path.
    /// A saved file remains if the later task fails; partial files are also preserved on error.
    #[arg(long, value_name = "NEW_FILE", conflicts_with = "snapshot")]
    save_snapshot: Option<PathBuf>,
    /// Process memory-ledger ceiling in MiB; not an operating-system RSS cap.
    #[arg(long, value_name = "MIB")]
    memory_mib: u64,
    /// Contiguous group width; the exact partition enters model identity and work.
    #[arg(long, default_value_t = 128)]
    group_width: usize,
    /// Exact integer kernel ID, hardware checked before IO. Unsupported requests never fall back.
    /// Examples: s_scalar_i32, x3b_avx2_widened_i16, x2_avx_vnni, a2_dotprod.
    #[arg(long, value_name = "ID", default_value_t = Int8KernelKind::Scalar)]
    kernel: Int8KernelKind,
    #[arg(long, default_value_t = 2048)]
    context_tokens: usize,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 65_536)]
    max_input_bytes: usize,
    /// Generated text bytes; complete JSON/result staging is additionally admitted.
    #[arg(long, default_value_t = 65_536)]
    max_output_bytes: usize,
    /// Maximum complete weight-plus-load/save-scratch payload, not per tensor.
    #[arg(long, default_value_t = 6144)]
    max_load_mib: u64,
    /// Source-conversion metadata/hash allowance, at least 128 MiB; not reserved for snapshot reload.
    #[arg(long, default_value_t = 128)]
    source_preparation_mib: u64,
    /// Input, pinned tokenizer, task planning and publication staging allowance.
    #[arg(long, default_value_t = 256)]
    preparation_mib: u64,
    /// Cooperative total deadline, including reads, source verification and conversion.
    /// Blocking IO is not preempted; the remaining budget is checked between stages.
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
    /// Source verification/conversion checkpoints, separate from persistence and execution.
    #[arg(long, default_value_t = 1_000_000_000)]
    load_checkpoints: u64,
    /// Checkpoints for the single snapshot save OR reload stage; never replenishes execution.
    #[arg(long, default_value_t = 1_000_000_000)]
    snapshot_checkpoints: u64,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Int4Task { Text(Task), Classify(classify::ScoringArgs) }
#[derive(Clone, Copy)]
enum Int4BatchCommand { Fixed(batch::BatchArgs), Corpus(batch::corpus::CorpusArgs) }
impl Int4BatchCommand {
    fn task(self) -> Task {
        match self { Self::Fixed(args) => args.task(), Self::Corpus(args) => args.task() }
    }
    fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        match self { Self::Fixed(batch) => batch.execute(args, input, output),
            Self::Corpus(corpus) => corpus.execute(args, input, output) }
    }
}
pub(super) struct Int4Command { task: Int4Task, args: Int4Args, batch: Option<Int4BatchCommand> }

pub(super) fn definition() -> clap::Command {
    let task = |name: &'static str, about: &'static str| Int4Args::augment_args(clap::Command::new(name).about(about))
        .mut_arg("prefill_rows", |arg| arg.help("Explicit grouped-INT4 layer-major prompt execution, 1..=64 rows; extra workspace is process-admitted"));
    clap::Command::new("source-int4")
        .about("Run candidate INT4 tasks from pinned sources or a digest-pinned local snapshot")
        .long_about("Non-certified grouped INT4 candidate inference on the existing resource host. Select --source-dir plus --source-manifest, optionally with --save-snapshot NEW_FILE, or select --snapshot FILE plus an independent --expected-model-sha256. Always requires --memory-mib. Sequential native execution defaults to scalar; --kernel explicitly selects an existing hardware-checked integer leaf. Text tasks can opt in to bounded layer-major prompt execution with --prefill-rows; classification cannot. text-batch runs one finite NDJSON cohort and emits one completed aggregate JSON object. text-corpus processes bounded windows with one resident model and cumulative budgets, emitting completed-window NDJSON with a required corpus_complete footer. No automatic selection, fallback, download, active catalog, thinking mode or tools. Snapshots are local candidate storage, not canonical FNLPQ or release activation. Outputs are completed results, not provisional token events.")
        .subcommand_required(true)
        .subcommand(task("generate", "Generate from a bounded UTF-8 prompt using the pinned template"))
        .subcommand(task("chat", "Complete a bounded JSON system/user/assistant transcript"))
        .subcommand(classify::definition())
        .subcommand(batch::definition())
        .subcommand(batch::corpus::definition())
}
impl Int4Command {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "source-int4 task required"))?;
        let batch = match name {
            "text-batch" => Some(Int4BatchCommand::Fixed(batch::BatchArgs::from_arg_matches(matches)?)),
            "text-corpus" => Some(Int4BatchCommand::Corpus(batch::corpus::CorpusArgs::from_arg_matches(matches)?)),
            _ => None,
        };
        if let Some(batch) = batch {
            return Ok(Self { task: Int4Task::Text(batch.task()), args: Int4Args::from_arg_matches(matches)?, batch: Some(batch) });
        }
        let task = match name {
            "generate" => Int4Task::Text(Task::Generate),
            "chat" => Int4Task::Text(Task::Chat),
            "classify" => Int4Task::Classify(classify::ScoringArgs::from_matches(matches)?),
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "source-int4 task refused")),
        };
        Ok(Self { task, args: Int4Args::from_arg_matches(matches)?, batch: None })
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        if let Some(batch) = self.batch {
            if self.task != Int4Task::Text(batch.task()) { return Err(CandidateError::Arguments); }
            return batch.execute(self.args, input, output);
        }
        let limits = match self.task {
            Int4Task::Text(_) => self.args.validate()?,
            Int4Task::Classify(scoring) => scoring.validate(&self.args)?,
        };
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute_task(self.task, self.args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

#[derive(Clone, Copy, Debug)]
struct Int4Limits {
    memory_bytes: u64, preparation_bytes: u64, source_preparation_bytes: u64,
    max_load_bytes: u64, result_bytes: usize, max_prompt_tokens: usize,
    load: Int4ModelLoadRequirement, native: Int4MemoryRequirement,
    snapshot_pin: Option<Int4SnapshotPin>, snapshot_need: Int4SnapshotRequirements,
    /// Real-host-checked executable handle, not a serialized capability report.
    kernel: PackedInt4Kernel,
    /// Explicit physical text-prompt schedule; independently priced in the execution stage.
    prefill: Option<Int4PrefillLimits>,
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
        if self.max_new_tokens == 0 || self.max_new_tokens > 1024 || self.max_new_tokens > self.context_tokens
            || self.max_output_bytes == 0 || self.max_output_bytes > MAX_CONTENT_BYTES {
            return Err(CandidateError::Arguments);
        }
        let result_bytes = self.max_output_bytes.checked_mul(8)
            .and_then(|n| n.checked_add(self.max_new_tokens.checked_mul(64)?))
            .and_then(|n| n.checked_add(65_536)).ok_or(CandidateError::Arguments)?;
        let limits = self.validate_storage(result_bytes, self.max_new_tokens,
            self.context_tokens - self.max_new_tokens + 1, SAMPLER_BYTES, 0)?;
        let options = self.options(166_101)?;
        if options.banned_token_ids.contains(&166_101) { return Err(CandidateError::Arguments); }
        options.validate(self.generation_limits(limits)).map_err(|_| CandidateError::Arguments)?;
        Ok(limits)
    }
    // Common model/storage arithmetic only. Scored tasks provide their actual
    // complete-result depth/bytes and repeated-prompt allowance, with no sampler.
    fn validate_storage(&self, result_bytes: usize, output_tokens: usize, max_prompt_tokens: usize,
        sampler_bytes: u64, extra_preparation_bytes: u64) -> Result<Int4Limits, CandidateError> {
        let snapshot_pin = self.checked_model_input()?;
        let prefill = self.prefill()?;
        // Scored tasks declare no sampler; prefill is not silently inherited
        // through their shared parser, including programmatic construction.
        if sampler_bytes == 0 && prefill.is_some() { return Err(CandidateError::Arguments); }
        let extra_scratch = prefill.map(|limits| limits.max_extra_scratch_bytes).unwrap_or(0);
        let kernel = PackedInt4Kernel::detect(self.kernel).map_err(|_| CandidateError::Arguments)?;
        if self.input.as_os_str().is_empty()
            || self.max_input_bytes == 0 || self.max_input_bytes > MAX_INPUT_BYTES
            || max_prompt_tokens == 0 || max_prompt_tokens > self.context_tokens
            || self.timeout_seconds == 0 || self.timeout_seconds > 86_400
            || !(2..=1_000_000_000_000).contains(&self.load_checkpoints)
            || !(2..=1_000_000_000_000).contains(&self.snapshot_checkpoints)
            || !(2..=1_000_000_000_000).contains(&self.max_checkpoints) {
            return Err(CandidateError::Arguments);
        }
        let load = Int4ModelLoadRequirement::for_group_width(self.group_width).map_err(|_| CandidateError::Arguments)?;
        let native = Int4MemoryRequirement::for_context(self.context_tokens, self.group_width)
            .map_err(|_| CandidateError::Arguments)?;
        let memory_bytes = mib(self.memory_mib)?;
        let preparation_bytes = mib(self.preparation_mib)?;
        let source_preparation_bytes = mib(self.source_preparation_mib)?;
        let max_load_bytes = mib(self.max_load_mib)?;
        let snapshot_need = Int4SnapshotRequirements::for_group_width(self.group_width)
            .map_err(|_| CandidateError::Arguments)?;
        let load_scratch = if snapshot_pin.is_some() { snapshot_need.scratch_bytes } else { load.scratch_payload_bytes };
        let peak_codec_scratch = if self.save_snapshot.is_some() { load_scratch.max(snapshot_need.scratch_bytes) } else { load_scratch };
        if source_preparation_bytes < 128 * MIB
            || sum(&[load.weight_payload_bytes, peak_codec_scratch])? > max_load_bytes {
            return Err(CandidateError::Arguments);
        }
        let preparation_floor = sum(&[128 * MIB, self.max_input_bytes as u64 * 32,
            (result_bytes as u64 + 4096) * 2, extra_preparation_bytes])?;
        if preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        let resident = sum(&[load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let load_stage = if snapshot_pin.is_some() { snapshot_need.scratch_bytes }
            else { sum(&[source_preparation_bytes, load.scratch_payload_bytes])? };
        let save_stage = if self.save_snapshot.is_some() { snapshot_need.scratch_bytes } else { 0 };
        // Matches the hosted output charge: four complete result payloads,
        // token/score vectors, and fixed bookkeeping. The CLI additionally
        // retains its own separately charged serialization staging.
        let output_charge = sum(&[result_bytes as u64 * 4, output_tokens as u64 * 8, 4096])?;
        let execute_stage = sum(&[native.kv_bytes, native.rope_bytes, native.scratch_payload_bound,
            sampler_bytes, ALLOCATOR_BYTES, output_charge, extra_scratch])?;
        let minimum_memory_bytes = sum(&[preparation_bytes, resident,
            load_stage.max(save_stage).max(execute_stage), CLEANUP_BYTES])?;
        if minimum_memory_bytes > memory_bytes { return Err(CandidateError::Arguments); }
        Ok(Int4Limits { memory_bytes, preparation_bytes, source_preparation_bytes, max_load_bytes,
            result_bytes, max_prompt_tokens, load, native, snapshot_pin, snapshot_need, kernel, prefill, minimum_memory_bytes })
    }
    // Recheck the exclusive source/snapshot contract for programmatically
    // constructed arguments too, before input IO or caller-sized allocations.
    fn checked_model_input(&self) -> Result<Option<Int4SnapshotPin>, CandidateError> {
        let valid_path = |p: &PathBuf| !p.as_os_str().is_empty() && p.as_os_str() != "-";
        if self.save_snapshot.as_ref().is_some_and(|p| !valid_path(p)) { return Err(CandidateError::Arguments); }
        match (&self.source_dir, &self.source_manifest, &self.snapshot, &self.expected_model_sha256) {
            (Some(directory), Some(manifest), None, None) if valid_path(directory) && valid_path(manifest) => Ok(None),
            (None, None, Some(path), Some(expected)) if valid_path(path) && self.save_snapshot.is_none() => {
                // Same exact lowercase 32-byte hex parser as seeds; this value
                // is a content expectation, never sampling entropy or a file-derived identity.
                Ok(Some(Int4SnapshotPin { group_width: self.group_width, model_content_sha256: parse_seed(expected)? }))
            }
            _ => Err(CandidateError::Arguments),
        }
    }
    fn generation_limits(&self, limits: Int4Limits) -> GenerationLimits {
        GenerationLimits { max_prompt_tokens: limits.max_prompt_tokens, max_new_tokens: self.max_new_tokens,
            max_output_bytes: self.max_output_bytes, max_sampler_bytes: SAMPLER_BYTES }
    }
    fn options(&self, eos: u32) -> Result<GenerationOptions, CandidateError> {
        self.prefill()?;
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

impl Int4Limits {
    fn snapshot_budget(self, reading: bool) -> Result<Int4SnapshotBudget, CandidateError> {
        let need = self.snapshot_need;
        Ok(Int4SnapshotBudget { max_weight_bytes: need.weight_payload_bytes, max_scratch_bytes: need.scratch_bytes,
            max_io_bytes: if reading { need.read_io_bytes().map_err(|_| CandidateError::Arguments)? } else { need.file_bytes } })
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
