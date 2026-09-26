//! Explicit local-candidate inference, separate from release/catalog activation.
//!
//! Nothing here selects a model implicitly, downloads data, executes tools, or
//! upgrades the candidate loader's evidence grade. The ordinary hosted runtime
//! owns model/compute admission. CLI preparation is separately charged before
//! input, metadata, tokenizer or output staging allocations.

#![cfg_attr(not(feature = "asupersync-runtime"), allow(dead_code))]

use std::{io::{self, Read, Write}, path::PathBuf, process::ExitCode};
use clap::{Args, FromArgMatches};
use serde::Serialize;
use crate::{canonjson, error::ErrorCode,
    native_engine::{generation::{GenerationLimits, GenerationOptions, GenerationSampling},
        strict_int8::Int8MemoryRequirement},
    tasks::chat::{ChatMessage, ChatRole}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;
mod source;
mod batch;
mod scored;
mod scored_batch;
mod extract;

const MIB: u64 = 1024 * 1024;
const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_CONTENT_BYTES: usize = 1024 * 1024;
const SAMPLER_BYTES: u64 = 32 * MIB;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Task { Generate, Chat }

#[derive(Args)]
pub(crate) struct CandidateArgs {
    /// UTF-8 prompt (generate) or JSON message array (chat); '-' reads stdin.
    #[arg(default_value = "-")]
    input: PathBuf,
    /// Explicit local current-candidate INT8 .fnlpq; never an active catalog.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,
    /// Process memory-ledger ceiling, in MiB (not an operating-system RSS cap).
    #[arg(long, value_name = "MIB")]
    memory_mib: u64,
    /// Admitted native context, including the complete prompt and generation.
    #[arg(long, default_value_t = 2048)]
    context_tokens: usize,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    /// Generated content-byte ceiling; serialized result staging is additional.
    #[arg(long, default_value_t = 65_536)]
    max_output_bytes: usize,
    /// Input-byte ceiling, checked while reading, including transcript JSON.
    #[arg(long, default_value_t = 65_536)]
    max_input_bytes: usize,
    /// Maximum resident native weight payload admitted during streaming load.
    #[arg(long, default_value_t = 6144)]
    max_weight_mib: u64,
    /// Modeled input/tokenizer/planner/serialization reserve; no RSS guarantee.
    #[arg(long, default_value_t = 256)]
    preparation_mib: u64,
    /// Cooperative wall-time budget including input, preparation and loading.
    /// Blocking reads are not preempted; expiration is checked between stages.
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
    /// Maximum native-execution checkpoints, separate from loader checkpoints.
    #[arg(long, default_value_t = 1_000_000_000)]
    max_checkpoints: u64,
    /// Exactly 64 lowercase hexadecimal digits; absent means greedy decoding.
    #[arg(long, value_name = "HEX")]
    seed: Option<String>,
    #[arg(long, requires = "seed")]
    temperature_milli: Option<u32>,
    #[arg(long, requires = "seed")]
    top_k: Option<usize>,
    #[arg(long, requires = "seed")]
    top_p_ppm: Option<u32>,
    /// Include raw full-vocabulary log-probabilities, not confidence scores.
    #[arg(long)]
    logprobs: bool,
}

pub(crate) enum CandidateCommand {
    Text { task: Task, args: CandidateArgs },
    Source(source::SourceCommand),
    Scored(scored::ScoredCommand),
    Extract(extract::ExtractCommand),
    Batch(batch::BatchCommand),
    ScoreBatch(scored_batch::ScoreBatchCommand),
}

pub(crate) fn definition() -> clap::Command {
    clap::Command::new("candidate")
        .about("Explicit non-certified local INT8 inference (requires asupersync-runtime)")
        .long_about("Execute an explicitly selected local current-candidate INT8 artifact. This is not release activation, publisher authentication, numerical qualification or a production certification. No network, automatic download, thinking mode or tool execution is available. Single requests emit a completed JSON object; batch emits ordered candidate-framed NDJSON, not token events.")
        .subcommand_required(true)
        .subcommands(source::definitions())
        .subcommands(scored::definitions())
        .subcommand(batch::definition())
        .subcommand(scored_batch::definition())
        .subcommand(extract::definition())
        .subcommand(CandidateArgs::augment_args(clap::Command::new("generate")
            .about("Generate from a bounded UTF-8 prompt using the pinned chat template")))
        .subcommand(CandidateArgs::augment_args(clap::Command::new("chat")
            .about("Complete a bounded JSON array of system/user/assistant messages")))
}

impl CandidateCommand {
    pub(crate) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "candidate task required"))?;
        if name == "score-batch" {
            return scored_batch::ScoreBatchCommand::from_arg_matches(matches).map(Self::ScoreBatch);
        }
        if name == "extract" {
            return extract::ExtractCommand::from_arg_matches(matches).map(Self::Extract);
        }
        if name == "batch" {
            return batch::BatchCommand::from_arg_matches(matches).map(Self::Batch);
        }
        if let Some(kind) = source::Kind::named(name) {
            return source::SourceCommand::from_matches(kind, matches).map(Self::Source);
        }
        if let Some(kind) = scored::Kind::named(name) {
            return scored::ScoredCommand::from_matches(kind, matches).map(Self::Scored);
        }
        let task = match name {
            "generate" => Task::Generate,
            "chat" => Task::Chat,
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "candidate task refused")),
        };
        Ok(Self::Text { task, args: CandidateArgs::from_arg_matches(matches)? })
    }

    /// Called BEFORE the root dispatcher acquires any stdio locks. Batch
    /// transfers owned handles into the one hosted blocking invocation.
    pub(crate) fn run_stdio(self) -> ExitCode {
        match self {
            Self::Batch(command) => command.run_owned(io::stdin(), io::stdout(), &mut io::stderr()),
            Self::ScoreBatch(command) => command.run_owned(io::stdin(), io::stdout(), &mut io::stderr()),
            other => other.run(&mut io::stdin(), &mut io::stdout(), &mut io::stderr()),
        }
    }

    pub(crate) fn run(self, input: &mut impl Read, output: &mut impl Write,
        diagnostics: &mut impl Write) -> ExitCode {
        let result = self.execute(input, output);
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                // No nested parser, IO, artifact, runtime or model diagnostic
                // may echo paths, argv values, prompts or generated content.
                let _ = writeln!(diagnostics, "fnlp candidate: {}", error.message());
                error.exit_code()
            }
        }
    }

    fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (task, args) = match self {
            Self::Source(command) => return command.execute(input, output),
            Self::Scored(command) => return command.execute(input, output),
            Self::Extract(command) => return command.execute(input, output),
            // A borrowed stream cannot outlive the hosted blocking closure.
            // The executable dispatcher always takes run_stdio for batch.
            Self::Batch(_) | Self::ScoreBatch(_) => return Err(CandidateError::Arguments),
            Self::Text { task, args } => (task, args),
        };
        let limits = args.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(task, args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        {
            let _ = (task, limits, input, output);
            // Feature refusal is BEFORE opening even the input or model path.
            Err(CandidateError::Unavailable)
        }
    }
}

#[derive(Clone, Copy)]
struct Limits {
    memory_bytes: u64,
    weight_bytes: u64,
    preparation_bytes: u64,
    result_bytes: usize,
    max_prompt_tokens: usize,
    kv_bytes: u64,
}

impl CandidateArgs {
    fn validate(&self) -> Result<Limits, CandidateError> {
        if self.max_input_bytes == 0 || self.max_input_bytes > MAX_INPUT_BYTES
            || self.max_output_bytes == 0 || self.max_output_bytes > MAX_CONTENT_BYTES
            || self.max_new_tokens == 0 || self.max_new_tokens > 1024
            || self.max_new_tokens > self.context_tokens
            || self.timeout_seconds == 0 || self.timeout_seconds > 86_400
            || self.max_checkpoints < 2 || self.max_checkpoints > 1_000_000_000_000
            || self.model.as_os_str().is_empty() || self.input.as_os_str().is_empty()
        { return Err(CandidateError::Arguments); }
        let memory = Int8MemoryRequirement::for_context(self.context_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        let mib = |value: u64| value.checked_mul(MIB).filter(|&bytes| bytes != 0)
            .ok_or(CandidateError::Arguments);
        let memory_bytes = mib(self.memory_mib)?;
        let weight_bytes = mib(self.max_weight_mib)?;
        let preparation_bytes = mib(self.preparation_mib)?;
        // Covers worst-case JSON escaping, token/score vectors and fixed task
        // metadata. The bounded writer also enforces this cap during delivery.
        let result_bytes = self.max_output_bytes.checked_mul(8)
            .and_then(|bytes| bytes.checked_add(self.max_new_tokens.checked_mul(64)?))
            .and_then(|bytes| bytes.checked_add(65_536)).ok_or(CandidateError::Arguments)?;
        let preparation_floor = 128 * MIB + self.max_input_bytes as u64 * 32
            + (result_bytes as u64 + 4096) * 2;
        if preparation_bytes < preparation_floor || preparation_bytes > memory_bytes
            || weight_bytes > memory_bytes { return Err(CandidateError::Arguments); }
        let limits = Limits { memory_bytes, weight_bytes, preparation_bytes, result_bytes,
            max_prompt_tokens: self.context_tokens - self.max_new_tokens + 1, kv_bytes: memory.kv_bytes };
        self.options(166_101)?.validate(self.generation_limits(limits))
            .map_err(|_| CandidateError::Arguments)?;
        Ok(limits)
    }

    fn generation_limits(&self, limits: Limits) -> GenerationLimits {
        GenerationLimits { max_prompt_tokens: limits.max_prompt_tokens,
            max_new_tokens: self.max_new_tokens, max_output_bytes: self.max_output_bytes,
            max_sampler_bytes: SAMPLER_BYTES }
    }

    fn options(&self, eos: u32) -> Result<GenerationOptions, CandidateError> {
        let mut options = GenerationOptions::greedy(self.max_new_tokens, self.max_output_bytes, eos);
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

fn parse_seed(value: &str) -> Result<[u8; 32], CandidateError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(CandidateError::Arguments);
    }
    let mut seed = [0_u8; 32];
    for (index, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| CandidateError::Arguments)?;
    }
    Ok(seed)
}

/// Read at most cap+1 bytes, including on a pipe with no metadata. This never
/// reads the rest of an oversized stream just to discover its final length.
fn read_input(input: &mut impl Read, cap: usize) -> Result<String, CandidateError> {
    let size = cap.checked_add(1).ok_or(CandidateError::Input)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|_| CandidateError::Memory)?;
    input.take(size as u64).read_to_end(&mut bytes).map_err(|_| CandidateError::Input)?;
    if bytes.len() > cap { return Err(CandidateError::Input); }
    String::from_utf8(bytes).map_err(|_| CandidateError::Input)
}

fn parse_messages(input: &str, cap: usize) -> Result<Vec<ChatMessage>, CandidateError> {
    let value = canonjson::parse_str_with_limits(input, canonjson::ParseLimits {
        max_depth: 4, max_string_bytes: cap,
    }).map_err(|_| CandidateError::Input)?;
    let values = value.as_array().ok_or(CandidateError::Input)?;
    if values.is_empty() || values.len() > 128 { return Err(CandidateError::Input); }
    let messages: Vec<ChatMessage> = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    if messages.last().is_none_or(|message| message.role != ChatRole::User)
        || messages.iter().enumerate().any(|(index, message)| index != 0 && message.role == ChatRole::System)
    { return Err(CandidateError::Input); }
    Ok(messages)
}

/// A complete response is staged before its first stdout write. The caller
/// retains the hosted output guard through write AND flush. A broken pipe can
/// still truncate transport bytes; it is always an error, never task success.
fn publish<T: Serialize>(value: &T, cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(cap).map_err(|_| CandidateError::Memory)?;
    let mut stage = BoundedOutput { bytes, cap };
    serde_json::to_writer(&mut stage, value).map_err(|_| CandidateError::Output)?;
    stage.write_all(b"\n").map_err(|_| CandidateError::Output)?;
    output.write_all(&stage.bytes).and_then(|()| output.flush()).map_err(|_| CandidateError::Output)
}
struct BoundedOutput { bytes: Vec<u8>, cap: usize }
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self.bytes.len().checked_add(bytes.len())
            .filter(|&next| next <= self.cap)
            .ok_or_else(|| io::Error::other("candidate output limit"))?;
        self.bytes.try_reserve_exact(next - self.bytes.len())
            .map_err(|_| io::Error::other("candidate output allocation"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateError {
    Arguments,
    #[cfg(not(feature = "asupersync-runtime"))]
    Unavailable,
    Input, Memory, Runtime, Model, Identity, Planning, Timeout, Execution, Output, Batch,
}
impl CandidateError {
    fn message(self) -> &'static str {
        match self {
            Self::Arguments => "invalid or inconsistent finite limits/options",
            #[cfg(not(feature = "asupersync-runtime"))]
            Self::Unavailable => "this build requires the asupersync-runtime feature for candidate inference",
            Self::Input => "input refused: check byte limits, UTF-8 and the selected task input/options contract",
            Self::Memory => "process preparation memory admission failed",
            Self::Runtime => "process runtime initialization refused",
            Self::Model => "explicit local current-candidate model loading refused",
            Self::Identity => "candidate model identity changed or is incompatible",
            Self::Planning => "prompt, schema or task contract refused before weight loading",
            Self::Timeout => "cooperative request deadline expired; no result published",
            Self::Execution => "native execution failed or was cancelled; no result published",
            Self::Output => "completed result could not be delivered",
            Self::Batch => "corpus stopped or contains failed records; earlier completed frames may exist",
        }
    }
    fn exit_code(self) -> ExitCode {
        match self {
            Self::Arguments | Self::Input | Self::Planning => ErrorCode::Usage.as_process_exit(),
            _ => ErrorCode::Generic.as_process_exit(),
        }
    }
}
