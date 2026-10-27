//! Live INT4 token delivery using the same explicit model and text policies.
//! A bounded terminal frame is mandatory; provisional tokens are not a result.
use super::*;
use crate::candidate_cli::stream::{Failure, StreamBounds, FRAME_BYTES, PROTOCOL};
use crate::native_engine::decode::DecodeCancellationKind;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StreamArgs {
    /// Entire serialized NDJSON transport, including the completed task result.
    /// --max-output-bytes separately bounds generated content bytes.
    #[arg(long, default_value_t = 16_777_216)]
    max_stream_bytes: u64,
}

pub(super) fn definition() -> clap::Command {
    let task = |name: &'static str, about: &'static str| StreamArgs::augment_args(Int4Args::augment_args(
        clap::Command::new(name).about(about)))
        .mut_arg("prefill_rows", |arg| arg.help("Explicit grouped-INT4 layer-major prompt execution, 1..=64 rows; extra workspace is process-admitted"));
    clap::Command::new("stream")
        .about("Stream candidate INT4 generation/chat as bounded provisional token NDJSON")
        .long_about("Use the same explicit pinned source or digest-pinned snapshot as buffered INT4 generation. Emits run_start, token and, only after physical native drain and independent final validation, run_complete. Token byte arrays may split UTF-8 characters. A token, EOS, partial line or EOF never certifies completion. Errors report a content-free run_error on stderr and exit nonzero; stdout is never retried. The final frame retains the complete unchanged native task result. --max-stream-bytes prices all frames and reserves terminal space. Seed, stop suffixes, penalties, minimum tokens, log probabilities and explicit prefill/kernel selection retain buffered semantics. Blocking writers backpressure inference and cannot be safely preempted mid-write. No download, automatic model selection, thinking, tool execution or release qualification.")
        .subcommand_required(true)
        .subcommand(task("generate", "Generate from one UTF-8 prompt with provisional token events"))
        .subcommand(task("chat", "Complete one JSON message transcript with provisional token events"))
}

pub(super) fn from_matches(matches: &clap::ArgMatches)
    -> Result<(Task, StreamArgs, Int4Args), clap::Error> {
    let (name, matches) = matches.subcommand().ok_or_else(|| clap::Error::raw(
        clap::error::ErrorKind::MissingSubcommand, "INT4 stream text operation required"))?;
    let task = match name {
        "generate" => Task::Generate,
        "chat" => Task::Chat,
        _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
            "INT4 stream text operation refused")),
    };
    Ok((task, StreamArgs::from_arg_matches(matches)?, Int4Args::from_arg_matches(matches)?))
}

impl StreamArgs {
    pub(super) fn validate(self, args: &Int4Args) -> Result<(Int4Limits, StreamBounds), Failure> {
        let mut limits = args.validate()?;
        if self.max_stream_bytes == 0 || self.max_stream_bytes > 1024 * MIB {
            return Err(Failure::usage("stream_transport_options"));
        }
        let event_bytes = args.max_output_bytes.checked_mul(4).and_then(|n| n.checked_add(FRAME_BYTES))
            .ok_or_else(|| Failure::usage("stream_size_arithmetic"))?;
        let terminal_bytes = limits.result_bytes.checked_add(FRAME_BYTES)
            .ok_or_else(|| Failure::usage("stream_size_arithmetic"))?;
        let wire_floor = (args.max_new_tokens as u64).checked_mul(FRAME_BYTES as u64)
            .and_then(|n| n.checked_add(args.max_output_bytes as u64 * 4))
            .and_then(|n| n.checked_add(FRAME_BYTES as u64 + terminal_bytes as u64))
            .ok_or_else(|| Failure::usage("stream_size_arithmetic"))?;
        if self.max_stream_bytes < wire_floor { return Err(Failure::usage("stream_transport_too_small")); }
        // Reconciliation vectors, one reserved event and bounded canonical
        // staging remain live until final publication, including failed writes.
        let sink_memory_bytes = (event_bytes as u64).checked_add(terminal_bytes as u64)
            .and_then(|n| n.checked_mul(8)).and_then(|n| n.checked_add(16 * MIB))
            .and_then(|n| n.checked_add(args.max_output_bytes as u64 * 2))
            .and_then(|n| n.checked_add(args.max_new_tokens as u64 * 64))
            .ok_or_else(|| Failure::usage("stream_memory_arithmetic"))?;
        // The existing minimum includes model weights, full native KV/scratch,
        // extra packed prefill, sampler, output, CLI preparation and cleanup.
        // The streaming host additionally owns the moved prompt and sink.
        limits.minimum_memory_bytes = limits.minimum_memory_bytes
            .checked_add(limits.preparation_bytes).and_then(|n| n.checked_add(sink_memory_bytes))
            .ok_or_else(|| Failure::usage("stream_memory_arithmetic"))?;
        if limits.minimum_memory_bytes > limits.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
                code: "stream_memory_floor", cancellation: None });
        }
        Ok((limits, StreamBounds { tokens: args.max_new_tokens, content_bytes: args.max_output_bytes,
            capture_logprobs: args.logprobs, event_bytes, terminal_bytes,
            stream_bytes: self.max_stream_bytes, sink_memory_bytes }))
    }

    pub(super) fn run_owned<R: Read, W: Write + Send + 'static>(self, task: Task, args: Int4Args,
        mut input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute(task, args, &mut input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: PROTOCOL, schema_version: 1, event: "run_error",
                    scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
                    request_seq: 1, status: "incomplete", code: failure.code, exit_code: failure.exit,
                    cancellation: failure.cancellation, provisional_tokens_may_exist: true };
                if publish(&report, FRAME_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }

    fn execute<W: Write + Send + 'static>(self, task: Task, args: Int4Args,
        input: &mut impl Read, output: W) -> Result<(), Failure> {
        let (limits, bounds) = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::streaming::execute(task, args, limits, bounds, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        {
            let _ = (task, args, limits, bounds, input, output);
            Err(Failure::usage("stream_profile_unavailable"))
        }
    }
}

#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, event: &'static str, scope: &'static str,
    evidence: &'static str, request_seq: u64, status: &'static str, code: &'static str,
    exit_code: ErrorCode, cancellation: Option<DecodeCancellationKind>, provisional_tokens_may_exist: bool,
}

#[cfg(test)] pub(super) mod tests;
