//! Explicit live token NDJSON; separate from completed-result and batch modes.
use super::*;
use clap::Subcommand;
use crate::native_engine::decode::DecodeCancellationKind;
pub(super) mod events;
#[cfg(test)] pub(super) mod tests;

pub(super) const PROTOCOL: &str = "fnlp-candidate-token-stream-v1";
pub(super) const FRAME_BYTES: usize = 4096;

#[derive(Args)]
pub(crate) struct StreamCommand {
    #[command(subcommand)] operation: Operation,
}
#[derive(Subcommand)]
enum Operation {
    /// Generate from one UTF-8 prompt, emitting provisional token events.
    Generate(StreamArgs),
    /// Complete one JSON message array, emitting provisional token events.
    Chat(StreamArgs),
}
#[derive(Args)]
pub(super) struct StreamArgs {
    #[command(flatten)] pub common: CandidateArgs,
    /// Entire serialized NDJSON transport, including the complete final result.
    /// Independent of --max-output-bytes, which bounds generated content only.
    #[arg(long, default_value_t = 16_777_216)] pub max_stream_bytes: u64,
}
#[derive(Clone, Copy)]
pub(super) struct StreamBounds {
    pub tokens: usize, pub content_bytes: usize, pub capture_logprobs: bool,
    pub event_bytes: usize, pub terminal_bytes: usize,
    pub stream_bytes: u64, pub sink_memory_bytes: u64,
}
impl StreamArgs {
    pub(super) fn validate(&self) -> Result<(Limits, StreamBounds), Failure> {
        let limits = self.common.validate()?;
        if self.max_stream_bytes == 0 || self.max_stream_bytes > 1024 * MIB {
            return Err(Failure::usage("stream_transport_options"));
        }
        let a = &self.common;
        let event_bytes = a.max_output_bytes.checked_mul(4).and_then(|n| n.checked_add(FRAME_BYTES))
            .ok_or_else(|| Failure::usage("stream_size_arithmetic"))?;
        let terminal_bytes = limits.result_bytes.checked_add(FRAME_BYTES)
            .ok_or_else(|| Failure::usage("stream_size_arithmetic"))?;
        // u8 JSON arrays cost at most four bytes per source byte. Fixed framing
        // is bounded independently for EVERY token, including an empty EOS.
        let wire_floor = (a.max_new_tokens as u64).checked_mul(FRAME_BYTES as u64)
            .and_then(|n| n.checked_add(a.max_output_bytes as u64 * 4))
            .and_then(|n| n.checked_add(FRAME_BYTES as u64 + terminal_bytes as u64))
            .ok_or_else(|| Failure::usage("stream_size_arithmetic"))?;
        if self.max_stream_bytes < wire_floor { return Err(Failure::usage("stream_transport_too_small")); }
        // Retained byte/token/score reconciliation plus one staged permit and
        // canonical serializer trees. This is a model, not an allocator cap.
        let sink_memory_bytes = (event_bytes as u64).checked_add(terminal_bytes as u64)
            .and_then(|n| n.checked_mul(8)).and_then(|n| n.checked_add(16 * MIB))
            .and_then(|n| n.checked_add(a.max_output_bytes as u64 * 2))
            .and_then(|n| n.checked_add(a.max_new_tokens as u64 * 64))
            .ok_or_else(|| Failure::usage("stream_memory_arithmetic"))?;
        let minimum = limits.preparation_bytes.checked_mul(2)
            .and_then(|n| n.checked_add(sink_memory_bytes)).and_then(|n| n.checked_add(limits.weight_bytes))
            .ok_or_else(|| Failure::usage("stream_memory_arithmetic"))?;
        if minimum > limits.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "stream_memory_floor", cancellation: None });
        }
        Ok((limits, StreamBounds { tokens: a.max_new_tokens, content_bytes: a.max_output_bytes,
            capture_logprobs: a.logprobs, event_bytes, terminal_bytes,
            stream_bytes: self.max_stream_bytes, sink_memory_bytes }))
    }
}
pub(super) fn definition() -> clap::Command {
    StreamCommand::augment_args(clap::Command::new("stream")
        .about("Stream native generation/chat as bounded provisional token NDJSON")
        .long_about("Explicit non-certified local INT8 model only. Emits run_start, token and, only after native drain and independent final validation, run_complete. Token data contains exact u8 byte arrays, which may split UTF-8 characters. EOF, an EOS token or a partial line is NOT successful completion. Errors emit a content-free run_error on stderr and exit nonzero; stdout is never retried after a failed write. The final frame contains the same complete task result as buffered generation. No network, thinking or tool execution. Blocking writes apply backpressure and are not safely preemptible."))
}
impl StreamCommand {
    fn into_parts(self) -> (Task, StreamArgs) {
        match self.operation { Operation::Generate(args) => (Task::Generate, args),
            Operation::Chat(args) => (Task::Chat, args) }
    }
    pub(super) fn run_owned<R: Read, W: Write + Send + 'static>(self, mut input: R,
        output: W, diagnostics: &mut impl Write) -> ExitCode {
        let (task, args) = self.into_parts();
        let result = execute(task, args, &mut input, output);
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: PROTOCOL, schema_version: 1, event: "run_error",
                    scope: "real-artifact-current-candidate", evidence: "non_authoritative", request_seq: 1,
                    status: "incomplete", code: failure.code, exit_code: failure.exit,
                    cancellation: failure.cancellation, provisional_tokens_may_exist: true };
                if publish(&report, FRAME_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
}
fn execute<W: Write + Send + 'static>(task: Task, args: StreamArgs, input: &mut impl Read, output: W)
    -> Result<(), Failure> {
    let (limits, bounds) = args.validate()?;
    #[cfg(feature = "asupersync-runtime")]
    { runtime::streaming::execute(task, args.common, limits, bounds, input, output) }
    #[cfg(not(feature = "asupersync-runtime"))]
    {
        let _ = (task, args, limits, bounds, input, output);
        Err(Failure::usage("stream_profile_unavailable"))
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Failure { pub exit: ErrorCode, pub code: &'static str, pub cancellation: Option<DecodeCancellationKind> }
impl Failure {
    pub(super) fn usage(code: &'static str) -> Self { Self { exit: ErrorCode::Usage, code, cancellation: None } }
    pub(super) fn output(code: &'static str) -> Self { Self { exit: ErrorCode::Generic, code, cancellation: None } }
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.code) }
}
impl From<CandidateError> for Failure {
    fn from(error: CandidateError) -> Self {
        let (exit, code) = match error {
            CandidateError::Arguments => (ErrorCode::Usage, "stream_options"),
            CandidateError::Input => (ErrorCode::InputDecodeOrParse, "stream_input"),
            CandidateError::Planning => (ErrorCode::Usage, "stream_plan_refused"),
            CandidateError::Memory | CandidateError::Runtime => (ErrorCode::AdmissionOrResourceLimit, "stream_admission"),
            CandidateError::Identity => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "stream_model_identity"),
            CandidateError::Timeout => (ErrorCode::BudgetOrTimeout, "stream_invocation_deadline"),
            _ => (ErrorCode::Generic, "stream_setup_or_delivery"),
        };
        Self { exit, code, cancellation: None }
    }
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, event: &'static str, scope: &'static str,
    evidence: &'static str, request_seq: u64, status: &'static str, code: &'static str,
    exit_code: ErrorCode, cancellation: Option<DecodeCancellationKind>, provisional_tokens_may_exist: bool,
}
