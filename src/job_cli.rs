//! Explicit, model-free CLI management of retained owned-job results.
//! No start/resume inference route, model discovery, recovery deletion, or
//! secret argv option is registered. The command uses the real process host.
use std::{io::{Read, Write}, path::{Path, PathBuf}, process::ExitCode, time::Duration};
use clap::{Args, Subcommand};
use serde::Serialize;
use crate::{canonjson, error::ErrorCode, local_io,
    jobs::{JobError, JobId, JobLimits, JobSecret, runner::JobRunError},
    hosted::{CancellationToken, HostedError, RunLimits,
        corpus::jobs::{HostedJobError, JobManagementLimits, StoredJobOperation, StoredJobRequest}},
    native_engine::decode::DecodeCancellationKind,
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy};

const LIMIT_FILE_BYTES: usize = 16 * 1024;
const RESPONSE_BYTES: usize = 16 * 1024;

#[derive(Args)]
#[command(about = "Authenticate and manage existing retained job results; no model execution", arg_required_else_help = true)]
pub(crate) struct JobCommand {
    #[command(subcommand)] operation: Operation,
}
#[derive(Subcommand)]
enum Operation {
    /// Authenticate committed state and report unresolved tails/stages without repair.
    Status(Common),
    /// Verify the full stored state; uncommitted tails/stages fail without repair.
    Verify(Common),
    /// Publish all committed results to the protected job's materialized.ndjson.
    Materialize(Publish),
    /// Describe this management-only JSON interface without opening any files.
    Schema,
}
#[derive(Args)]
struct Publish {
    #[command(flatten)] common: Common,
    /// Explicit spelling of the only supported publication ordering.
    #[arg(long)] ordered: bool,
}
#[derive(Args)]
struct Common {
    /// Existing owner-only job directory; never created or implicitly adopted.
    #[arg(long)] job_dir: PathBuf,
    /// Expected random job ID: exactly 32 lowercase hexadecimal characters.
    #[arg(long, value_parser = parse_job_id)] job_id: JobId,
    /// Private regular file containing exactly 32 key bytes, not hex or argv data.
    #[arg(long)] key_file: PathBuf,
    /// JSON encoding of the exact original JobLimits, authenticated against storage.
    #[arg(long)] limits: PathBuf,
    /// Explicit whole-process memory ceiling in bytes; no inferred machine RAM.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))] memory_bytes: u64,
    /// Modeled fsqlite RAM reservation, not the journal file's byte limit.
    #[arg(long, default_value_t = 67_108_864, value_parser = clap::value_parser!(u64).range(1..))]
    journal_memory_bytes: u64,
    /// Modeled serialization/allocator overhead beyond the bounded frame buffers.
    #[arg(long, default_value_t = 16_777_216, value_parser = clap::value_parser!(u64).range(1..))]
    serialization_memory_bytes: u64,
    /// Finite deadline; cooperative cancellation does not preempt blocking OS IO.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..=86_400))]
    max_seconds: u64,
    #[arg(long, default_value_t = 1_000_000, value_parser = clap::value_parser!(u64).range(2..))]
    max_checkpoints: u64,
    /// Canonical JSON is the default and only management output format.
    #[arg(long)] json: bool,
}
pub(crate) fn definition() -> clap::Command { JobCommand::augment_args(clap::Command::new("job")) }

#[derive(Serialize)]
struct Success<'a, T: Serialize> {
    schema_version: u32,
    operation: StoredJobOperation,
    status: &'static str,
    report: &'a T,
}
#[derive(Clone, Copy, Debug)]
struct Failure { exit: ErrorCode, code: &'static str }
#[derive(Serialize)]
struct ErrorResponse {
    schema_version: u32, operation: &'static str, status: &'static str,
    exit_code: ErrorCode, code: &'static str,
}
impl JobCommand {
    pub(crate) fn run(self, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let (operation, common) = match self.operation {
            Operation::Schema => return match emit(output, &schema()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(failure) => report_error("schema", failure, diagnostics),
            },
            Operation::Status(common) => (StoredJobOperation::Status, common),
            Operation::Verify(common) => (StoredJobOperation::Verify, common),
            Operation::Materialize(publish) => {
                let _ = publish.ordered; // Both spellings select ordered publication.
                (StoredJobOperation::MaterializeOrdered, publish.common)
            }
        };
        let name = operation_name(operation);
        let result = (|| {
            // Strict bounded configuration parsing occurs before any job open.
            // Key-file reads use the existing owner-only/no-follow primitive.
            let limits = read_limits(&common.limits)?;
            let key = JobSecret::read(&common.key_file).map_err(job_failure)?;
            let host = ResourceHostConfig { runtime_preset: RuntimePreset::CurrentThread,
                runtime_workers: 1, max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
                helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
                leak_response_policy: LeakResponsePolicy::RecordAndEscalate };
            let engine = NlpEngine::builder().resource_config(host).build()
                .map_err(|_| Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "runtime_admission" })?;
            let host_limits = JobManagementLimits {
                run: RunLimits { max_elapsed: Duration::from_secs(common.max_seconds),
                    max_checkpoints: common.max_checkpoints, cleanup_reserve_bytes: 65536 },
                journal_reserve_bytes: common.journal_memory_bytes,
                serialization_reserve_bytes: common.serialization_memory_bytes,
                io_reserve_bytes: 65536,
            };
            let report = engine.manage_owned_job(StoredJobRequest {
                root: common.job_dir, key, job_id: common.job_id, limits, operation,
            }, host_limits, CancellationToken::default()).map_err(host_failure)?;
            let _ = common.json;
            // The report's memory guard remains live through complete write AND
            // flush. No retained private result bytes are emitted to stdout.
            emit(output, &Success { schema_version: 1, operation, status: "ok", report: &report })
        })();
        match result { Ok(()) => ExitCode::SUCCESS, Err(failure) => report_error(name, failure, diagnostics) }
    }
}
fn operation_name(operation: StoredJobOperation) -> &'static str {
    match operation { StoredJobOperation::Status => "status", StoredJobOperation::Verify => "verify",
        StoredJobOperation::MaterializeOrdered => "materialize_ordered" }
}
fn parse_job_id(text: &str) -> Result<JobId, String> {
    if text.len() != 32 || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("expected 32 lowercase hexadecimal characters".to_owned());
    }
    let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
    let mut bytes = [0_u8; 16];
    for (out, pair) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        *out = (digit(pair[0]) << 4) | digit(pair[1]);
    }
    Ok(JobId(bytes))
}
fn read_limits(path: &Path) -> Result<JobLimits, Failure> {
    let file = local_io::open_document(path).map_err(|_| Failure { exit: ErrorCode::InputDecodeOrParse, code: "limits_input" })?;
    let mut bytes = Vec::new();
    file.take((LIMIT_FILE_BYTES + 1) as u64).read_to_end(&mut bytes)
        .map_err(|_| Failure { exit: ErrorCode::InputDecodeOrParse, code: "limits_input" })?;
    parse_limits(&bytes)
}
fn parse_limits(bytes: &[u8]) -> Result<JobLimits, Failure> {
    let failure = Failure { exit: ErrorCode::InputDecodeOrParse, code: "limits_input" };
    if bytes.len() > LIMIT_FILE_BYTES { return Err(failure); }
    let text = std::str::from_utf8(bytes).map_err(|_| failure)?;
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits { max_depth: 8, max_string_bytes: 4096 })
        .map_err(|_| failure)?;
    let limits: JobLimits = serde_json::from_value(value).map_err(|_| failure)?;
    limits.validate().map_err(job_failure)?;
    Ok(limits)
}
fn emit<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<(), Failure> {
    let failure = Failure { exit: ErrorCode::Generic, code: "output_io" };
    let bytes = canonjson::canonical_bytes(value).map_err(|_| Failure { exit: ErrorCode::Generic, code: "response_encoding" })?;
    if bytes.len() >= RESPONSE_BYTES { return Err(Failure { exit: ErrorCode::Generic, code: "response_bound" }); }
    writer.write_all(&bytes).and_then(|()| writer.write_all(b"\n")).and_then(|()| writer.flush()).map_err(|_| failure)
}
fn report_error(operation: &'static str, failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    let response = ErrorResponse { schema_version: 1, operation, status: "error", exit_code: failure.exit, code: failure.code };
    if emit(diagnostics, &response).is_err() { return ErrorCode::Generic.as_process_exit(); }
    failure.exit.as_process_exit()
}
fn cancelled(cause: DecodeCancellationKind) -> Failure {
    let budget = matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout
        | DecodeCancellationKind::PollQuota | DecodeCancellationKind::CostBudget);
    Failure { exit: if budget { ErrorCode::BudgetOrTimeout } else { ErrorCode::Cancelled },
        code: if budget { "execution_budget" } else { "cancelled" } }
}
fn job_failure(error: JobError) -> Failure {
    let (exit, code) = match error {
        JobError::Cancelled(cause) => return cancelled(cause),
        JobError::Corrupt | JobError::Authentication => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "stored_integrity"),
        JobError::Mismatch(_) => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "stored_contract_mismatch"),
        JobError::UnsafeStorage => (ErrorCode::AdmissionOrResourceLimit, "unsafe_storage"),
        JobError::Busy => (ErrorCode::AdmissionOrResourceLimit, "job_busy"),
        JobError::Limit | JobError::Allocation | JobError::Platform => (ErrorCode::AdmissionOrResourceLimit, "resource_limit"),
        JobError::WorkLimit => (ErrorCode::BudgetOrTimeout, "work_limit"),
        JobError::UncommittedTail => (ErrorCode::StructuredTaskNoResult, "uncommitted_tail_or_stage"),
        JobError::Incomplete => (ErrorCode::StructuredTaskNoResult, "job_incomplete"),
        JobError::PublicationUncertain => (ErrorCode::Generic, "publication_uncertain"),
        JobError::InvalidLimits => (ErrorCode::Usage, "invalid_limits"),
        JobError::InvalidInput | JobError::DuplicateId | JobError::InvalidIdentity => (ErrorCode::InputDecodeOrParse, "invalid_input"),
        _ => (ErrorCode::Generic, "stored_operation_failed"),
    };
    Failure { exit, code }
}
fn host_failure(error: HostedJobError) -> Failure {
    match error {
        HostedJobError::Job(JobRunError::Storage(error)) => job_failure(error),
        HostedJobError::Host(HostedError::Stopped { stop, .. }) => cancelled(stop.kind),
        HostedJobError::Host(HostedError::Reservation(_) | HostedError::Limits(_) | HostedError::Reentrant
            | HostedError::MissingRuntimeContext | HostedError::SingleCoordinatorRequired) =>
            Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "runtime_admission" },
        _ => Failure { exit: ErrorCode::Generic, code: "runtime_or_job_failed" },
    }
}
#[derive(Serialize)]
struct ManagementSchema {
    schema_version: u32, commands: [&'static str; 4], report_scope: &'static str,
    original_inputs_required: bool, model_required: bool, can_execute_or_resume: bool,
    stdout: &'static str, stderr: &'static str, publication: &'static str,
    tail_policy: &'static str, report_fields: [&'static str; 12],
}
fn schema() -> ManagementSchema {
    ManagementSchema { schema_version: 1, commands: ["status", "verify", "materialize", "schema"],
        report_scope: "authenticated-stored-state-not-input-replay-v1",
        original_inputs_required: false, model_required: false, can_execute_or_resume: false,
        stdout: "one canonical success JSON line containing operation and metadata report",
        stderr: "fixed-code JSON failure; no paths, input values, keys or parser excerpts",
        publication: "explicit ordered materialized.ndjson inside protected job; no replacement",
        tail_policy: "status reports orphans; verify/materialize refuse; no deletion or repair",
        report_fields: ["schema_version", "verification_scope", "job_id", "items", "committed", "attempts",
            "reserved_work", "committed_spool_bytes", "uncommitted_spool_bytes", "staged_output_present",
            "materialized", "unacknowledged_output_present"] }
}

#[cfg(test)] mod tests;
