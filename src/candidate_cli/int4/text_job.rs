//! Retained generation/chat with one fixed addressed grouped-INT4 recipe.
//! Whole completed items commit; resumes never renew sampling or native work.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use super::extract_job::StorageArgs;
use crate::{
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits, runner::generation::GenerationJobTask},
    native_engine::{generation::grouped_int4::Int4GenerationBudget, strict_int4::Int4RunBudget},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { task: GenerationJobTask, storage: StorageArgs }

pub(super) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let tasks = ["generate", "chat"].into_iter().map(|task| {
            let command = StorageArgs::augment_args(Int4Args::augment_args(clap::Command::new(task)))
                .about("Retain complete native results under one immutable generation recipe")
                .long_about("Run or resume retained generate/chat items on one resident grouped-INT4 model. Supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation, including resume. text is the prompt or final user turn; optional task_args contains only sample_index and, for chat, history. Shared seed, stops, penalties, logprobs, context, task limits and model/group identity are authenticated and cannot change on resume. No seed means greedy; no implicit random seed is generated. Requires --store-results, existing protected --job-dir and --key-file, exact --job-id, and immutable --limits. All five native lifetime work axes are charged before each attempt and never refunded for EOS, failure or cancellation; mask_node_visits must be zero. Results commit only after whole native completion. Stdout is metadata only; --materialize publishes verified private results after every item commits. The retained adapter is serial; --prefill-rows and cohort scheduling are unsupported. No partial-token checkpoints, input persistence, flush controls, per-record policy changes, automatic retry, thinking mode, tools, model download or release activation. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
                .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population, also on resume; '-' reads stdin"))
                .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for each complete NDJSON record; immutable JobLimits.max_input_bytes_per_item must fit"))
                .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable work and attempts never renew"))
                .mut_arg("prefill_rows", |arg| arg.help("Unsupported on retained text jobs; the native adapter executes whole items serially"))
                .mut_arg("store_results", |arg| arg.help("Required consent to retain private generated text, tokens and optional logprobs"))
                .mut_arg("limits_file", |arg| arg.help("Exact immutable JobLimits JSON; all five lifetime native counters and zero mask work"));
            if name == "resume" {
                command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                    .action(clap::ArgAction::SetTrue)
                    .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
            } else { command }
        });
        clap::Command::new(name).about(about).subcommand_required(true).subcommands(tasks)
    };
    clap::Command::new("text-job")
        .about("Start or resume retained grouped-INT4 generation and chat")
        .subcommand_required(true)
        .subcommand(operation("start", "Create an explicitly retained INT4 generation/chat job"))
        .subcommand(operation("resume", "Authenticate the original population and run pending text items"))
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    common: Int4Limits,
    transport: PopulationReadLimits,
    native: Int4GenerationBudget,
    buffer_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}

impl JobArgs {
    fn task_name(&self) -> &'static str {
        match self.task { GenerationJobTask::Generate => "generate", GenerationJobTask::Chat => "chat" }
    }
    /// Reject configuration mistakes before opening private storage or input.
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_text_job_storage"));
        }
        if args.prefill()?.is_some() { return Err(Failure::usage("text_job_serial_schedule")); }
        let common = args.validate()?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("text_job_memory"));
            }
        }
        Ok(common)
    }

    /// Immutable JobLimits are the lifetime authority, never multiplied by
    /// population size or allowed attempts. The core also binds the projected
    /// logits axis, which Int4RunBudget itself does not carry separately.
    fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let work = job.max_work.int4_model_work();
        if job.max_work.mask_node_visits != 0 || work.forward_positions == 0 || work.projected_logits == 0
            || work.attention_pairs == 0 || work.projections.dot_products == 0
            || work.projections.multiply_accumulates == 0 {
            return Err(Failure::usage("lifetime_text_work_limits"));
        }
        let native = Int4GenerationBudget { native: Int4RunBudget::exact(work),
            max_kv_bytes: common.native.kv_bytes, max_sampler_bytes: SAMPLER_BYTES };
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Match JobHostLimits::required_buffer_bytes for the whole original
        // population, manifest and index. CLI preparation remains charged in
        // parallel; common already prices native KV, sampler and result memory.
        let buffer_bytes = sum(&[
            product(job.max_snapshot_bytes, 2)?, product(job.max_items, 1024)?,
            product(job.max_input_bytes_per_item as u64, 8)?,
            product(job.max_result_bytes as u64, 4)?, 2 * MIB,
            preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes,
        ]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        common.minimum_memory_bytes = common.minimum_memory_bytes.checked_add(buffer_bytes)
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { common, transport, native, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}

impl JobCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<(Self, Int4Args), clap::Error> {
        let invalid = |message| clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, message);
        let (operation, matches) = matches.subcommand().ok_or_else(|| invalid("text-job operation required"))?;
        let (name, matches) = matches.subcommand().ok_or_else(|| invalid("text-job task required"))?;
        let task = match name {
            "generate" => GenerationJobTask::Generate, "chat" => GenerationJobTask::Chat,
            _ => return Err(invalid("text-job task refused")),
        };
        let mode = match operation {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(invalid("text-job operation refused")),
        };
        let args = JobArgs { task, storage: StorageArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name(); let task = self.args.task_name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, task, status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored status or explicitly resume with original inputs and options; do not assume rollback" };
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write) -> Result<(), Failure> {
        let common = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, common, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, common, input, output); Err(Failure::usage("owned_text_job_profile_unavailable")) }
    }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, task: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
