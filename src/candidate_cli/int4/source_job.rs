//! Explicit durable jobs for the four source tasks on grouped INT4.
//! The complete original population and lifetime budgets remain immutable.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::source::{SourceMaskBudget, grouped_int4::Int4SourceBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}, source::Kind},
    jobs::{JobLimits, population::PopulationReadLimits},
};
use super::extract_job::StorageArgs;

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand { mode: RunMode, args: JobArgs }
pub(super) struct JobArgs {
    kind: Kind,
    source: source::SourceArgs,
    defaults: DefaultsArgs,
    storage: StorageArgs,
}
#[derive(Args)]
struct DefaultsArgs {
    /// Complete SourceBatchArgs JSON, including task, options, budget and any QA passages.
    #[arg(long, value_name = "FILE")]
    defaults: Option<PathBuf>,
}

pub(super) fn definitions() -> Vec<clap::Command> {
    [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer].into_iter().map(definition).collect()
}
pub(super) fn definition(kind: Kind) -> clap::Command {
    let name = match kind { Kind::Ner => "ner-job", Kind::Keyphrases => "keyphrases-job",
        Kind::Summarize => "summarize-job", Kind::Answer => "answer-job" };
    let operation = |name: &'static str, about: &'static str| {
        DefaultsArgs::augment_args(StorageArgs::augment_args(source::definition(kind).name(name).about(about)
            .long_about("Start or resume one fixed source-task job on a resident source-converted or independently pinned snapshot INT4 model. Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, and immutable --limits. Supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation; inputs are admitted in memory and are not persisted. For answer, text is the question and typed task_args supplies passages, options and budget unless an explicit complete --defaults file does so. Other tasks use shipped options unless explicit bounded defaults or per-record arguments override them. A record cannot switch task families or change the complete resident KV allowance. Flush controls and duplicate IDs anywhere in the population reject. Resume authenticates exact population, model/group identity, options and recipe before replay or tail repair. Lifetime native work includes every group dot and does not renew on resume; failed attempts remain debited. Stdout contains only progress/provenance metadata. Results remain in the explicit owner-only spool; --materialize publishes verified materialized.ndjson only after all items commit. Structural citations do not certify semantic support, and failed inference is never successful QA abstention. Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64.")
            .mut_arg("options", |arg| arg.hide(true))
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population, also on resume; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer cap for each complete NDJSON record and decoded source input; immutable stored input ceiling must fit"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; persisted work and attempts never renew"))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private source-task results in the owner-only spool"))
    };
    let start = operation("start", "Create a new explicitly retained source-task INT4 job");
    let resume = operation("resume", "Authenticate the complete original population and execute pending INT4 source tasks")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new(name).about("Start or resume retained source tasks with immutable INT4 lifetime budgets")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct JobExecutionLimits {
    pub common: Int4Limits,
    pub transport: PopulationReadLimits,
    pub native: Int4SourceBatchLimits,
    pub buffer_bytes: u64,
    pub preparation_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
    pub journal_reserve_bytes: u64,
    pub serialization_reserve_bytes: u64,
}
impl JobArgs {
    /// Pure admission, before reading private files or constructing the host.
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_storage_or_job_options"));
        }
        if self.source.options_path().is_some()
            || self.defaults.defaults.as_deref().is_some_and(|path| !local(path)) {
            return Err(Failure::usage("source_job_defaults"));
        }
        let common = self.source.validate(args).map_err(Failure::from)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok(common)
    }
    /// JobLimits is the sole lifetime work authority. Transport limits and
    /// attempts are independent and must never multiply native allowances.
    pub(super) fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits,
        job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = Int4SourceBatchLimits {
            max_model_work: job.max_work.int4_model_work(),
            masks: SourceMaskBudget { per_mask: self.source.masks(),
                max_visits_per_item: self.source.mask_visits(), max_visits_per_run: job.max_work.mask_node_visits },
        };
        native.validate().map_err(|_| Failure::usage("lifetime_native_or_mask_limits"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Full population/manifest/index copies plus the separately owned
        // hosted preparation capture, IO, journal, and result serialization.
        let buffer_bytes = sum(&[
            product(job.max_snapshot_bytes, 2)?, product(job.max_items, 1024)?,
            product(job.max_input_bytes_per_item as u64, 8)?,
            product(job.max_result_bytes as u64, 4)?, 2 * MIB,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes,
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
    pub(super) fn from_matches(kind: Kind, matches: &clap::ArgMatches) -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "source job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "source job operation refused")),
        };
        if matches.value_source("options") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "source jobs use complete --defaults or per-record task_args, not --options"));
        }
        let args = JobArgs { kind, source: source::SourceArgs::from_matches(matches)?,
            defaults: DefaultsArgs::from_arg_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        let task = self.args.kind.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let error = ErrorReport { schema_version: 1, operation, task, status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&error, job_cli::REPORT_BYTES, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }
    pub(super) fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args,
        input: R, output: &mut impl Write) -> Result<(), Failure> {
        let common = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, common, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            let _ = (self, args, common, input, output);
            Err(Failure::usage("owned_job_profile_unavailable"))
        }
    }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32,
    operation: &'static str,
    task: &'static str,
    status: &'static str,
    code: &'static str,
    exit_code: ErrorCode,
    durable_progress_may_exist: bool,
    recovery: &'static str,
}
