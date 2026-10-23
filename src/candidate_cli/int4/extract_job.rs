//! Explicit durable schema extraction on one resident grouped-INT4 model.
//! Frozen lifetime work is copied exactly; invocation limits never renew it.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::extract::{ExtractionMaskBudget, grouped_int4::Int4ExtractionBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobId, JobLimits, population::PopulationReadLimits},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand {
    mode: RunMode,
    args: JobArgs,
}
pub(super) struct JobArgs {
    pub extraction: extract::ExtractionArgs,
    pub storage: StorageArgs,
}
#[derive(Args)]
pub(super) struct StorageArgs {
    /// Required consent to retain private extraction results in the owner-only spool.
    #[arg(long)]
    pub store_results: bool,
    /// Existing owner-only job directory; never created or silently adopted.
    #[arg(long, value_name = "DIR")]
    pub job_dir: PathBuf,
    /// Expected random 128-bit job ID: exactly 32 lowercase hexadecimal characters.
    #[arg(long, value_parser = job_cli::parse_job_id, value_name = "HEX")]
    pub job_id: JobId,
    /// Protected regular file containing exactly 32 raw secret bytes, not hexadecimal text.
    #[arg(long, value_name = "FILE")]
    pub key_file: PathBuf,
    /// Exact immutable JobLimits JSON with lifetime native/mask work, attempts and storage caps.
    #[arg(long = "limits", value_name = "FILE")]
    pub limits_file: PathBuf,
    /// Publish verified materialized.ndjson after every item commits; never replace it.
    #[arg(long)]
    pub materialize: bool,
    /// Per-invocation transport bound; immutable snapshot/item ceilings also apply.
    #[arg(long, default_value_t = 64)]
    pub max_stream_mib: u64,
    /// Includes blank lines. Durable populations do not accept flush controls.
    #[arg(long, default_value_t = 100_000)]
    pub max_input_lines: u64,
    /// Modeled database RAM, separately charged from the journal file-size ceiling.
    #[arg(long, default_value_t = 64)]
    pub journal_memory_mib: u64,
    /// Additional canonical JSON/serializer staging beyond stored-result byte buffers.
    #[arg(long, default_value_t = 16)]
    pub serialization_memory_mib: u64,
}

pub(super) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        StorageArgs::augment_args(extract::definition().name(name).about(about)
            .long_about("Retain and resume schema/source extraction with one resident source-converted or independently pinned snapshot INT4 model. Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, and one exact shared --schema. Supply the COMPLETE original NDJSON population on every invocation; inputs are admitted in memory and are not persisted for you. Non-null task_args and flush controls are refused. Resume authenticates exact input bytes, schema, model/group identity and recipe before replay or tail repair. Lifetime native work includes every INT4 group dot and does not renew on resume; failed attempts remain debited. Stdout is metadata only; retained private results are read through explicit job verification/materialization. --materialize publishes only after all items commit. Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. No automatic retry, key/directory creation, release activation or semantic-accuracy claim.")
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population, also on resume; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer cap on each complete NDJSON record; immutable JobLimits.max_input_bytes_per_item must fit"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this complete invocation; persisted work and attempt ceilings do not renew")))
    };
    let start = operation("start", "Create a new explicitly retained INT4 extraction job");
    let resume = operation("resume", "Authenticate the original population and run pending INT4 extraction items")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new("extract-job")
        .about("Start or resume retained grouped-INT4 extraction with exact lifetime budgets")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

/// Host reservations include the complete frozen population, independently of
/// streaming buffer size. The CLI keeps its own preparation charge live too.
#[derive(Clone, Copy, Debug)]
pub(super) struct JobExecutionLimits {
    pub common: Int4Limits,
    pub transport: PopulationReadLimits,
    pub native: Int4ExtractionBatchLimits,
    pub buffer_bytes: u64,
    pub preparation_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
    pub journal_reserve_bytes: u64,
    pub serialization_reserve_bytes: u64,
}

impl JobArgs {
    /// Pure option validation, before reading any key, limits, schema or input.
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_storage_or_job_options"));
        }
        let common = self.extraction.validate(args).map_err(Failure::from)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok(common)
    }

    /// The limits file is the sole lifetime work authority. In particular,
    /// max_items/max_attempts must never multiply or replenish native counters.
    pub(super) fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits,
        job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = Int4ExtractionBatchLimits {
            max_model_work: job.max_work.int4_model_work(),
            masks: ExtractionMaskBudget { per_mask: self.extraction.masks(),
                max_visits_per_item: self.extraction.mask_visits(),
                max_visits_per_run: job.max_work.mask_node_visits },
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
        // Same explicit accounting as JobHostLimits::required_buffer_bytes.
        // It prices complete population/manifest/index copies, not an epoch.
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
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "extract-job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "extract-job operation refused")),
        };
        let args = JobArgs { extraction: extract::ExtractionArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }

    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let error = ErrorReport { schema_version: 1, operation, status: "error", code: failure.code,
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
    status: &'static str,
    code: &'static str,
    exit_code: ErrorCode,
    durable_progress_may_exist: bool,
    recovery: &'static str,
}
