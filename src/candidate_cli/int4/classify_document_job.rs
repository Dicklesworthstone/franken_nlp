//! Retained classification of complete original documents, one attempt each.
//! Map policy and private label settings are immutable across authenticated resume.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::{classify::ClassificationBatchArgs, scored_int4::Int4ScoredBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode},
        scored_batch::{self as shared, Defaults, ScoreTask}},
    jobs::{JobLimits, population::PopulationReadLimits},
    tasks::ir::TaskBudget,
};
use super::{classify_map::{MapArgs, MapLimits}, extract_job::StorageArgs};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { map: MapArgs, storage: StorageArgs, classification: ClassificationArgs }

#[derive(Args)]
struct ClassificationArgs {
    /// Fixed private {labels,mode?,policy?} JSON; no document, budget or identity.
    #[arg(long, value_name = "FILE")]
    defaults: PathBuf,
}

pub(super) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = ClassificationArgs::augment_args(StorageArgs::augment_args(classify_map::definition()
            .name(name).about(about)
            .long_about("Retain and resume classification of COMPLETE original documents with fixed --defaults {labels,mode?,policy?}. By default, supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation; explicit retained-input replay is available. text is never a pre-split chunk and task_args may only be absent, null or an empty object. Each document is losslessly partitioned and every native chunk must finish before that original document commits. Retained results preserve full-vocabulary chunk scores, original byte/scalar coordinates and the explicitly named scalar-weighted exclusive or independent multi-label aggregate. --cohort-chunks shares native layers within a document; repeat the exact declared schedule and width on resume. Requires --store-results, protected existing --job-dir and --key-file, exact --job-id, immutable --limits and unchanged private label settings. All five per-document work flags cover every chunk; JobLimits.max_work is the sole lifetime allowance, including failed attempts, and mask_node_visits must be zero. Neither new chunks, attempts nor resume renew that work. Complete input, planning, native, retained-result and durable-job memory are admitted before reading the population. Stdout is metadata only; --materialize publishes private retained results after every document commits. No partial-document success, automatic retry, calibrated confidence or full-context equivalence. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for each complete NDJSON record and original document; immutable stored input ceiling must fit"))
            .mut_arg("max_map_result_bytes", |arg| arg.help("Complete original-document result cap; must fit immutable JobLimits.max_result_bytes"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable work and attempts never renew"))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-document classification results"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("classify-document-job")
        .about("Start or resume retained complete-document INT4 classification")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-document classification job"))
        .subcommand(operation("resume", "Authenticate the original population and classify pending documents"))
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    map: MapLimits,
    transport: PopulationReadLimits,
    buffer_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}
impl JobArgs {
    fn validate(&self, args: &Int4Args) -> Result<MapLimits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !local(&self.classification.defaults)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_classification_document_job_storage"));
        }
        let map = self.map.validate(args)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > map.common.memory_bytes) {
                return Err(Failure::usage("classification_document_job_memory"));
            }
        }
        Ok(map)
    }
    fn load_settings(&self, text: &str, budget: TaskBudget, args: &Int4Args, map: MapLimits)
        -> Result<ClassificationBatchArgs, CandidateError> {
        let Defaults::Classify(Some(settings)) = shared::parse_defaults_with_limits(ScoreTask::Classify,
            map.planning, args.max_input_bytes, Some(text), budget)? else {
            return Err(CandidateError::Input);
        };
        self.map.admit_labels(settings.labels.len(), settings.mode)?;
        Ok(settings)
    }
    fn lifetime_limits(&self, args: &Int4Args, mut map: MapLimits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if map.memory.max_output_bytes > job.max_result_bytes as u64 {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = Int4ScoredBatchLimits { max_item_work: map.mapping.max_model_work,
            max_model_work: job.max_work.int4_model_work() };
        if job.max_work.mask_node_visits != 0 || native.validate().is_err() {
            return Err(Failure::usage("lifetime_classification_document_work"));
        }
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = map.common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Match the common durable host quote. It coexists with the entire
        // map's planning, native, retained-result and complete-output owners.
        let buffer_bytes = sum(&[product(job.max_snapshot_bytes, 2)?, product(job.max_items, 1024)?,
            product(job.max_input_bytes_per_item as u64, 8)?, product(job.max_result_bytes as u64, 4)?,
            2 * MIB, preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes])?;
        map.common.minimum_memory_bytes = map.common.minimum_memory_bytes.checked_add(buffer_bytes)
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if map.common.minimum_memory_bytes > map.common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { map, transport, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}
impl JobCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "classification document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "classification document job operation refused")),
        };
        let args = JobArgs { map: MapArgs::from_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)?,
            classification: ClassificationArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &args.input)?;
        let map = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, map, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, map, input, output); Err(Failure::usage("owned_classification_document_job_profile_unavailable")) }
    }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
