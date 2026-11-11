//! Retained sentiment of complete original documents, one attempt each.
//! Fixed axes, policy, mapping and native schedule authenticate every resume.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use super::super::extract_job::StorageArgs;
use crate::{
    batch::{scored_int4::Int4ScoredBatchLimits,
        sentiment::grouped_int4::SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode},
        scored_batch::{self as shared, Defaults, ScoreTask}},
    jobs::{JobLimits, population::PopulationReadLimits},
    tasks::sentiment::{SentimentPolicy, batch::SentimentBatchArgs},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { map: MapArgs, storage: StorageArgs, sentiment: SentimentArgs }

#[derive(Args)]
struct SentimentArgs {
    /// Fixed private {axes?,policy?} JSON; absent uses all four default axes.
    #[arg(long, value_name = "FILE")]
    defaults: Option<PathBuf>,
    /// Optional hard axis cap; omission preserves existing four-axis budgeting.
    #[arg(long)]
    max_axes: Option<usize>,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = SentimentArgs::augment_args(StorageArgs::augment_args(super::one_off_definition()
            .name(name).about(about)
            .long_about("Retain and resume dimensional sentiment of COMPLETE original documents with fixed optional --defaults {axes?,policy?}; omission selects all four axes and the standard policy. By default, supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation; explicit retained-input replay is available. Text is never a pre-split chunk and task_args may only be absent, null or an empty object. Every chunk and requested axis must finish before the original document commits. Retained results include complete per-chunk distributions, original byte/scalar coordinates and named scalar-weighted descriptive aggregates. Optional --max-axes bounds the fixed dimension selection and can reduce admitted memory; a bound below four requires --defaults selecting no more than that many axes. An explicit axis bound times --cohort-chunks must fit 64 heads. Omitting --max-axes preserves existing four-axis task budgeting and checks the actual fixed selection. --cohort-chunks shares native layers within one document; repeat the exact task budget, schedule and width on resume. Requires --store-results, protected existing --job-dir and --key-file, exact --job-id, immutable --limits and unchanged axes/policy. All five per-document work flags cover every chunk and axis; JobLimits.max_work is the sole lifetime allowance, including failed attempts, and mask_node_visits must be zero. Chunks, attempts and resume never renew that allowance. Complete input, planning, native, retained-result and durable-job memory are admitted before population IO. Stdout is metadata only; --materialize publishes private results after every document commits. No partial-document success, automatic retry, calibrated confidence or full-context equivalence. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for each complete NDJSON record and original document; immutable stored input ceiling must fit"))
            .mut_arg("max_map_result_bytes", |arg| arg.help("Complete semantic document cap; stored result ceiling also covers cohort framing when enabled"))
            .mut_arg("cohort_chunks", |arg| arg.help("Share native layers across N chunks; actual axes and any explicit --max-axes times N must be <=64"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable work and attempts never renew"))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-document sentiment results"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("sentiment-document-job")
        .about("Start or resume retained complete-document INT4 sentiment")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-document sentiment job"))
        .subcommand(operation("resume", "Authenticate the original population and score pending documents"))
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
    /// Pure admission precedes key, defaults, lifetime limits and source IO.
    fn validate(&self, args: &Int4Args) -> Result<MapLimits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || self.sentiment.defaults.as_ref().is_some_and(|path| !local(path))
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_sentiment_document_job_storage"));
        }
        if self.sentiment.max_axes.is_some_and(|axes| !(1..=4).contains(&axes)
            || self.map.mapping.cohort_chunks.is_some_and(|chunks|
                chunks.checked_mul(axes).is_none_or(|heads| heads == 0 || heads > 64))) {
            return Err(Failure::usage("sentiment_document_job_geometry"));
        }
        // Omission retains the existing four-axis task and memory authority;
        // only an explicit cap imposes a declared window before defaults IO.
        let map = self.map.validate_with_max_axes(args, self.sentiment.max_axes.unwrap_or(4))?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > map.common.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok(map)
    }
    fn load_settings(&self, text: Option<&str>, budget: TaskBudget, args: &Int4Args)
        -> Result<(SentimentBatchArgs, SentimentPolicy), CandidateError> {
        let Defaults::Sentiment { args: settings, policy } = shared::parse_defaults_with_limits(
            ScoreTask::Sentiment, self.map.scoring.classification_limits(args),
            args.max_input_bytes, text, budget)? else { return Err(CandidateError::Input); };
        // Fixed settings cannot expand an explicit capability. With no cap,
        // preserve the existing actual-axis check and four-axis task budget.
        if self.sentiment.max_axes.is_some_and(|axes| settings.axes.len() > axes)
            || self.map.mapping.cohort_chunks.is_some_and(|chunks|
                chunks.checked_mul(settings.axes.len()).is_none_or(|heads| heads > 64)) {
            return Err(CandidateError::Planning);
        }
        Ok((settings, policy))
    }
    fn result_bytes(&self, map: MapLimits) -> Result<u64, CandidateError> {
        map.mapping.max_result_bytes.checked_add(if self.map.mapping.cohort_chunks.is_some() { 4096 } else { 0 })
            .ok_or(CandidateError::Arguments)
    }
    fn lifetime_limits(&self, args: &Int4Args, mut map: MapLimits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        // The transport option may be more generous, but the durable source
        // capability is exactly the immutable stored per-item ceiling. Keep
        // the larger pre-I/O memory quote as a conservative reservation.
        map.mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
        if self.result_bytes(map)? > job.max_result_bytes as u64 {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = Int4ScoredBatchLimits { max_item_work: map.mapping.max_model_work,
            max_model_work: job.max_work.int4_model_work() };
        if job.max_work.mask_node_visits != 0 || native.validate().is_err() {
            return Err(Failure::usage("lifetime_sentiment_document_work"));
        }
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = map.common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Durable population owners coexist with the complete map quote.
        let buffer_bytes = sum(&[self.storage.input_retention.buffer_bytes(job)?,
            preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes])?;
        // Freeze/replan identities coexist during replay; a one-shot map has
        // no such durable descriptors, so its quote alone is insufficient.
        let chunk_metadata_bytes = product(map.mapping.chunks.max_chunks as u64,
            SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES)?;
        map.common.minimum_memory_bytes = sum(&[map.common.minimum_memory_bytes,
            buffer_bytes, chunk_metadata_bytes])?;
        if map.common.minimum_memory_bytes > map.common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { map, transport, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}
impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "sentiment document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "sentiment document job operation refused")),
        };
        let args = JobArgs { map: MapArgs::from_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)?,
            sentiment: SentimentArgs::from_arg_matches(matches)? };
        let mut native = Int4Args::from_arg_matches(matches)?;
        native.input_encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        Ok((Self { mode, args }, native))
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args,
        input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
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
        { let _ = (self, args, map, input, output); Err(Failure::usage("owned_sentiment_document_job_profile_unavailable")) }
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
