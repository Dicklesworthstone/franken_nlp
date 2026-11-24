//! Explicit retained semantic results, with immutable two-stage lifetime budgets.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]
use super::*;
use crate::{
    batch::extract::{ExtractionMaskBudget, semantic::int8::Int8SemanticExtractionBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, RunMode}, stream::Failure},
    jobs::{JobId, JobLimits, population::PopulationReadLimits},
    tokenizer::untrusted::SafeBpeWorkLimits,
};

pub(crate) struct JobCommand {
    pub(in crate::candidate_cli) mode: RunMode,
    pub(in crate::candidate_cli) semantic: SemanticCommand,
    pub(in crate::candidate_cli) storage: StorageArgs,
    encoding: InvocationEncodingArgs,
}
#[derive(Args)]
pub(in crate::candidate_cli) struct StorageArgs {
    /// Required consent to retain original extraction JSON and private field diagnostics.
    #[arg(long)] pub store_results: bool,
    #[command(flatten)] pub input_retention: job_cli::InputRetentionArgs,
    /// Existing owner-only directory; never created or silently adopted.
    #[arg(long, value_name = "DIR")] pub job_dir: PathBuf,
    /// Expected random job ID: exactly 32 lowercase hexadecimal characters.
    #[arg(long, value_parser = job_cli::parse_job_id, value_name = "HEX")] pub job_id: JobId,
    /// Protected regular file containing exactly 32 raw secret bytes.
    #[arg(long, value_name = "FILE")] pub key_file: PathBuf,
    /// Exact immutable JobLimits JSON; sole authority for lifetime work and attempts.
    #[arg(long = "limits", value_name = "FILE")] pub limits_file: PathBuf,
    /// Publish verified materialized.ndjson only after all complete items commit.
    #[arg(long)] pub materialize: bool,
    /// Per-invocation input transport, in addition to immutable population limits.
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    /// Counts blank lines too. Durable populations do not accept flush controls.
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    /// Modeled database RAM, separately charged from journal file size.
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    /// Additional canonical JSON/serializer staging beyond complete result buffers.
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        InvocationEncodingArgs::augment_args(StorageArgs::augment_args(super::definition()
            .name(name).about(about)
            .long_about(concat!(
                "Run a fixed exact-schema extraction followed by all configured same-model field reviews for each original document. ",
                "The retained item contains both the unchanged independently validated JSON and complete correlated diagnostics; a failed stage commits neither. ",
                "Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, --schema and --semantic-options. ",
                "Non-null task_args and flush controls are refused. The complete original population is supplied on every invocation unless explicitly retained. ",
                "Resume authenticates exact input, schema, extraction/judge identities, all options and tokenizer policy before repair or native inference. ",
                "JobLimits.max_work is the sole lifetime authority for all five native axes and extraction masks. ",
                "Each admitted attempt durably spends options.max_model_work plus its mask allowance before the first extraction forward. ",
                "Early EOS, failed reviews, cancelled publication, retries, flushes and resume never refund or multiply lifetime work. ",
                "Committed items are skipped. Stdout is metadata only; private results stay in the protected spool or explicit materialization. ",
                "This remains experimental, uncalibrated same-model diagnostic output, without independent proof, automatic acceptance or field repair.")))
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer decoded/record cap; immutable JobLimits.max_input_bytes_per_item must fit"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable native work and attempts never renew"))
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .after_long_help(concat!(
                "--compute-threads 2..=64 and --prefill-rows 1..=64 compose on one persistent numerical team across extraction, reviews and pending items. ",
                "Physical widths may change on authenticated resume without changing the recipe or replenishing lifetime work. ",
                "Compact safe-bpe-v1 has one per-complete-attempt --max-tokenizer-* policy and a separate actual cumulative --max-total-tokenizer-* allowance per invocation. ",
                "Both are authenticated and must match on resume. Only the declared tokenizer invocation allowance is fresh; durable attempts/native work/masks remain spent. ",
                "All failed preparation, dynamic reviews and reconstruction debit the same invocation ledger. Byte-fallback omits compact metadata and refuses tokenizer-only flags. ",
                "Bounded options and lifetime limits are read under caller preparation first; complete source/review/recipe/population/native/team/output memory is then admitted before key, schema, population or model IO. ",
                "Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. No automatic retry, key/directory creation, selected rows, cohorts, thinking, sampling, tools or quality promotion.")))
    };
    let start = operation("start", "Create a retained native INT8 extraction-and-review job");
    let resume = operation("resume", "Authenticate and finish pending INT8 extraction-and-review items")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new("extract-semantic-job")
        .about("Start or resume complete retained INT8 semantic extraction results")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

#[derive(Clone, Copy)]
pub(in crate::candidate_cli) struct JobEnvelope {
    pub native: Int8SemanticExtractionBatchLimits,
    pub transport: PopulationReadLimits,
    pub preparation_bytes: u64,
    pub io_bytes: u64,
    pub journal_bytes: u64,
    pub serialization_bytes: u64,
    pub encoding_work_limits: Option<SafeBpeWorkLimits>,
}
impl JobCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "extract-semantic-job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "extract-semantic-job operation refused")),
        };
        Ok(Self { mode, semantic: SemanticCommand::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)?,
            encoding: InvocationEncodingArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli) fn validate(&self) -> Result<(CandidateArgs, Limits), Failure> {
        let storage = &self.storage;
        storage.input_retention.validate(self.mode, &self.semantic.input).map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !storage.store_results || !local(&storage.job_dir) || !local(&storage.key_file)
            || !local(&storage.limits_file) || !(1..=1024 * 1024).contains(&storage.max_stream_mib)
            || !(1..=1_000_000_000).contains(&storage.max_input_lines)
            || storage.journal_memory_mib == 0 || storage.serialization_memory_mib == 0 {
            return Err(Failure::usage("semantic_job_storage_options"));
        }
        self.encoding_work_limits()?;
        let (args, limits) = self.semantic.validate().map_err(population_failure)?;
        for mib in [storage.journal_memory_mib, storage.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > limits.memory_bytes) {
                return Err(Failure::usage("semantic_job_memory_options"));
            }
        }
        Ok((args, limits))
    }

    pub(in crate::candidate_cli) fn encoding_work_limits(&self) -> Result<Option<SafeBpeWorkLimits>, Failure> {
        self.encoding.limits(&self.semantic.encoding_policy().map_err(population_failure)?)
            .map_err(population_failure)
    }

    pub(in crate::candidate_cli) fn lifetime_limits(&self, args: &CandidateArgs, limits: Limits,
        options: &Int8SemanticOptions, job: JobLimits) -> Result<JobEnvelope, Failure> {
        job.validate().map_err(|_| Failure::usage("semantic_job_limits"))?;
        self.semantic.admit_options(args, limits, options).map_err(population_failure)?;
        if job.max_input_bytes_per_item > self.semantic.host.max_input_bytes
            || options.limits.max_output_bytes > job.max_result_bytes as u64 {
            return Err(Failure::usage("semantic_job_complete_item_ceiling"));
        }
        let native = Int8SemanticExtractionBatchLimits {
            max_model_work: job.max_work.model,
            masks: ExtractionMaskBudget { per_mask: self.semantic.host.masks(),
                max_visits_per_item: self.semantic.host.max_mask_node_visits,
                max_visits_per_run: job.max_work.mask_node_visits },
        };
        native.validate_item(options.max_model_work)
            .map_err(|_| Failure::usage("semantic_job_lifetime_work"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("semantic_job_transport"))?;
        Ok(JobEnvelope { native, transport, preparation_bytes: limits.preparation_bytes,
            io_bytes: product(IO_BUFFER_BYTES as u64, 2)?,
            journal_bytes: product(self.storage.journal_memory_mib, MIB)?,
            serialization_bytes: product(self.storage.serialization_memory_mib, MIB)?,
            encoding_work_limits: self.encoding_work_limits()? })
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let error = ErrorReport { protocol: PROTOCOL, schema_version: 1, operation,
                    status: "error", code: failure.code, exit_code: failure.exit,
                    cancellation: failure.cancellation, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&error, job_cli::REPORT_BYTES, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }

    fn execute_owned<R: Read + Send + 'static>(self, input: R, output: &mut impl Write) -> Result<(), Failure> {
        let (args, limits) = self.validate()?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::semantic::jobs::execute(self, args, limits, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, limits, input, output); Err(Failure::usage("semantic_job_profile_unavailable")) }
    }
}
pub(in crate::candidate_cli) fn storage_failure(error: job_cli::Failure) -> Failure {
    Failure { exit: error.exit, code: error.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("semantic_job_memory_arithmetic"))
}
pub(in crate::candidate_cli) const PROTOCOL: &str = "fnlp-candidate-int8-semantic-extraction-job-v1";
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}
#[cfg(test)] mod tests;
