//! Explicit durable schema extraction on one resident grouped-INT4 model.
//! Frozen lifetime work is copied exactly; invocation limits never renew it.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::extract::{ExtractionMaskBudget, grouped_int4::Int4ExtractionBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobId, JobLimits, population::PopulationReadLimits},
    tokenizer::untrusted::SafeBpeWorkLimits,
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
    encoding: EncodingArgs,
}
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EncodingArgs {
    /// Total schema/source bytes encoded in this invocation; safe-bpe-v1 only, default 67108864.
    #[arg(long)]
    max_total_tokenizer_input_bytes: Option<u64>,
    /// Total untrusted tokens encoded in this invocation; safe-bpe-v1 only, default 4194304.
    #[arg(long)]
    max_total_tokenizer_output_tokens: Option<u64>,
    /// Total BPE pair checks in this invocation; safe-bpe-v1 only, default 268435456.
    #[arg(long)]
    max_total_tokenizer_pair_checks: Option<u64>,
    /// Total examined pair bytes in this invocation; safe-bpe-v1 only, default 4294967296.
    #[arg(long)]
    max_total_tokenizer_pair_bytes: Option<u64>,
}
impl EncodingArgs {
    fn limits(self, args: &Int4Args) -> Result<Option<SafeBpeWorkLimits>, Failure> {
        if args.input_encoding.is_default() {
            return if self == Self::default() { Ok(None) }
                else { Err(Failure::usage("extract_job_input_encoding")) };
        }
        let total = SafeBpeWorkLimits {
            max_input_bytes: self.max_total_tokenizer_input_bytes.unwrap_or(64 * MIB),
            max_output_tokens: self.max_total_tokenizer_output_tokens.unwrap_or(4 * MIB),
            max_pair_checks: self.max_total_tokenizer_pair_checks.unwrap_or(256 * MIB),
            max_pair_bytes: self.max_total_tokenizer_pair_bytes.unwrap_or(4096 * MIB),
        };
        total.validate().map_err(|_| Failure::usage("extract_job_tokenizer_work"))?;
        Ok(Some(total))
    }
}
#[derive(Args)]
pub(super) struct StorageArgs {
    /// Required consent to retain private extraction results in the owner-only spool.
    #[arg(long)]
    pub store_results: bool,
    #[command(flatten)]
    pub input_retention: job_cli::InputRetentionArgs,
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
        input_encoding::definition(compute::definition(EncodingArgs::augment_args(
            StorageArgs::augment_args(extract::definition().name(name).about(about)
            .long_about("Retain and resume schema/source extraction with one resident source-converted or independently pinned snapshot INT4 model. Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, and one exact shared --schema. By default, supply the COMPLETE original NDJSON population on every invocation; explicit retained-input replay is available. Non-null task_args and flush controls are refused. Resume authenticates exact input bytes, schema, model/group identity and recipe before replay or tail repair. Lifetime native work includes every INT4 group dot and does not renew on resume; failed attempts remain debited. Stdout is metadata only; retained private results are read through explicit job verification/materialization. --materialize publishes only after all items commit. Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. Serial execution remains the default. --compute-threads 2..=64 reuses one admitted numerical row team across the invocation; physical width may change on authenticated resume without changing the recipe or replenishing any lifetime work. Worker scratch and the installed thread envelope are priced before private IO. No prefill, cohorts, selected-row override, automatic retry, key/directory creation, release activation or semantic-accuracy claim. --input-encoding safe-bpe-v1 explicitly preserves source/schema bytes while encoding them compactly. The two segments share each record's tokenizer allowance; all records, failed preparations and reconstruction also debit one invocation ledger. The four --max-total-tokenizer-* ceilings and per-record policy are authenticated: repeat them on resume. A resumed invocation receives only that declared tokenizer allowance; persisted native work and attempts remain spent. Encoder workspace is admitted beside CLI preparation and owned job input. There is no byte fallback, truncation or claim of identical model output.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer cap on each complete NDJSON record; immutable JobLimits.max_input_bytes_per_item must fit"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this complete invocation; persisted work and attempt ceilings do not renew"))))))
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

/// Host reservations include the selected population owner and full descriptor
/// inventory. The CLI keeps its own preparation charge live too.
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
    pub encoding_workspace_bytes: u64,
    pub encoding_work_limits: Option<SafeBpeWorkLimits>,
}

impl JobArgs {
    /// Pure option validation, before reading any key, limits, schema or input.
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_storage_or_job_options"));
        }
        // Physical workers and explicitly authenticated compact input share
        // the full-head, sequential-prompt extraction contract.
        if args.prefill()?.is_some() { return Err(CandidateError::Arguments.into()); }
        self.encoding.limits(args)?;
        let common = self.extraction.validate_batch_with_prefill(args).map_err(Failure::from)?;
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
        let encoding = self.extraction.encoding_policy(args, common.max_prompt_tokens)?;
        let encoding_workspace_bytes = crate::batch::extract::grouped_int4::memory::encoding_workspace_bytes(&encoding)
            .map_err(|_| Failure::usage("extract_job_input_encoding"))?;
        let encoding_work_limits = self.encoding.limits(args)?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Share the host's physical input-owner quote; descriptor/index and
        // retained-result staging remain present in both input modes.
        let buffer_bytes = sum(&[
            self.storage.input_retention.buffer_bytes(job)?,
            preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes, encoding_workspace_bytes,
        ]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        common.minimum_memory_bytes = common.minimum_memory_bytes.checked_add(buffer_bytes)
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { common, transport, native, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes,
            encoding_workspace_bytes, encoding_work_limits })
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
            storage: StorageArgs::from_arg_matches(matches)?, encoding: EncodingArgs::from_arg_matches(matches)? };
        let mut native = Int4Args::from_arg_matches(matches)?;
        native.compute = compute::Args::from_matches(matches)?;
        native.input_encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        Ok((Self { mode, args }, native))
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
        self.args.storage.input_retention.validate(self.mode, &args.input)?;
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
