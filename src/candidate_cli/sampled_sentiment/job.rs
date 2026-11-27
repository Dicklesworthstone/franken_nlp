//! Explicit retained INT8 sample bundles under an authenticated lifetime ledger.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]
use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget, sentiment::sampled::sampled_sentiment_job_preparation_bytes},
    candidate_cli::jobs::{self as job_cli, RunMode},
    jobs::{JobId, JobLimits},
    tokenizer::untrusted::SafeBpeWorkLimits,
};

pub(in crate::candidate_cli) struct JobCommand { mode: RunMode, args: JobArgs }
pub(in crate::candidate_cli) struct JobArgs {
    pub sampled: SampledSentimentCommand,
    pub storage: StorageArgs,
}
#[derive(Args)]
pub(in crate::candidate_cli) struct StorageArgs {
    /// Required consent to retain private scores and explanations.
    #[arg(long)] pub store_results: bool,
    #[command(flatten)] pub input_retention: job_cli::InputRetentionArgs,
    #[command(flatten)] pub encoding_work: job_cli::EncodingWorkArgs,
    /// Existing owner-only directory; never created or silently adopted.
    #[arg(long, value_name = "DIR")] pub job_dir: PathBuf,
    /// Expected random 128-bit job ID, exactly 32 lowercase hexadecimal characters.
    #[arg(long, value_parser = job_cli::parse_job_id, value_name = "HEX")] pub job_id: JobId,
    /// Protected regular file with exactly 32 raw secret bytes, not a hexadecimal key.
    #[arg(long, value_name = "FILE")] pub key_file: PathBuf,
    /// Exact immutable JobLimits JSON, including lifetime work and attempt ceilings.
    #[arg(long = "limits", value_name = "FILE")] pub limits_file: PathBuf,
    /// Publish verified materialized.ndjson after ALL items commit; never replace it.
    #[arg(long)] pub materialize: bool,
    /// Per-invocation transport ceiling, separate from immutable population limits.
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    /// Includes blank lines; durable populations do not accept flush controls.
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about("Retain complete independent INT8 score/explanation bundles for a frozen {id,text,task_args?} NDJSON population. task_args accepts axes and an optional complete tighter task budget; omitted axes selects all four dimensions. Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, and an explicit --seed. Every axis and sample must complete and validate before one document commits. Seed, sampler, grammar, task, encoding, tokenizer totals, per-item work/mask policy and prompt schedule must match on resume. JobLimits.max_work and attempts are lifetime allowances, including failed attempts, and never renew. CLI native/mask flags cap ONE complete document. Resume skips committed records and preserves item-addressed draws for pending records. One resident engine and optional persistent --compute-threads team serve the invocation; worker width may change on resume. --prefill-rows composes with workers and stays frozen in the recipe. Stdout is metadata only; --materialize publishes verified results only after every item commits. No automatic retry, key/directory creation, partial bundle or implicit input retention. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(format!("Safe-BPE per-record limits and the four --max-total-tokenizer-* invocation ceilings are authenticated and must match on resume. Failed preparation spends the same invocation ledger. A resumed invocation receives that declared tokenizer allowance while durable native/mask work and attempts remain spent. Byte fallback refuses total-tokenizer flags.\n\n{}", job_cli::INPUT_RETENTION_HELP))
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for one complete NDJSON envelope; the immutable item ceiling must fit"))
            .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle per document; must fit JobLimits.max_result_bytes"))
            .mut_arg("max_mask_node_visits", |arg| arg.help("Per-document mask ceiling; lifetime mask allowance comes from --limits"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; durable work and attempts never renew")));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("sentiment-sampled-job")
        .about("Start or resume retained independent INT8 sentiment samples")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a new retained sampled-sentiment population"))
        .subcommand(operation("resume", "Authenticate the original contract and sample pending documents"))
}

/// Pure configuration remains available when the optional job backend is absent.
#[derive(Clone, Copy, Debug)]
pub(in crate::candidate_cli) struct Envelope {
    pub task: TaskBudget,
    pub planning: SampledSentimentLimits,
    pub max_item_work: Int8Work,
    pub masks: ExtractionMaskBudget,
    pub sampler_bytes: u64,
    pub preparation_bytes: u64,
}
impl JobArgs {
    pub(in crate::candidate_cli) fn encoding_work_limits(&self) -> Result<Option<SafeBpeWorkLimits>, Failure> {
        self.storage.encoding_work.limits(&self.sampled.encoding).map_err(storage_failure)
    }
    pub(in crate::candidate_cli) fn validate(&self) -> Result<(CandidateArgs, Limits), Failure> {
        self.storage.input_retention.limits().map_err(storage_failure)?;
        self.encoding_work_limits()?;
        let storage = &self.storage;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !storage.store_results || !local(&storage.job_dir) || !local(&storage.key_file)
            || !local(&storage.limits_file) || !(1..=1024 * 1024).contains(&storage.max_stream_mib)
            || !(1..=1_000_000_000).contains(&storage.max_input_lines)
            || storage.journal_memory_mib == 0 || storage.serialization_memory_mib == 0 {
            return Err(Failure::usage("int8_sampled_sentiment_job_storage"));
        }
        let (common, limits) = self.sampled.common()
            .map_err(|_| Failure::usage("int8_sampled_sentiment_job_options"))?;
        for mib in [storage.journal_memory_mib, storage.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > limits.memory_bytes) {
                return Err(Failure::usage("int8_sampled_sentiment_job_memory"));
            }
        }
        let task = self.sampled.budget(limits, SentimentAxis::ALL.len())
            .map_err(|_| Failure::usage("int8_sampled_sentiment_job_options"))?;
        self.preparation(limits, task, self.sampled.task_limits())?;
        Ok((common, limits))
    }
    fn preparation(&self, limits: Limits, task: TaskBudget, planning: SampledSentimentLimits)
        -> Result<u64, Failure> {
        let encoding = self.sampled.encoding.policy(self.sampled.max_input_bytes, limits.max_prompt_tokens)
            .map_err(|_| Failure::usage("int8_sampled_sentiment_job_encoding"))?;
        let base = sampled_sentiment_job_preparation_bytes(task, planning, &encoding)
            .map_err(|_| Failure::usage("int8_sampled_sentiment_job_preparation"))?;
        // The private invocation ledger coexists with the encoder workspace.
        let ledger = u64::from(self.encoding_work_limits()?.is_some()) * 1024;
        let required = base.checked_add(ledger)
            .filter(|&bytes| bytes <= limits.preparation_bytes)
            .ok_or_else(|| Failure::usage("int8_sampled_sentiment_job_preparation"))?;
        Ok(required)
    }
    pub(in crate::candidate_cli) fn lifetime_limits(&self, limits: Limits, job: JobLimits)
        -> Result<Envelope, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > self.sampled.max_input_bytes || job.max_id_bytes > 256 {
            return Err(Failure::usage("int8_sampled_sentiment_job_input_ceiling"));
        }
        if limits.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("int8_sampled_sentiment_job_result_ceiling"));
        }
        let item = self.sampled.work_ceiling();
        let total = job.max_work.model;
        if item.forward_positions > total.forward_positions || item.projected_logits > total.projected_logits
            || item.attention_pairs > total.attention_pairs || !item.projections.fits(total.projections)
            || self.sampled.max_mask_node_visits > job.max_work.mask_node_visits {
            return Err(Failure::usage("int8_sampled_sentiment_job_lifetime_work"));
        }
        let task = self.sampled.budget(limits, SentimentAxis::ALL.len())
            .map_err(|_| Failure::usage("int8_sampled_sentiment_job_options"))?;
        let mut planning = self.sampled.task_limits();
        planning.max_input_bytes = job.max_input_bytes_per_item;
        let preparation_bytes = self.preparation(limits, task, planning)?;
        Ok(Envelope { task, planning, max_item_work: item,
            masks: ExtractionMaskBudget { per_mask: self.sampled.masks(),
                max_visits_per_item: self.sampled.max_mask_node_visits,
                max_visits_per_run: job.max_work.mask_node_visits },
            sampler_bytes: json_sampling_workspace_bytes()
                .map_err(|_| Failure::usage("int8_sampled_sentiment_job_sampler"))?,
            preparation_bytes })
    }
}
impl JobCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, request) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "sampled sentiment job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: request.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "sampled sentiment job operation refused")),
        };
        Ok(Self { mode, args: JobArgs {
            sampled: SampledSentimentCommand::from_matches(request)?,
            storage: StorageArgs::from_arg_matches(request)?,
        } })
    }
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let mode = self.mode;
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: "fnlp-candidate-int8-sentiment-sampled-job-v1",
                    schema_version: 1, operation: mode.name(), status: "error", code: failure.code,
                    exit_code: failure.exit, cancellation: failure.cancellation,
                    durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &self.args.sampled.input).map_err(storage_failure)?;
        let (common, limits) = self.args.validate()?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::job::execute(self.mode, self.args, common, limits, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, common, limits, input, output);
            Err(Failure::usage("int8_sampled_sentiment_job_profile_unavailable")) }
    }
}
pub(in crate::candidate_cli) fn storage_failure(error: job_cli::Failure) -> Failure {
    Failure { exit: error.exit, code: error.code, cancellation: None }
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}

#[cfg(test)] pub(in crate::candidate_cli) mod tests;
