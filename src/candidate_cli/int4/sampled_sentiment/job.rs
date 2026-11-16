//! Explicit result retention and authenticated replay of independent samples.
//! The immutable job ledger, not a restarted process, owns lifetime work.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use super::super::extract_job::StorageArgs;
use crate::{
    batch::{extract::ExtractionMaskBudget,
        sentiment::sampled::{SampledSentimentBatchLimits, sampled_sentiment_job_preparation_bytes}},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)] mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { sampled: SampledArgs, storage: StorageArgs }

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = compute::definition(input_encoding::definition(StorageArgs::augment_args(
            super::definition().name(name).about(about)
                .long_about("Retain and resume independent sampled sentiment for a frozen {id,text,task_args?} NDJSON population. task_args accepts axes and an optional complete tighter task budget; omitted axes selects all four affect dimensions. --seed and the complete sampling, grammar, encoding, prefill and per-item work policies must match on resume. Every axis and sample must complete and validate before one document is committed. Requires --store-results, an existing protected --job-dir, --key-file, --job-id and immutable --limits. All five native work counters and mask_node_visits in JobLimits.max_work are lifetime allowances, including failed attempts; item count, attempts and restart never replenish them. CLI work and mask flags cap ONE complete document, including every independent sample. Resume skips committed records and uses the same item-addressed draws for pending records. By default, repeat the complete original population; --spool-input and --use-retained-input explicitly enable authenticated private input retention. Stdout contains metadata only. --materialize publishes verified results after every record commits. One resident model, vocabulary and native allocation serve the invocation. --compute-threads 2..64 retains one numerical team; alternatively --prefill-rows 1..64 uses bounded layer-major prompts with serial compute. Compact safe-bpe-v1 input is supported. Memory is admitted before key/model/population IO, and cancellation keeps its original cause. No automatic retry, partial sample bundle, input truncation, calibrated confidence or default sentiment promotion. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
                .after_help(job_cli::INPUT_RETENTION_HELP)
                .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
                .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for each complete NDJSON envelope; immutable stored input ceiling must fit"))
                .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle cap per document; must fit JobLimits.max_result_bytes"))
                .mut_arg("max_forward_positions", |arg| arg.help("Per-document forward ceiling across every requested axis and sample"))
                .mut_arg("max_projected_logits", |arg| arg.help("Per-document full-vocabulary projection ceiling"))
                .mut_arg("max_attention_pairs", |arg| arg.help("Per-document attention-pair ceiling"))
                .mut_arg("max_dot_products", |arg| arg.help("Per-document grouped dot-product ceiling"))
                .mut_arg("max_multiply_accumulates", |arg| arg.help("Per-document multiply-accumulate ceiling"))
                .mut_arg("max_mask_node_visits", |arg| arg.help("Per-document complete mask allowance; lifetime ceiling comes from --limits"))
                .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; durable work and attempts never renew")))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private scores and explanations"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("sentiment-sampled-job")
        .about("Start or resume retained independent sentiment samples")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained sampled-sentiment population"))
        .subcommand(operation("resume", "Authenticate the population and sample pending documents"))
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    common: Int4Limits,
    task: TaskBudget,
    planning: SampledSentimentLimits,
    native: SampledSentimentBatchLimits,
    transport: PopulationReadLimits,
    buffer_bytes: u64,
    execution_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}

impl JobArgs {
    /// Settings and the complete one-engine geometry are checked before even
    /// the bounded limits file is opened. It alone supplies lifetime caps.
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let storage = &self.storage;
        storage.input_retention.limits().map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !storage.store_results || !local(&storage.job_dir) || !local(&storage.key_file)
            || !local(&storage.limits_file) || !(1..=1024 * 1024).contains(&storage.max_stream_mib)
            || !(1..=1_000_000_000).contains(&storage.max_input_lines)
            || storage.journal_memory_mib == 0 || storage.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_sampled_sentiment_job_storage"));
        }
        let common = self.sampled.validate(args)?;
        let task = self.sampled.budget(args, common, SentimentAxis::ALL.len())?;
        let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
        let preparation = sampled_sentiment_job_preparation_bytes(task, self.sampled.task_limits(args), &encoding)
            .map_err(|_| Failure::usage("sampled_sentiment_job_preparation"))?;
        if preparation > common.preparation_bytes {
            return Err(Failure::usage("sampled_sentiment_job_preparation"));
        }
        for mib in [storage.journal_memory_mib, storage.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("sampled_sentiment_job_memory_options"));
            }
        }
        Ok(common)
    }

    fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes || job.max_id_bytes > 256 {
            return Err(Failure::usage("sampled_sentiment_job_input_ceiling"));
        }
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("sampled_sentiment_job_result_ceiling"));
        }
        let task = self.sampled.budget(args, common, SentimentAxis::ALL.len())?;
        let mut planning = self.sampled.task_limits(args);
        planning.max_input_bytes = job.max_input_bytes_per_item;
        let native = SampledSentimentBatchLimits {
            max_item_work: self.sampled.work_ceiling(), max_model_work: job.max_work.int4_model_work(),
            masks: ExtractionMaskBudget { per_mask: self.sampled.masks(),
                max_visits_per_item: self.sampled.max_mask_node_visits,
                max_visits_per_run: job.max_work.mask_node_visits },
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        };
        native.validate().map_err(|_| Failure::usage("sampled_sentiment_job_lifetime_work"))?;
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        let buffer_bytes = sum(&[self.storage.input_retention.buffer_bytes(job).map_err(storage_failure)?,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes])?;
        // Sampled outputs own strings/counters rather than native token arrays.
        // The job-buffer quote separately covers durable frame/JSON staging.
        let output_bytes = sum(&[product(task.max_output_bytes, 4)?, 4096])?;
        let execution_bytes = sum(&[buffer_bytes, common.native.kv_bytes, common.native.rope_bytes,
            common.native.scratch_payload_bound, ALLOCATOR_BYTES, native.max_sampler_bytes,
            output_bytes, common.prefill.map_or(0, |prefill| prefill.max_extra_scratch_bytes),
            common.compute.workspace_bytes])?;
        let load_stage = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let save_stage = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
        common.minimum_memory_bytes = sum(&[common.preparation_bytes, common.load.weight_payload_bytes,
            ALLOCATOR_BYTES, load_stage.max(save_stage).max(execution_bytes), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
                code: "sampled_sentiment_job_population_memory", cancellation: None });
        }
        Ok(JobExecutionLimits { common, task, planning, native, transport, buffer_bytes, execution_bytes,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}

impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "sampled sentiment job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "sampled sentiment job operation refused")),
        };
        let job = JobArgs { sampled: SampledArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        let mut args = Int4Args::from_arg_matches(matches)?;
        args.input_encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        args.compute = compute::Args::from_matches(matches)?;
        Ok((Self { mode, args: job }, args))
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args,
        input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: "fnlp-candidate-int4-sentiment-sampled-job-v1",
                    schema_version: 1, operation, status: "error", code: failure.code,
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
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &args.input).map_err(storage_failure)?;
        let admitted = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, admitted, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, admitted, input, output);
            Err(Failure::usage("sampled_sentiment_job_profile_unavailable")) }
    }
}

fn storage_failure(error: job_cli::Failure) -> Failure {
    Failure { exit: error.exit, code: error.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("sampled_sentiment_job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}
