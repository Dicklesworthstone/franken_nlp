//! Explicit retained independent sentiment samples. The original seed,
//! complete task policy and lifetime native/mask limits authenticate resume.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget, sentiment::sampled::{SampledSentimentBatchLimits,
        sampled_sentiment_batch_preparation_bytes}},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits},
};
use super::super::extract_job::StorageArgs;

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { sampled: SampledArgs, storage: StorageArgs }

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about("Retain and resume independently sampled sentiment bundles for a frozen NDJSON {id,text,task_args?} population. task_args accepts axes and an optional complete task budget that may only tighten the invocation ceilings. Requires --seed, --store-results, an existing protected --job-dir, the original --job-id and --key-file, and immutable --limits. Resume authenticates every original record, effective seed, sample count, sampling policy, input encoding, model/group identity and planner limits before repair or native execution. A document commits only after every requested score/explanation sample validates. Committed items are skipped; failed attempts retain their complete reserved work and masks against JobLimits.max_work. Five native flags and --max-mask-node-visits bound EACH ITEM; the limits file is the sole lifetime authority. No work is replenished by resume. By default supply the COMPLETE original population each time, or explicitly retain and replay inputs. Stdout contains metadata only; private scores and explanations remain in the protected result spool. --materialize publishes only after all items commit. The model and decoder vocabulary load once. --compute-threads 2..64 retains one numerical team across the entire invocation; alternatively --prefill-rows 1..64 uses bounded layer-major prompts with serial compute. Physical schedule changes do not change sample addresses or lifetime work. Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. This is an explicit candidate with no quality, confidence or speed certification. No automatic retry, key/directory creation, thinking, tools or flush controls.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for each complete NDJSON record; immutable JobLimits.max_input_bytes_per_item must fit"))
            .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle per item; must fit immutable JobLimits.max_result_bytes"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; persisted work and attempts do not renew"))
            .mut_arg("max_forward_positions", |arg| arg.help("Per-item forward ceiling across every axis and sample; lifetime ceiling comes from --limits"))
            .mut_arg("max_projected_logits", |arg| arg.help("Per-item full-vocabulary projection ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_attention_pairs", |arg| arg.help("Per-item attention-pair ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_dot_products", |arg| arg.help("Per-item grouped dot-product ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_multiply_accumulates", |arg| arg.help("Per-item multiply-accumulate ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_mask_node_visits", |arg| arg.help("Per-item complete mask allowance; lifetime ceiling comes from --limits")))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private sentiment samples and explanations in the owner-only spool"));
        compute::definition(input_encoding::definition(command))
    };
    let start = operation("start", "Create an explicitly retained sampled sentiment job");
    let resume = operation("resume", "Authenticate the frozen population and run pending sampled sentiment items")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new("sentiment-sampled-job")
        .about("Start or resume durable independent sentiment samples and explanations")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    common: Int4Limits,
    transport: PopulationReadLimits,
    native: SampledSentimentBatchLimits,
    buffer_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}
impl JobArgs {
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_sampled_sentiment_job_storage"));
        }
        // This quotes the complete bundle, one sampler, the selected persistent
        // team OR prompt workspace, and simultaneous caller/host preparation.
        let common = self.sampled.validate(args)?;
        let task = self.sampled.budget(args, common, SentimentAxis::ALL.len())?;
        let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
        let preparation = sampled_sentiment_batch_preparation_bytes(task,
            self.sampled.task_limits(args), &encoding).map_err(|_| CandidateError::Arguments)?;
        if preparation > common.preparation_bytes { return Err(CandidateError::Arguments.into()); }
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("sampled_sentiment_job_memory"));
            }
        }
        Ok(common)
    }

    fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = SampledSentimentBatchLimits {
            max_item_work: self.sampled.work_ceiling(), max_model_work: job.max_work.int4_model_work(),
            masks: ExtractionMaskBudget { per_mask: self.sampled.masks(),
                max_visits_per_item: self.sampled.max_mask_node_visits,
                max_visits_per_run: job.max_work.mask_node_visits },
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        };
        native.validate().map_err(|_| Failure::usage("lifetime_sampled_sentiment_work_limits"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        let buffer_bytes = sum(&[self.storage.input_retention.buffer_bytes(job)?,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes,
            serialization_reserve_bytes]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        // SampledArgs already includes host preparation alongside the caller's
        // charge. Replace that one host owner with its complete job-buffer
        // quote; do not price the same preparation owner twice.
        common.minimum_memory_bytes = common.minimum_memory_bytes
            .checked_sub(preparation_reserve_bytes).and_then(|n| n.checked_add(buffer_bytes))
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { common, transport, native, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
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
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "sampled sentiment job operation refused")),
        };
        let args = JobArgs { sampled: SampledArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        let mut native = Int4Args::from_arg_matches(matches)?;
        native.input_encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        native.compute = compute::Args::from_matches(matches)?;
        Ok((Self { mode, args }, native))
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(
        self, args: Int4Args, input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
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
        let common = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, common, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, common, input, output);
            Err(Failure::usage("owned_sampled_sentiment_job_profile_unavailable")) }
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
