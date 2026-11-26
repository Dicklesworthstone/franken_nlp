//! Explicit retained generation/chat; the live text-batch protocol is unchanged.
use super::*;
use crate::candidate_cli::text_batch::TextTask;

#[derive(Args)]
pub(crate) struct TextJobCommand { #[command(subcommand)] operation: TextOperation }
#[derive(Subcommand)]
enum TextOperation {
    /// Create a NEW retained generation/chat job in protected local storage.
    Start(TextJobArgs),
    /// Authenticate original inputs/options and run only uncommitted items.
    Resume(TextResumeArgs),
}
#[derive(Args)]
struct TextResumeArgs {
    #[command(flatten)] common: TextJobArgs,
    /// Repair only uncommitted tails, after authenticating the original contract.
    #[arg(long)] discard_uncommitted_tail: bool,
}
#[derive(Args)]
pub(in crate::candidate_cli) struct TextJobArgs {
    #[arg(long, value_enum, default_value = "generate")] pub task: TextTask,
    /// Fixed per-item generation options. Input is the COMPLETE original NDJSON population.
    #[command(flatten)] pub common: CandidateArgs,
    #[command(flatten)] pub encoding_work: EncodingWorkArgs,
    /// Consent to retain private generated content, tokens and optional scores.
    #[arg(long)] pub store_results: bool,
    #[command(flatten)] pub input_retention: InputRetentionArgs,
    #[arg(long)] pub job_dir: PathBuf,
    #[arg(long, value_parser = parse_job_id)] pub job_id: JobId,
    /// Protected file containing exactly 32 raw job-authentication bytes; not a sampling seed.
    #[arg(long)] pub key_file: PathBuf,
    /// Immutable JobLimits JSON, including ALL five native counters and attempts.
    #[arg(long = "limits")] pub limits_file: PathBuf,
    #[arg(long)] pub materialize: bool,
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}
impl TextJobArgs {
    pub(in crate::candidate_cli) fn task_name(&self) -> &'static str {
        match self.task { TextTask::Generate => "generate", TextTask::Chat => "chat" }
    }
    pub(in crate::candidate_cli) fn validate(&self) -> Result<Limits, Failure> {
        self.input_retention.limits()?;
        self.encoding_work_limits()?;
        let limits = self.common.validate()?;
        let local = |p: &std::path::Path| !p.as_os_str().is_empty() && p.as_os_str() != "-";
        if !self.store_results || !local(&self.job_dir) || !local(&self.key_file) || !local(&self.limits_file)
            || !(1..=1024 * 1024).contains(&self.max_stream_mib)
            || !(1..=1_000_000_000).contains(&self.max_input_lines)
            || self.journal_memory_mib == 0 || self.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_text_job_storage"));
        }
        for mib in [self.journal_memory_mib, self.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|n| n > limits.memory_bytes) {
                return Err(Failure::usage("text_job_memory"));
            }
        }
        // Refuse impossible native occupancy before reading even the limits
        // file. Its population/job buffer quote is added after bounded parsing.
        if self.execution_memory_bytes(limits)? > limits.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "text_job_native_memory" });
        }
        Ok(limits)
    }
    pub(in crate::candidate_cli) fn encoding_work_limits(&self) -> Result<Option<SafeBpeWorkLimits>, Failure> {
        self.encoding_work.limits(&self.common.input_encoding)
    }
    pub(in crate::candidate_cli) fn encoding_workspace_bytes(&self, limits: Limits) -> Result<u64, Failure> {
        match (self.common.input_encoding_policy(limits)?, self.encoding_work_limits()?) {
            (UntrustedEncoding::ByteFallback, None) => Ok(0),
            (UntrustedEncoding::SafeBpeV1(call), Some(_)) => call.requirements()
                .map_err(|_| Failure::usage("text_job_input_encoding"))?.workspace_bytes
                .checked_add(1024).ok_or_else(|| Failure::usage("text_job_memory_arithmetic")),
            _ => Err(Failure::usage("text_job_input_encoding")),
        }
    }
    pub(in crate::candidate_cli) fn execution_memory_bytes(&self, limits: Limits) -> Result<u64, Failure> {
        if limits.compute != self.common.compute.admit_prefill(self.common.policy.prefill()?)? {
            return Err(Failure::usage("text_job_schedule_admission"));
        }
        let native = Int8MemoryRequirement::for_context(self.common.context_tokens)
            .map_err(|_| Failure::usage("text_job_native_quote"))?;
        let prefill = self.common.policy.prefill()?.map(|limits| limits.validate()).transpose()
            .map_err(|_| Failure::usage("text_job_prefill_quote"))?.unwrap_or(0);
        let output = (limits.result_bytes as u64).checked_mul(4)
            .and_then(|n| n.checked_add((self.common.max_new_tokens as u64).checked_mul(8)?))
            .and_then(|n| n.checked_add(4096)).ok_or_else(|| Failure::usage("text_job_memory_arithmetic"))?;
        // Matches Session::load and the owned job host: resident metadata and
        // allocator slack coexist with the native engine, guarded result and
        // CLI preparation. This is a conservative quote, not a RAM permit.
        [limits.preparation_bytes, limits.weight_bytes, 128 * MIB, 64 * MIB,
            native.kv_bytes, native.rope_bytes, native.scratch_payload_bound,
            64 * MIB, SAMPLER_BYTES, prefill, limits.compute.workspace_bytes, output, 65_536,
            self.encoding_workspace_bytes(limits)?]
            .iter().try_fold(0_u64, |total, bytes| total.checked_add(*bytes)
                .ok_or_else(|| Failure::usage("text_job_memory_arithmetic")))
    }
    pub(in crate::candidate_cli) fn check_lifetime(&self, job: JobLimits, limits: Limits) -> Result<(), Failure> {
        job.validate().map_err(|_| Failure::usage("text_job_lifetime"))?;
        if job.max_input_bytes_per_item > self.common.max_input_bytes || job.max_result_bytes < limits.result_bytes
            || job.max_snapshot_bytes > self.max_stream_mib.checked_mul(MIB).ok_or_else(|| Failure::usage("text_job_transport"))? {
            return Err(Failure::usage("text_job_input_or_result_ceiling"));
        }
        let w = job.max_work.model;
        if w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
            || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
            return Err(Failure::usage("text_job_native_work"));
        }
        Ok(())
    }
}
pub(in crate::candidate_cli) fn definition() -> clap::Command {
    TextJobCommand::augment_args(clap::Command::new("text-job")
        .about("Start or resume retained native generation/chat jobs")
        .long_about("Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. Input is the COMPLETE original {id,text,task_args?} NDJSON population, not the text-batch prompt/messages protocol; resume can explicitly restore retained inputs. Optional task_args contains sample_index and chat history only; text becomes the final user turn. Shared generation flags, explicit seed, stops and penalties are frozen in the authenticated recipe. No seed means greedy, never random seed generation. Results commit only after whole native completion; no partial token stream is retained. Failed attempts retain all native work charges. --store-results, protected existing storage, original job key and immutable limits are required. Stdout is completion metadata only. No automatic retry, model download, tools or thinking mode. --compute-threads 2..=64 keeps one persistent INT8 output-row team across every pending item, prompt and generated token. --prefill-rows opts in to bounded layer-major prompt execution and may use that same team. Worker batch storage and coordinator prompt scratch are admitted together before private IO. Either physical width may change on authenticated resume without changing the semantic recipe, original sample addresses, committed records or lifetime work. Completion metadata reports the selected schedule only after every worker joins. Width 1 keeps serial execution. Documents still execute in original order; --cohort-rows/active-rows remain unsupported.")
        .after_help(INPUT_RETENTION_HELP))
        .mut_subcommand("start", encoding_definition)
        .mut_subcommand("resume", encoding_definition)
}
fn encoding_definition(command: clap::Command) -> clap::Command {
    compute::definition(input_encoding::definition(command)).after_long_help(
        "Both generation and chat jobs accept explicit --input-encoding safe-bpe-v1. One per-transcript tokenizer allowance covers the original prompt or every history message plus the final user turn. --max-input-bytes and the native prompt capacity bound transcript bytes and tokens; --max-tokenizer-pair-checks and --max-tokenizer-pair-bytes bound its BPE work. One additional --max-total-tokenizer-* invocation allowance spans every item and failed or cancelled preparation. The policy and all caps are authenticated: repeat them on resume. A resumed invocation receives the same declared tokenizer allowance while persisted native work and attempts remain spent. Compact encoding preserves original bytes and excludes template controls, but changes prompt tokens and makes no identical-output, tokenizer-parity, quality or speed claim. Encoder workspace is admitted alongside the sampler, native buffers and any prompt/worker scratch, and the actual factory remains owned through physical worker join. Compact completion metadata reports the encoding policy and invocation limits. Byte-fallback remains the default and adds no encoding fields. Retention still requires explicit consent.")
}
impl TextJobCommand {
    pub(crate) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        let (_, request) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "text job operation required"))?;
        let common = match &mut command.operation {
            TextOperation::Start(args) => &mut args.common,
            TextOperation::Resume(args) => &mut args.common.common,
        };
        common.compute = compute::Args::from_matches(request)?;
        common.input_encoding = input_encoding::InputEncodingArgs::from_matches(request)?;
        Ok(command)
    }

    fn into_parts(self) -> (RunMode, TextJobArgs) {
        match self.operation { TextOperation::Start(args) => (RunMode::Start, args),
            TextOperation::Resume(args) => (RunMode::Resume { discard_uncommitted: args.discard_uncommitted_tail }, args.common) }
    }
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let (mode, args) = self.into_parts();
        match execute_owned(mode, args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation: mode.name(), status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored status or explicitly resume with original inputs and options; do not assume rollback" };
                if publish(&report, REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
}
fn execute_owned<R: Read + Send + 'static>(mode: RunMode, args: TextJobArgs, input: R, output: &mut impl Write)
    -> Result<(), Failure> {
    args.input_retention.validate(mode, &args.common.input)?;
    let limits = args.validate()?;
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    { crate::candidate_cli::runtime::owned_jobs::generation::execute(mode, args, limits, input, output) }
    #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        let _ = (mode, args, limits, input, output);
        Err(Failure::usage("owned_text_job_profile_unavailable"))
    }
}
#[cfg(test)] pub(in crate::candidate_cli) mod tests;
