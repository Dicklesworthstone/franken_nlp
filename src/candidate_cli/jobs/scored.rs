//! Explicit retained finite-score jobs; separate from the live score-batch pipe.
use super::*;
use crate::candidate_cli::{scored::ScoredArgs, scored_batch::{self, ScoreTask}};
pub(in crate::candidate_cli) mod document;

#[derive(Args)]
pub(crate) struct ScoreJobCommand {
    #[command(subcommand)] operation: ScoreOperation,
}
#[derive(Subcommand)]
enum ScoreOperation {
    /// Start a NEW native classification/sentiment/judgment job; retain private results.
    Start(ScoreJobArgs),
    /// Authenticate the COMPLETE original population and run only pending items.
    Resume(ScoreResumeArgs),
}
#[derive(Args)]
struct ScoreResumeArgs {
    #[command(flatten)] common: ScoreJobArgs,
    /// Repair uncommitted tails/stages only AFTER original-contract authentication.
    #[arg(long)] discard_uncommitted_tail: bool,
}
#[derive(Args)]
pub(in crate::candidate_cli) struct ScoreJobArgs {
    #[arg(long, value_parser = ["classify", "sentiment", "judge"])]
    pub task: String,
    #[command(flatten)] pub host: ScoredArgs,
    #[command(flatten)] pub prefill: crate::candidate_cli::scored::prefill::ScoringPrefillArgs,
    #[arg(skip)] pub compute: compute::Args,
    #[arg(skip)] pub encoding: input_encoding::InputEncodingArgs,
    #[arg(skip)] pub judge_encoding: crate::candidate_cli::judge::EncodingArgs,
    #[arg(skip)] pub document_encoding: input_encoding::DocumentInputEncodingArgs,
    #[command(flatten)] pub encoding_work: EncodingWorkArgs,
    /// Required consent to retain private native output in the protected spool.
    #[arg(long)] pub store_results: bool,
    #[command(flatten)] pub input_retention: InputRetentionArgs,
    /// Existing owner-only directory. Never created or adopted automatically.
    #[arg(long)] pub job_dir: PathBuf,
    /// Expected random 128-bit job ID: exactly 32 lowercase hex characters.
    #[arg(long, value_parser = parse_job_id)] pub job_id: JobId,
    /// Protected regular file containing exactly 32 RAW secret bytes, not hex.
    #[arg(long)] pub key_file: PathBuf,
    /// Immutable JobLimits JSON, including lifetime work and attempt ceilings.
    #[arg(long = "limits")] pub limits_file: PathBuf,
    /// Same settings JSON as score-batch; no document, budget or identity fields.
    #[arg(long)] pub defaults: Option<PathBuf>,
    /// Classify/sentiment only: immutable complete-document chunk/aggregate settings.
    /// Each original document commits once, after every chunk completes.
    #[arg(long, value_name = "FILE")]
    pub document_config: Option<PathBuf>,
    /// Publish verified materialized.ndjson only after all items commit.
    #[arg(long)] pub materialize: bool,
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    /// Includes blank lines. Flush records are forbidden in durable populations.
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}
impl ScoreJobArgs {
    pub(in crate::candidate_cli) fn kind(&self) -> Result<ScoreTask, Failure> {
        ScoreTask::named(&self.task).ok_or_else(|| Failure::usage("score_job_task"))
    }
    pub(in crate::candidate_cli) fn prefill_limits(&self)
        -> Result<Option<crate::native_engine::strict_int8::prefill::Int8PrefillLimits>, Failure> {
        scored_batch::corpus_prefill(self.kind()?, &self.prefill).map_err(Into::into)
    }
    pub(in crate::candidate_cli) fn validate(&self) -> Result<(CandidateArgs, Limits), Failure> {
        document::validate_mode(self)?;
        self.input_retention.limits()?;
        self.encoding_work_limits()?;
        self.encoding_policy()?;
        let prefill = self.prefill_limits()?;
        let admitted = self.compute.admit_prefill(prefill)?;
        let local = |p: &std::path::Path| !p.as_os_str().is_empty() && p.as_os_str() != "-";
        if !self.store_results || !local(&self.job_dir) || !local(&self.key_file) || !local(&self.limits_file)
            || self.defaults.as_ref().is_some_and(|p| !local(p))
            || !(1..=1024 * 1024).contains(&self.max_stream_mib)
            || !(1..=1_000_000_000).contains(&self.max_input_lines)
            || self.journal_memory_mib == 0 || self.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_scored_job_storage"));
        }
        let (mut common, mut limits) = if self.document_config.is_some() {
            self.host.common()?
        } else {
            self.host.common_with_encoding(&self.encoding)?
        };
        common.input_encoding = self.encoding.clone();
        common.compute = self.compute;
        limits.compute = admitted;
        for mib in [self.journal_memory_mib, self.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > limits.memory_bytes) {
                return Err(Failure::usage("score_job_memory"));
            }
        }
        // Refuse impossible native occupancy before even the bounded limits
        // file. The complete original/retained population quote follows parsing.
        if self.execution_memory_bytes(limits)? > limits.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "score_job_native_memory" });
        }
        Ok((common, limits))
    }
    pub(in crate::candidate_cli) fn encoding_policy(&self) -> Result<UntrustedEncoding, Failure> {
        if self.document_config.is_some() {
            document::validate_mode(self)?;
            return self.document_encoding.policy().map_err(Into::into);
        }
        match self.kind()? {
            ScoreTask::Judge => self.judge_encoding.policy(&self.encoding).map_err(Into::into),
            ScoreTask::Classify | ScoreTask::Sentiment => {
                if !self.judge_encoding.is_default() {
                    return Err(Failure::usage("score_job_judge_encoding_limits"));
                }
                self.host.encoding_policy(&self.encoding).map_err(Into::into)
            }
        }
    }
    pub(in crate::candidate_cli) fn encoding_work_limits(&self) -> Result<Option<SafeBpeWorkLimits>, Failure> {
        self.encoding_work.limits(&self.encoding)
    }
    pub(in crate::candidate_cli) fn encoding_workspace_bytes(&self) -> Result<u64, Failure> {
        match (self.encoding_policy()?, self.encoding_work_limits()?) {
            (UntrustedEncoding::ByteFallback, None) => Ok(0),
            (UntrustedEncoding::SafeBpeV1(call), Some(_)) => call.requirements()
                .map_err(|_| Failure::usage("score_job_input_encoding"))?.workspace_bytes
                .checked_add(1024).ok_or_else(|| Failure::usage("score_job_memory_arithmetic")),
            _ => Err(Failure::usage("score_job_input_encoding")),
        }
    }
    pub(in crate::candidate_cli) fn execution_memory_bytes(&self, limits: Limits) -> Result<u64, Failure> {
        let prefill = self.prefill_limits()?;
        if limits.compute != self.compute.admit_prefill(prefill)? {
            return Err(Failure::usage("score_job_schedule_admission"));
        }
        let native = Int8MemoryRequirement::for_context(self.host.context_tokens)
            .map_err(|_| Failure::usage("score_job_native_quote"))?;
        let extra = prefill.map(|limits| limits.validate()).transpose()
            .map_err(|_| Failure::usage("score_job_prefill_quote"))?.unwrap_or(0);
        // Each result remains guarded through spool sync/journal commit. The
        // complete invocation work ceiling conservatively bounds its tokens.
        let output_tokens = self.host.work_ceiling().projected_logits
            / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64;
        let output = (self.host.max_result_bytes as u64).checked_mul(4)
            .and_then(|bytes| bytes.checked_add(output_tokens.checked_mul(8)?))
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| Failure::usage("score_job_memory_arithmetic"))?;
        // Model metadata/slack, caller preparation, coordinator and worker
        // scratch all coexist. Host-owned population/journal buffers are added
        // separately after the immutable lifetime limits are read.
        [limits.preparation_bytes, limits.weight_bytes, 128 * MIB, 64 * MIB,
            native.kv_bytes, native.rope_bytes, native.scratch_payload_bound,
            64 * MIB, extra, limits.compute.workspace_bytes, output, 65_536,
            self.encoding_workspace_bytes()?]
            .into_iter().try_fold(0_u64, |total, bytes| total.checked_add(bytes)
                .ok_or_else(|| Failure::usage("score_job_memory_arithmetic")))
    }
    pub(in crate::candidate_cli) fn parse_defaults(&self, json: Option<&str>, budget: crate::tasks::ir::TaskBudget)
        -> Result<scored_batch::Defaults, Failure> {
        scored_batch::parse_defaults(self.kind()?, &self.host, json, budget).map_err(Into::into)
    }
    pub(in crate::candidate_cli) fn task_budget(&self, limits: Limits)
        -> Result<crate::tasks::ir::TaskBudget, Failure> {
        let mut budget = self.host.budget(limits);
        if self.document_config.is_some() && matches!(self.kind()?, ScoreTask::Sentiment) {
            // Every affect head belongs to one semantic task. The physical
            // document runner still reuses one native cache between heads.
            budget.max_kv_bytes = budget.max_kv_bytes.checked_mul(4)
                .ok_or_else(|| Failure::usage("score_document_job_budget"))?;
        }
        Ok(budget)
    }
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    ScoreJobCommand::augment_args(clap::Command::new("score-job")
        .about("Start or resume retained native classification/sentiment/judgment corpus jobs")
        .long_about("Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64. The input is the COMPLETE original NDJSON {id,text,task_args?} population; resume can explicitly restore it from retained input storage. Uses the same full-vocabulary scorers and defaults as score-batch; no generated-label shortcut or calibrated-confidence claim. Task settings and defaults are frozen for resume. All three tasks accept --prefill-rows for bounded prompts and --compute-threads for one persistent INT8 row team across every pending record and scoring head. Both workspace owners are admitted together. Worker width may change on authenticated resume; classification/sentiment prompt width may also change. Judgment preserves its existing frozen prompt-schedule recipe and rejects a changed prefill schedule on resume. Lifetime work and failed attempts remain debited. Explicit --store-results, protected directory, original secret and immutable JobLimits are mandatory. Stdout is completion metadata only, emitted after all workers physically join. No automatic retry.")
        .after_help(INPUT_RETENTION_HELP))
        .mut_subcommand("start", encoding_definition)
        .mut_subcommand("resume", encoding_definition)
}
fn encoding_definition(command: clap::Command) -> clap::Command {
    compute::definition(crate::candidate_cli::judge::encoding_definition(command))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Cumulative BPE pair checks per complete scoring record; safe-bpe-v1 only"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Cumulative examined BPE pair bytes per complete scoring record; safe-bpe-v1 only"))
        .mut_arg("max_tokenizer_input_bytes", |arg| arg.help("Cumulative encoded bytes for judgment or --document-config preparation, including all chunk probes; safe-bpe-v1 only"))
        .mut_arg("max_tokenizer_output_tokens", |arg| arg.help("Cumulative untrusted tokens for judgment or --document-config preparation; independent of native context, safe-bpe-v1 only"))
        .after_long_help(
        "Classification, sentiment and judgment jobs accept explicit --input-encoding safe-bpe-v1. One per-record allowance covers every untrusted segment, including both pairwise orders, all rubric criteria and every faithfulness evidence head. Judgment and complete-document classification/sentiment additionally accept --max-tokenizer-input-bytes and --max-tokenizer-output-tokens, independent of wire size and native head context. One additional --max-total-tokenizer-* invocation ledger covers all records and failed or cancelled preparation. Both policies and all caps are authenticated: repeat them on resume. Each resumed invocation receives the same bounded tokenizer allowance while persisted native work and attempts remain spent. Compact input preserves original bytes and excludes template controls, but changes prompt tokens and makes no identical-output, tokenizer-parity, quality or measured-speed claim. Encoder workspace is admitted before private IO and held through physical completion. Byte-fallback remains the default. Judgment keeps its existing frozen prefill schedule on resume.\n\nFor --task classify or sentiment, add --document-config FILE to process complete original documents larger than one context. The strict version-1 JSON sets max_chunks, max_chunk_bytes, max_tokenizer_calls, max_retained_result_bytes, max_result_bytes and minimum_coverage_ppm. One original document is one durable item: every native chunk must complete before its full map and source-weighted aggregate commit. --max-result-bytes caps each chunk; the JSON max_result_bytes caps the complete result and must fit JobLimits. All five native work ceilings span the invocation and must fit immutable lifetime work; more chunks or retries never multiply them. Chunking, coverage, output bounds, defaults and tokenizer policies are authenticated and must match on resume. Worker and prefill widths may change without changing task semantics. Per-record task_args may replace fixed defaults and only tighten task budgets. Compact document encoding shares one allowance across every sizing probe and chunk/head reconstruction. No partial successful document is committed. Stdout remains metadata only; complete_document identifies this mode.")
}
impl ScoreJobCommand {
    pub(crate) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        let (_, request) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "score job operation required"))?;
        let args = match &mut command.operation {
            ScoreOperation::Start(args) => args,
            ScoreOperation::Resume(args) => &mut args.common,
        };
        args.encoding = input_encoding::InputEncodingArgs::from_matches(request)?;
        args.judge_encoding = crate::candidate_cli::judge::EncodingArgs::from_arg_matches(request)?;
        if args.document_config.is_some() {
            args.document_encoding = input_encoding::DocumentInputEncodingArgs::from_matches(request)?;
        }
        args.compute = compute::Args::from_matches(request)?;
        Ok(command)
    }
    fn into_parts(self) -> (RunMode, ScoreJobArgs) {
        match self.operation { ScoreOperation::Start(args) => (RunMode::Start, args),
            ScoreOperation::Resume(args) => (RunMode::Resume { discard_uncommitted: args.discard_uncommitted_tail }, args.common) }
    }
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let (mode, args) = self.into_parts();
        match execute_owned(mode, args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation: mode.name(), status: "error",
                    code: failure.code, exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
}
fn execute_owned<R: Read + Send + 'static>(mode: RunMode, args: ScoreJobArgs, input: R, output: &mut impl Write)
    -> Result<(), Failure> {
    args.input_retention.validate(mode, &args.host.input)?;
    let (common, limits) = args.validate()?;
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    { crate::candidate_cli::runtime::owned_jobs::scored::execute(mode, args, common, limits, input, output) }
    #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        let _ = (mode, args, common, limits, input, output);
        // BEFORE key, defaults, population, model or runtime IO.
        Err(Failure::usage("owned_score_job_profile_unavailable"))
    }
}

#[cfg(test)] pub(in crate::candidate_cli) mod tests;
