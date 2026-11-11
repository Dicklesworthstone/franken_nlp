//! Retained complete-source QA with a distinct authenticated question per item.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]
use super::*;
use crate::{
    candidate_cli::{int4::extract_job::StorageArgs, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits,
        runner::source::grouped_int4::question_document::memory::{
            Int4QuestionDocumentJobMemoryLimits, Int4QuestionDocumentJobMemoryRequirement}},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)] mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs {
    source: source::SourceArgs,
    mapping: MappingArgs,
    policy: question::QuestionPolicyArgs,
    synthesis: question::synthesis::SynthesisArgs,
    questions: QuestionJobArgs,
    storage: StorageArgs,
}
#[derive(Args, Clone, Copy)]
struct QuestionJobArgs {
    /// Exact decoded question cap; every stored record supplies its own question.
    #[arg(long, default_value_t = question::QUESTION_BYTES)]
    max_question_bytes: usize,
    /// Execute N independent discovery chunks together, 1..=64.
    #[arg(long, value_name = "N")]
    cohort_chunks: Option<usize>,
    /// Execute N independent groups per hierarchy level, 1..=64.
    #[arg(long, value_name = "N", requires = "hierarchical_answer")]
    cohort_hierarchy_groups: Option<usize>,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = source::definition(Kind::Answer).name(name).about(about)
            .long_about("Retain complete original {id,text,task_args:{question}} NDJSON sources with one distinct private question per record. By default, supply the exact COMPLETE original population on start and every resume; explicit retained-input replay is available. Each task_args object contains only question; fixed AnswerOptions, discovery, evidence synthesis, optional hierarchy and native schedules belong to the authenticated job recipe. --synthesize-answer uses all collected verified quotes; --hierarchical-answer explicitly permits bounded lossy quote selection when that complete evidence cannot fit one pass. Generated answer prose is never factual evidence. Discovery and every hierarchy pass finish before the whole source/question result commits. --cohort-chunks and --cohort-hierarchy-groups independently select native schedules; repeat them on resume. Requires --store-results, protected existing --job-dir and --key-file, exact --job-id and immutable --limits. JobLimits.max_input_bytes_per_item bounds the entire encoded NDJSON record, including both source and question, JSON escaping, ids and framing. All five work flags and --max-total-mask-node-visits bound one complete document; JobLimits.max_work is the sole lifetime allowance including failures. Both verification lifetimes are derived with checked arithmetic from per-document verification and immutable max_attempts. Persisted native-work, mask and attempt allowances do not renew on restart. Compact tokenizer usage is bounded for each complete preparation and execution operation, not persisted as a lifetime counter. Admit complete population, native phase peak, evidence frontier and output before population IO. Stdout reports metadata; --materialize publishes private results after all items commit. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population with a question per record; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
                .help("Outer ceiling for encoded original records; immutable per-item storage cap must fit"))
            .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
            .mut_arg("options", |arg| arg.help("Complete AnswerOptions JSON fixed across every record and resume"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative invocation allowance; persisted work and attempts never renew"));
        let command = StorageArgs::augment_args(QuestionJobArgs::augment_args(
            question::QuestionPolicyArgs::augment_args(question::synthesis::SynthesisArgs::augment_args(
                MappingArgs::augment_args(command)))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete source/question results"));
        let command = encoding::population_definition(command);
        let details = command.get_after_long_help().map(|help| help.to_string()).unwrap_or_default();
        let command = compute::definition(command)
            .after_long_help(format!("{details}\n\n--compute-threads 2..=64 forms one persistent exact row team for all pending source/question items and every dependent synthesis pass. This physical width may change on authenticated resume while recipes, native-work debits and committed results remain unchanged. Extra worker scratch is admitted before private IO. Chunk and hierarchy-group cohorts cannot combine with a worker team. Serial is the default; no speed or model-parity qualification is implied."));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("answer-document-job")
        .about("Start or resume complete-document QA with per-record questions")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-source question job"))
        .subcommand(operation("resume", "Authenticate the original population and answer pending sources"))
}

#[derive(Clone, Copy)]
struct JobExecutionLimits {
    common: Int4Limits,
    config: Int4QuestionDocumentBatchConfig,
    encoding: Option<encoding::EncodingLimits>,
    transport: PopulationReadLimits,
    memory: Int4QuestionDocumentJobMemoryRequirement,
    reduction_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
    cli_preparation_bytes: u64,
}
impl JobArgs {
    /// Numeric/path checks precede every private configuration read. The
    /// complete population quote follows bounded options and lifetime parsing,
    /// and still precedes model load and opening the original population.
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0
            || !(1..=MAX_INPUT_BYTES).contains(&self.questions.max_question_bytes) {
            return Err(Failure::usage("explicit_question_document_job_storage"));
        }
        let encoding = self.mapping.encoding.limits(args, &self.mapping,
            !self.synthesis.flat.synthesize_answer, true)?;
        let mut storage = args.clone();
        storage.input_encoding = input_encoding::InputEncodingArgs::default();
        let common = if args.compute.threads() != 1 {
            if args.prefill()?.is_some() || self.questions.cohort_chunks.is_some()
                || self.questions.cohort_hierarchy_groups.is_some() {
                return Err(CandidateError::Arguments.into());
            }
            self.source.validate_batch_with_prefill(&storage)?
        } else { self.source.validate(&storage)? };
        self.mapping.validate(args, &self.source, common, self.questions.cohort_chunks)?;
        self.synthesis.validate()?;
        self.policy.verification()?;
        let hierarchy = self.synthesis.hierarchy()?;
        if let Some(max_groups) = self.questions.cohort_hierarchy_groups {
            let hierarchy = hierarchy.ok_or(CandidateError::Arguments)?;
            Int4HierarchyCohortSchedule { max_groups, discovery_cohorts: self.questions.cohort_chunks.is_some() }
                .validate(hierarchy.max_passes).map_err(|_| CandidateError::Arguments)?;
        }
        let budget = self.source.budget(args, common);
        let passes = hierarchy.map_or(u64::from(self.synthesis.flat.synthesize_answer), |h| h.max_passes as u64);
        let per_chunk = sum(&[args.context_tokens as u64 * 32,
            u64::from(budget.max_grammar_states) * 512, shared::OPTIONS_BYTES as u64 * 2])?;
        let preparation_floor = sum(&[times(per_chunk, self.mapping.max_chunks as u64 + passes)?,
            256 * MIB, args.max_input_bytes as u64 * 64,
            (self.mapping.max_map_result_bytes as u64 + 4096) * 2,
            encoding.map(encoding::EncodingLimits::workspace_bytes).transpose()?.unwrap_or(0)])?;
        if common.preparation_bytes < preparation_floor {
            return Err(Failure::usage("question_document_preparation"));
        }
        for mib in [s.journal_memory_mib, s.serialization_memory_mib, self.mapping.reduction_reserve_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes == 0 || bytes > common.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok(common)
    }

    fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits,
        options: AnswerOptions, job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        let encoding = self.mapping.encoding.limits(args, &self.mapping,
            !self.synthesis.flat.synthesize_answer, true)?;
        if job.max_input_bytes_per_item > args.max_input_bytes
            || self.questions.max_question_bytes > job.max_input_bytes_per_item {
            return Err(Failure::usage("stored_question_record_ceiling"));
        }
        let mut mapping = self.mapping.limits(args, &self.source, self.questions.cohort_chunks)?;
        mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
        let mut planning = self.source.planning(args);
        planning.max_input_bytes = planning.max_input_bytes.max(self.questions.max_question_bytes);
        let planning = self.synthesis.planning(planning);
        let verification = self.policy.verification()?;
        let config = Int4QuestionDocumentBatchConfig { options, budget: self.source.budget(args, common),
            planning, mapping, verification, synthesis: self.synthesis.limits(planning)?,
            hierarchy: self.synthesis.hierarchy()?, max_question_bytes: self.questions.max_question_bytes,
            max_model_work: job.max_work.int4_model_work(), max_mask_visits: job.max_work.mask_node_visits,
            max_map_verification: verification_lifetime(mapping.verification, job.max_attempts)?,
            max_verification: verification_lifetime(verification, job.max_attempts)? };
        let transport = PopulationReadLimits { max_stream_bytes: times(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        let io_reserve_bytes = 2 * IO_BUFFER_BYTES as u64 + job_cli::REPORT_BYTES as u64;
        let journal_reserve_bytes = times(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = times(self.storage.serialization_memory_mib, MIB)?;
        let reduction_reserve_bytes = mib(self.mapping.reduction_reserve_mib)?;
        let memory = Int4QuestionDocumentJobMemoryLimits {
            document: Int4QuestionDocumentMemoryLimits { context_tokens: args.context_tokens,
                allocator_reserve_bytes: ALLOCATOR_BYTES, preparation_reserve_bytes: common.preparation_bytes,
                io_reserve_bytes, reduction_reserve_bytes, cohort_chunks: self.questions.cohort_chunks,
                hierarchy_groups: self.questions.cohort_hierarchy_groups },
            transport, journal_reserve_bytes, serialization_reserve_bytes,
        }.requirements(args.group_width, config, job, self.storage.input_retention.input_memory()?)
            .map_err(|_| Failure::usage("question_document_job_admission"))?;
        let q = memory.document;
        let execution = sum(&[q.input_bytes, q.native.kv_bytes, q.workspace_bytes,
            q.document_workspace_bytes, q.output_reservation_bytes, common.compute.workspace_bytes])?;
        common.minimum_memory_bytes = sum(&[grouped_memory_floor(args, common, execution)?, io_reserve_bytes])?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "question_document_job_memory" });
        }
        let cli_preparation_bytes = sum(&[common.preparation_bytes, io_reserve_bytes])?;
        Ok(JobExecutionLimits { common, config, encoding, transport, memory, reduction_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes, cli_preparation_bytes })
    }
}
fn verification_lifetime(per: GroundingBudget, attempts: u64) -> Result<GroundingBudget, Failure> {
    let count = usize::try_from(attempts).map_err(|_| Failure::usage("question_verification_lifetime"))?;
    let invalid = || Failure::usage("question_verification_lifetime");
    if count == 0 { return Err(invalid()); }
    Ok(GroundingBudget { max_fields: per.max_fields.checked_mul(count).ok_or_else(invalid)?,
        max_matches: per.max_matches.checked_mul(count).ok_or_else(invalid)?,
        max_scan_steps: per.max_scan_steps.checked_mul(attempts).ok_or_else(invalid)? })
}

impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(|| clap::Error::raw(
            clap::error::ErrorKind::MissingSubcommand, "question document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "question document job operation refused")),
        };
        let args = JobArgs { source: source::SourceArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?, policy: question::QuestionPolicyArgs::from_arg_matches(matches)?,
            synthesis: question::synthesis::SynthesisArgs::from_arg_matches(matches)?,
            questions: QuestionJobArgs::from_arg_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)? };
        let mut common = Int4Args::from_arg_matches(matches)?;
        common.input_encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        common.compute = compute::Args::from_matches(matches)?;
        Ok((Self { mode, args }, common))
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
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &args.input)?;
        let admitted = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, admitted, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, admitted, input, output); Err(Failure::usage("owned_question_document_job_profile_unavailable")) }
    }
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
