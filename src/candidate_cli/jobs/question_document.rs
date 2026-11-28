//! Explicitly retained complete-document INT8 QA with per-record questions.
//! Immutable stored limits own native, mask and attempt lifetime authority.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]
use std::{io::{Read, Write}, path::PathBuf, process::ExitCode};
use clap::{Args, FromArgMatches};
use serde::Serialize;
use crate::{
    batch::source::quantized::question::Int8QuestionDocumentBatchConfig,
    candidate_cli::{MIB, publish, batch::IO_BUFFER_BYTES,
        int4::extract_job::StorageArgs, jobs::{self as job_cli, RunMode},
        question_corpus::{self as question, QuestionDocumentArgs, QuestionDocumentLimits}, stream::Failure},
    error::ErrorCode,
    jobs::{JobLimits, population::PopulationReadLimits, question_document::quantized::{
        Int8QuestionDocumentJobMemoryLimits, Int8QuestionDocumentJobMemoryRequirement}},
    native_engine::strict_int8::prefill::Int8PrefillLimits,
    tasks::answer::AnswerOptions,
    validation::grounded_fields::GroundingBudget,
};

pub(in crate::candidate_cli) struct QuestionDocumentJobCommand {
    pub(in crate::candidate_cli) mode: RunMode,
    pub(in crate::candidate_cli) args: JobArgs,
}
pub(in crate::candidate_cli) struct JobArgs {
    pub(in crate::candidate_cli) input: PathBuf,
    pub(in crate::candidate_cli) document: QuestionDocumentArgs,
    pub(in crate::candidate_cli) storage: StorageArgs,
}
#[derive(Args)]
struct InputArgs {
    /// COMPLETE original NDJSON population; omit only with --use-retained-input.
    #[arg(default_value = "-")]
    input: PathBuf,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = question::document_definition(clap::Command::new(name).about(about))
            .long_about("Retain complete original {id,text,task_args:{question}} NDJSON sources with a distinct private question per record. Every nonblank passage is independently answered, all source ranges are retained, and citations refer to the original document. All passage results must validate before one complete document can commit. Records supply only question; fixed AnswerOptions, document budgets, encoding and optional prompt schedule belong to the authenticated job recipe and must match on resume. Requires --store-results, an existing protected --job-dir, --key-file, --job-id and immutable --limits. JobLimits.max_input_bytes_per_item bounds the entire encoded record, including source, question, escaping, id and framing. Document flags bound each complete source; JobLimits.max_work is the sole lifetime native and mask allowance, including failed attempts. Verification lifetime is conservatively bounded by per-document verification times immutable max_attempts. Restarts never replenish those allowances. Supply the COMPLETE original population on every resume unless explicit retained-input replay is selected. Compact tokenizer totals apply to one original source/question preparation and exact execution reconstruction, including rejected probes; they are not journaled lifetime counters. One persistent projection team composes with --prefill-rows. Worker width may change on resume; prompt schedule and semantic policy stay bound. Complete population, source/question owners, native state, answer frontier and output are admitted before key/model/population IO. Stdout reports metadata only. --materialize publishes private results after all items commit. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative invocation ceiling; durable work and attempts never renew"));
        let command = StorageArgs::augment_args(InputArgs::augment_args(command))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete source/question results"))
            .mut_arg("materialize", |arg| arg.help("Publish verified materialized.ndjson after ALL items commit; never replace it"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("answer-job")
        .about("Start or resume complete-document INT8 QA with per-record questions")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-source question job"))
        .subcommand(operation("resume", "Authenticate the original population and answer pending sources"))
}

pub(in crate::candidate_cli) struct JobExecutionLimits {
    pub(in crate::candidate_cli) document: QuestionDocumentLimits,
    pub(in crate::candidate_cli) config: Int8QuestionDocumentBatchConfig,
    pub(in crate::candidate_cli) transport: PopulationReadLimits,
    pub(in crate::candidate_cli) memory: Int8QuestionDocumentJobMemoryRequirement,
    pub(in crate::candidate_cli) execution_bytes: u64,
    pub(in crate::candidate_cli) minimum_memory_bytes: u64,
    pub(in crate::candidate_cli) io_reserve_bytes: u64,
    pub(in crate::candidate_cli) journal_reserve_bytes: u64,
    pub(in crate::candidate_cli) serialization_reserve_bytes: u64,
}

impl JobArgs {
    /// Validate all path/numeric policy before even bounded limits/options IO.
    pub(in crate::candidate_cli) fn validate(&self) -> Result<QuestionDocumentLimits, Failure> {
        let storage = &self.storage;
        storage.input_retention.limits().map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !storage.store_results || !local(&storage.job_dir) || !local(&storage.key_file)
            || !local(&storage.limits_file) || !(1..=1024 * 1024).contains(&storage.max_stream_mib)
            || !(1..=1_000_000_000).contains(&storage.max_input_lines)
            || storage.journal_memory_mib == 0 || storage.serialization_memory_mib == 0 {
            return Err(Failure::usage("int8_question_document_job_storage"));
        }
        let document = self.document.admit(self.input.clone())
            .map_err(|_| Failure::usage("int8_question_document_job_options"))?;
        for mib in [storage.journal_memory_mib, storage.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > document.shared.memory_bytes) {
                return Err(Failure::usage("int8_question_document_job_memory_options"));
            }
        }
        Ok(document)
    }

    /// The encoded item ceiling and complete decoded source/question ceilings
    /// are independent. The parser/population quote prices the former exactly;
    /// no lifetime allowance is multiplied by population or retry counts.
    pub(in crate::candidate_cli) fn lifetime_limits(&self, document: QuestionDocumentLimits,
        job: JobLimits, options: AnswerOptions) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        let max_verification = verification_lifetime(document.verification, job.max_attempts)?;
        let config = self.document.configuration(&document, options, job.max_work.model,
            job.max_work.mask_node_visits, max_verification)
            .map_err(|_| Failure::usage("int8_question_document_job_lifetime_policy"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        let mut memory = Int8QuestionDocumentJobMemoryLimits {
            document: question::document_memory_limits(&document, io_reserve_bytes), transport,
            journal_reserve_bytes, serialization_reserve_bytes,
        }.requirements(config, job, self.storage.input_retention.input_memory().map_err(storage_failure)?)
            .map_err(|_| Failure::usage("int8_question_document_job_population_limits"))?;
        let prefill_bytes = document.prefill.map(Int8PrefillLimits::validate).transpose()
            .map_err(|_| Failure::usage("int8_question_document_job_prefill"))?.unwrap_or(0);
        memory.document.workspace_bytes = sum(&[memory.document.workspace_bytes, prefill_bytes])?;
        let d = memory.document;
        let execution_bytes = sum(&[d.input_bytes, d.native.kv_bytes, d.workspace_bytes,
            d.document_workspace_bytes, d.output_reservation_bytes, document.shared.compute.workspace_bytes])?;
        let execution_peak = sum(&[execution_bytes, document.shared.weight_bytes, 192 * MIB, 65_536])?;
        let loading_peak = sum(&[document.shared.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
        // The CLI Session retains its original preparation owner while the
        // host independently owns compiler/population execution preparation.
        let minimum_memory_bytes = sum(&[document.shared.preparation_bytes, execution_peak.max(loading_peak)])?;
        if minimum_memory_bytes > document.shared.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
                code: "int8_question_document_job_population_memory", cancellation: None });
        }
        Ok(JobExecutionLimits { document, config, transport, memory, execution_bytes,
            minimum_memory_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}

impl QuestionDocumentJobCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "question document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "question document job operation refused")),
        };
        Ok(Self { mode, args: JobArgs {
            input: InputArgs::from_arg_matches(matches)?.input,
            document: QuestionDocumentArgs::from_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)?,
        } })
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self,
        input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_job_failure(operation, failure, diagnostics),
        }
    }

    pub(in crate::candidate_cli) fn execute_owned<R: Read + Send + 'static>(self,
        input: R, output: &mut impl Write) -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &self.args.input).map_err(storage_failure)?;
        let admitted = self.args.validate()?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { crate::candidate_cli::runtime::owned_jobs::question_document::execute(self.mode, self.args, admitted, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, admitted, input, output);
            Err(Failure::usage("int8_question_document_job_profile_unavailable")) }
    }
}

fn verification_lifetime(per: GroundingBudget, attempts: u64) -> Result<GroundingBudget, Failure> {
    let invalid = || Failure::usage("int8_question_document_job_verification_lifetime");
    let count = usize::try_from(attempts).map_err(|_| invalid())?;
    if count == 0 { return Err(invalid()); }
    Ok(GroundingBudget {
        max_fields: per.max_fields.checked_mul(count).ok_or_else(invalid)?,
        max_matches: per.max_matches.checked_mul(count).ok_or_else(invalid)?,
        max_scan_steps: per.max_scan_steps.checked_mul(attempts).ok_or_else(invalid)?,
    })
}
pub(in crate::candidate_cli) fn storage_failure(error: job_cli::Failure) -> Failure {
    Failure { exit: error.exit, code: error.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("int8_question_document_job_memory_arithmetic"))
}
fn sum(values: &[u64]) -> Result<u64, Failure> {
    values.iter().try_fold(0_u64, |total, value| total.checked_add(*value)
        .ok_or_else(|| Failure::usage("int8_question_document_job_memory_arithmetic")))
}
fn report_job_failure(operation: &'static str, failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    let report = ErrorReport { protocol: "fnlp-candidate-int8-question-document-job-v1", schema_version: 1,
        operation, status: "error", code: failure.code, exit_code: failure.exit,
        cancellation: failure.cancellation, durable_progress_may_exist: true,
        recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
    if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() && failure.cancellation.is_none() {
        return ErrorCode::Generic.as_process_exit();
    }
    failure.exit.as_process_exit()
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}

#[cfg(test)] mod tests;
