//! Retained complete-source judgments, one exact claim/source pair per attempt.
//! Fixed policy, mapping, native schedule and lifetime authenticate every resume.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use super::super::extract_job::StorageArgs;
use crate::{
    batch::{scored_int4::Int4ScoredBatchLimits,
        judge::grouped_int4::FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { map: MapArgs, storage: StorageArgs, faithfulness: FaithfulnessArgs }

#[derive(Args)]
struct FaithfulnessArgs {
    /// Fixed strict FaithfulnessPolicy JSON; each population record supplies its own claim.
    #[arg(long, value_name = "FILE", required = true)]
    policy: PathBuf,
    /// Explicit per-chunk head ceiling (1..=32); repeat on resume, and omit for legacy jobs.
    #[arg(long, value_name = "N")]
    max_evidence_heads: Option<usize>,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = FaithfulnessArgs::augment_args(StorageArgs::augment_args(super::definition()
            .name(name).about(about)
            .long_about("Retain and resume faithfulness judgments of COMPLETE original sources with a required fixed --policy FaithfulnessPolicy JSON file. By default, supply the COMPLETE original {id,text,task_args:{claim}} NDJSON population on every invocation; explicit retained-input replay is available. text is never a pre-split chunk; every row supplies its own nonempty claim and no policy or budget overrides. The combined exact source and claim bytes must fit the immutable stored per-item input ceiling. Every source chunk and evidence head must finish before the original document commits. Retain full-vocabulary NLI scores, contrary and uncertain evidence, original byte/scalar coordinates and the named complete-source reduction. --cohort-chunks shares native layers within a document; each actual ragged window must fit 64 heads without resizing or fallback. Repeat the exact policy, task budget, mapping and schedule on resume. Requires --store-results, protected existing --job-dir and --key-file, exact --job-id and immutable --limits. All five per-document work flags cover every chunk and evidence head. JobLimits.max_work is the sole lifetime allowance, including failed attempts; mask_node_visits must be zero. Chunks, attempts and resume never renew that allowance. Complete input, planning, native, retained-result and durable-job memory are admitted before population IO. Stdout is metadata only; --materialize publishes private results after every document commits. No partial-document success, automatic retry, generation, selected-row heads, calibrated correctness probability or full-context equivalence. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population with a claim per record; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for combined source and claim bytes; immutable stored input ceiling must fit"))
            .mut_arg("max_map_result_bytes", |arg| arg.help("Complete semantic document cap; stored result ceiling also covers cohort framing when enabled"))
            .mut_arg("cohort_chunks", |arg| arg.help("Share native layers across N chunks; every actual ragged window must fit 64 heads"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable work and attempts never renew"))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-source faithfulness results"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("faithfulness-document-job")
        .about("Start or resume retained complete-source INT4 faithfulness")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-source faithfulness job"))
        .subcommand(operation("resume", "Authenticate the original population and judge pending documents"))
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
    /// Pure admission precedes key, policy, lifetime limits and population IO.
    fn validate(&self, args: &Int4Args) -> Result<MapLimits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !local(&self.faithfulness.policy)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_faithfulness_document_job_storage"));
        }
        let map = self.map.validate_with_max_heads(args, self.faithfulness.max_evidence_heads.unwrap_or(32))?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > map.common.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok(map)
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
        // This capability counts source AND claim. Keep the larger pre-IO
        // memory quote, while the durable recipe binds the exact stored cap.
        map.mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
        if self.result_bytes(map)? > job.max_result_bytes as u64 {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = Int4ScoredBatchLimits { max_item_work: map.mapping.max_model_work,
            max_model_work: job.max_work.int4_model_work() };
        if job.max_work.mask_node_visits != 0 || native.validate().is_err() {
            return Err(Failure::usage("lifetime_faithfulness_document_work"));
        }
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = map.common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        let buffer_bytes = sum(&[self.storage.input_retention.buffer_bytes(job)?,
            preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes])?;
        // Original and replayed descriptors retain each claim-aware chunk's
        // evidence-head identities alongside the complete-source quote.
        let metadata = product(map.mapping.chunks.max_chunks as u64,
            FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES)?;
        map.common.minimum_memory_bytes = sum(&[map.common.minimum_memory_bytes, buffer_bytes, metadata])?;
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
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "faithfulness document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "faithfulness document job operation refused")),
        };
        let args = JobArgs { map: MapArgs::from_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)?,
            faithfulness: FaithfulnessArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
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
        { let _ = (self, args, map, input, output); Err(Failure::usage("owned_faithfulness_document_job_profile_unavailable")) }
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
