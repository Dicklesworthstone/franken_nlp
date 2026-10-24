//! Explicit retained complete-document INT4 redaction over the existing job format.
//! Lifetime work is copied exactly; neither items, retries nor resume replenish it.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    candidate_cli::{batch::IO_BUFFER_BYTES, int4::extract_job::StorageArgs,
        jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobLimits, JobWork, population::PopulationReadLimits},
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

// The gated durable processor independently exports this exact recipe cap.
// Runtime checks equality before private population IO; no storage format is duplicated.
const RECIPE_RESERVE_BYTES: u64 = 4 * (1024 * 1024 - 1024);

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
pub(in crate::candidate_cli::int4) struct JobArgs { document: DocumentArgs, storage: StorageArgs }
#[derive(Clone, Copy)]
struct JobExecutionLimits {
    common: Int4Limits,
    document: DocumentLimits,
    transport: PopulationReadLimits,
    model_work: Int4Work,
    masks: u64,
    buffer_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about("Create or explicitly resume one retained complete-document INT4 redaction job. Requires --store-results, an existing owner-only --job-dir, original --job-id, an independent protected --key-file and exact immutable --limits. Supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation, including resume; inputs remain in memory and are not persisted. IDs are unique across the complete population, flush controls are refused, and task_args cannot override the fixed detector, action, verification or key policy. --max-input-bytes caps decoded ORIGINAL text; JobLimits.max_input_bytes_per_item separately caps the encoded JSON envelope. Whole-document rules and every native NER chunk feed one edited result; fresh transformed detection remains the default. Pseudonymization uses one full256 scope: its separate binary --key-stdin requires a population FILE and its commitment is checked before population IO. The job key file is never used as the pseudonym key. Exact model/group, original envelope bytes/order, detector limits, policy and actual key commitment authenticate before job repair or inference. Lifetime native and mask work comes solely from --limits and is never multiplied or renewed by attempts or resume. Stdout contains completion metadata only; private text and maps remain in the explicitly retained spool. --materialize publishes verified materialized.ndjson only after all items commit. Optional model snapshot creation occurs during setup and remains if later job authentication or execution fails. Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64.")
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population, also on resume; '-' reads stdin unless stdin supplies the pseudonym key"))
            .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded ORIGINAL document bytes per item; the immutable JobLimits cap separately bounds encoded JSON"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; persisted work and attempts never renew")))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain sensitive edited text and optional maps in the owner-only result spool"))
            .mut_arg("key_file", |arg| arg.help("Independent protected job-authentication key: exactly 32 raw bytes; never the pseudonym key"))
    };
    let start = operation("start", "Create a new retained complete-document INT4 redaction job");
    let resume = operation("resume", "Authenticate the original population and redact pending INT4 job items")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new("redact-job").about("Start or resume private complete-document redaction with fixed lifetime budgets")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

impl JobArgs {
    /// Pure option admission. This reads no source, secret, limits or population.
    fn validate(&self, args: &Int4Args) -> Result<DocumentLimits, Failure> {
        let s = &self.storage;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_storage_or_job_options"));
        }
        let limits = self.document.validate(args).map_err(Failure::from)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > limits.common.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok(limits)
    }
    fn lifetime_limits(&self, args: &Int4Args, document: DocumentLimits,
        job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        let mut common = document.common;
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let mut item = document.mapping.max_model_work;
        item.projected_logits -= item.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
        let item = JobWork::from_int4(item, document.mapping.max_mask_visits);
        if !item.fits(job.max_work) {
            return Err(Failure::usage("lifetime_native_or_mask_limits"));
        }
        let model_work = job.max_work.int4_model_work();
        let masks = job.max_work.mask_node_visits;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Exact JobHostLimits quote: complete population/index, encoded input,
        // retained-frame staging, second preparation, IO and database/serializer RAM.
        let buffer_bytes = sum(&[
            product(job.max_snapshot_bytes, 2)?, product(job.max_items, 1024)?,
            product(job.max_input_bytes_per_item as u64, 8)?,
            product(job.max_result_bytes as u64, 4)?, 2 * MIB,
            preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes,
        ]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        let standalone_capture = sum(&[args.max_input_bytes as u64 + 1, common.preparation_bytes])
            .map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        // Keep complete-map/edit workspace, full native KV, caller preparation,
        // secret metadata and detector-sized output guard from DocumentLimits.
        common.minimum_memory_bytes = common.minimum_memory_bytes.checked_sub(standalone_capture)
            .and_then(|n| n.checked_add(buffer_bytes))
            .and_then(|n| n.checked_add(RECIPE_RESERVE_BYTES))
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { common, document, transport, model_work, masks, buffer_bytes,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn configuration(&self, args: &Int4Args, limits: JobExecutionLimits,
        identity: crate::execution_identity::ExecutionIdentity, ner: NerOptions)
        -> crate::jobs::runner::redact::grouped_int4::Int4RedactionJobConfig {
        use crate::{jobs::runner::redact::grouped_int4::Int4RedactionJobConfig,
            tasks::redact::grouped_int4::corpus::Int4DocumentRedactionBatchConfig};
        Int4RedactionJobConfig { max_source_bytes: args.max_input_bytes,
            batch: Int4DocumentRedactionBatchConfig {
                ner_identity: identity, detector: self.document.detector(args, limits.document, ner),
                request: self.document.request(args, limits.document),
                max_model_work: limits.model_work, max_mask_visits: limits.masks } }
    }
}
impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "redact-job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "redact-job operation refused")),
        };
        let job = JobArgs { document: DocumentArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args: job }, Int4Args::from_arg_matches(matches)?))
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(
        self, args: Int4Args, input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let error = ErrorReport { schema_version: 1, operation, task: "redact",
                    redaction_mode: "complete_document", status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "preserve original population and both key scopes; authenticate stored status or explicitly resume; no rollback is implied" };
                if publish(&error, job_cli::REPORT_BYTES, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args,
        input: R, output: &mut impl Write) -> Result<(), Failure> {
        let admitted = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, admitted, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            let _ = (self, args, admitted, input, output);
            Err(Failure::usage("owned_job_profile_unavailable"))
        }
    }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, task: &'static str,
    redaction_mode: &'static str, status: &'static str, code: &'static str,
    exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
