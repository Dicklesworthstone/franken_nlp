//! Retained whole-document schema extraction over the native INT4 map pipeline.
//! The complete original document, including every chunk, is one durable item.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}},
    jobs::{JobLimits, JobWork, population::PopulationReadLimits},
    validation::grounded_fields::GroundingBudget,
};
use super::{extract_job::StorageArgs, extract_map::{MapArgs, MapLimits, object::ReconciliationArgs}};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand { mode: RunMode, args: JobArgs }
pub(super) struct JobArgs {
    map: MapArgs, storage: StorageArgs, lifetime: LifetimeArgs, reconciliation: ReconciliationArgs,
}

#[derive(Args, Clone, Copy)]
struct LifetimeArgs {
    /// Immutable verification-field ceiling across ALL durable attempts.
    #[arg(long, default_value_t = 4_096_000)]
    max_lifetime_extraction_fields: usize,
    /// Immutable original-source evidence-span ceiling across ALL attempts.
    #[arg(long, default_value_t = 16_384_000)]
    max_lifetime_extraction_evidence_spans: usize,
    /// Immutable independent verification scan ceiling, including failures.
    /// The per-document ceiling times --limits.max_attempts must fit.
    #[arg(long, default_value_t = 67_108_864_000)]
    max_lifetime_extraction_scan_steps: u64,
}
impl LifetimeArgs {
    fn verification(self) -> GroundingBudget {
        GroundingBudget { max_fields: self.max_lifetime_extraction_fields,
            max_matches: self.max_lifetime_extraction_evidence_spans,
            max_scan_steps: self.max_lifetime_extraction_scan_steps }
    }
    fn check(self, per_attempt: GroundingBudget, attempts: u64) -> Result<(), Failure> {
        let fits = |amount: u64, cap: u64| amount.checked_mul(attempts).is_some_and(|total| total <= cap);
        let fields = u64::try_from(per_attempt.max_fields)
            .map_err(|_| Failure::usage("lifetime_extraction_verification"))?;
        let matches = u64::try_from(per_attempt.max_matches)
            .map_err(|_| Failure::usage("lifetime_extraction_verification"))?;
        let field_cap = u64::try_from(self.max_lifetime_extraction_fields)
            .map_err(|_| Failure::usage("lifetime_extraction_verification"))?;
        let match_cap = u64::try_from(self.max_lifetime_extraction_evidence_spans)
            .map_err(|_| Failure::usage("lifetime_extraction_verification"))?;
        if attempts == 0 || !fits(fields, field_cap) || !fits(matches, match_cap)
            || !fits(per_attempt.max_scan_steps, self.max_lifetime_extraction_scan_steps) {
            return Err(Failure::usage("lifetime_extraction_verification"));
        }
        Ok(())
    }
}

pub(super) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = LifetimeArgs::augment_args(StorageArgs::augment_args(ReconciliationArgs::augment_definition(extract_map::document_job_definition())
            .name(name).about(about)
            .long_about("Retain and resume COMPLETE original documents using one fixed exact schema and the native INT4 extraction map pipeline. By default, supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation; explicit retained-input replay is available. text is the original document, never a pre-split chunk. Task arguments may only be absent, null or an empty object. Every chunk must finish and verify before its original document commits. The default retains independent exact chunk JSON values and original-document provenance, with required schema fields satisfiable in every chunk. Optional --reconcile-root-object derives a partial root-object schema for each chunk, preserves nested constraints, and combines complementary atomic properties into one independently validated original-schema object. Equal values retain every origin; conflicting values or missing final required properties fail the entire item. With --source-membership, the final object is additionally checked against the complete original source using the SAME remaining field/span/scan allowance. The object policy and every --max-object-* bound are authenticated; repeat them on resume. It cannot combine with --root-array-union. Optional --cohort-chunks N shares native layers across independent chunks while retaining each chunk's exact schema, grammar, source language and causal context. The declared schedule and width are authenticated: repeat the same flag on resume, and omit it for existing serial jobs. Optional --max-selected-rows N fixes one complete-legal-row ceiling for every chunk, including accepting EOS, and refuses excess without pruning or fallback. The strategy and exact row cap are authenticated: repeat the same flag on resume; omit it for existing full-vocabulary jobs. Complete native, decoder, schema, evidence, semantic-result and durable-job memory are admitted together. Source membership and exact reconciliation establish structural and lexical properties, not semantic accuracy or extraction completeness. Requires --store-results, protected existing --job-dir and --key-file, exact --job-id and immutable --limits. The original schema bytes, chunking, all five native work axes, masks and explicit lifetime verification ceilings are authenticated and never renewed on resume; failed attempts remain debited. Stdout contains metadata only, including the declared schedule and reduction policy when enabled. --materialize publishes retained results after every original document commits. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for a complete NDJSON record and decoded original document; immutable stored input ceiling must fit"))
            .mut_arg("max_map_result_bytes", |arg| arg.help("Complete original-document extraction result cap; must fit immutable JobLimits.max_result_bytes"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable work and attempts never renew"))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-document extraction results in the protected spool"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("extract-document-job")
        .about("Start or resume retained whole-document exact-schema INT4 extraction")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained whole-document schema extraction job"))
        .subcommand(operation("resume", "Authenticate the original population and run pending complete documents"))
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
    fn validate(&self, args: &Int4Args) -> Result<MapLimits, Failure> {
        let s = &self.storage;
        s.input_retention.limits()?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0
            || self.lifetime.max_lifetime_extraction_fields == 0
            || self.lifetime.max_lifetime_extraction_evidence_spans == 0
            || self.lifetime.max_lifetime_extraction_scan_steps == 0 {
            return Err(Failure::usage("explicit_extraction_document_job_storage"));
        }
        let mut map = self.map.validate_document_job(args)?;
        if self.reconciliation.limits(self.map.mapping)?.is_some()
            && self.map.encoding.policy()? != crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback {
            return Err(CandidateError::Arguments.into());
        }
        // The shared original/partial declaration and exact reducer quote is
        // additional to retained native maps, owned population and output.
        let object = self.reconciliation.workspace_bytes(&self.map, map)?;
        map.common.minimum_memory_bytes = sum(&[map.common.minimum_memory_bytes, object])?;
        map.workspace_bytes = sum(&[map.workspace_bytes, object])?;
        if map.common.minimum_memory_bytes > map.common.memory_bytes {
            return Err(CandidateError::Arguments.into());
        }
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > map.common.memory_bytes) {
                return Err(Failure::usage("extraction_document_job_memory"));
            }
        }
        Ok(map)
    }
    fn lifetime_limits(&self, args: &Int4Args, mut map: MapLimits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if map.max_result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_document_result_ceiling"));
        }
        let work = JobWork::from_int4(map.mapping.mapping.max_model_work, map.mapping.mapping.max_mask_visits);
        if !work.fits(job.max_work) { return Err(Failure::usage("lifetime_document_work_limits")); }
        self.lifetime.check(map.mapping.verification, job.max_attempts)?;
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = map.common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // The shared map quote retains every chunk program and the complete
        // independently verified output. Add the host's selected input owner,
        // journal and serialization owners without multiplying lifetime work.
        let buffer_bytes = sum(&[
            self.storage.input_retention.buffer_bytes(job)?, preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes,
        ]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        map.common.minimum_memory_bytes = map.common.minimum_memory_bytes.checked_add(buffer_bytes)
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if map.common.minimum_memory_bytes > map.common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { map, transport, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}
impl JobCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<(Self, Int4Args), clap::Error> {
        let invalid = |message| clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, message);
        let (name, matches) = matches.subcommand()
            .ok_or_else(|| invalid("extract-document-job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(invalid("extract-document-job operation refused")),
        };
        let args = JobArgs { map: MapArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)?, lifetime: LifetimeArgs::from_arg_matches(matches)?,
            reconciliation: ReconciliationArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, task: "extract", status: "error",
                    code: failure.code, exit_code: failure.exit, durable_progress_may_exist: true,
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
        { let _ = (self, args, map, input, output); Err(Failure::usage("owned_extraction_document_job_profile_unavailable")) }
    }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, task: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
