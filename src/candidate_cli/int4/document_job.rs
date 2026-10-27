//! Retained whole-document INT4 jobs over the existing native source pipelines.
//! Every original document is one durable item, including all reduction passes.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode}, source::Kind},
    jobs::{JobLimits, JobWork, population::PopulationReadLimits},
    validation::grounded_fields::GroundingBudget,
};
use super::{extract_job::StorageArgs, source_map::{MapArgs, MapLimits}};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand { mode: RunMode, args: JobArgs }
pub(super) struct JobArgs {
    map: MapArgs,
    map_only: bool,
    storage: StorageArgs,
    lifetime: LifetimeArgs,
    keyphrase_lifetime: Option<KeyphraseLifetimeArgs>,
}
#[derive(Args, Clone, Copy)]
struct KeyphraseLifetimeArgs {
    /// Immutable independent keyphrase-union scan work across ALL attempts.
    /// Default 536870912000; only applies with --reduce-keyphrases.
    #[arg(long, requires = "reduce_keyphrases")]
    max_lifetime_keyphrase_scan_work: Option<u64>,
}
#[derive(Args, Clone, Copy)]
struct LifetimeArgs {
    /// Immutable total verification-field ceiling across ALL durable attempts.
    #[arg(long, default_value_t = 4_096_000)]
    max_lifetime_source_fields: usize,
    /// Immutable total original-source occurrence ceiling across ALL attempts.
    #[arg(long, default_value_t = 16_384_000)]
    max_lifetime_source_matches: usize,
    /// Immutable source-verification scan ceiling, including failed attempts.
    #[arg(long, default_value_t = 67_108_864_000)]
    max_lifetime_source_scan_steps: u64,
    /// Separate exact-summary reducer scan ceiling across ALL attempts.
    /// The selected per-document ceiling times --limits.max_attempts must fit.
    #[arg(long, default_value_t = 536_870_912_000)]
    max_lifetime_summary_scan_steps: u64,
}
impl LifetimeArgs {
    fn verification(self) -> GroundingBudget {
        GroundingBudget { max_fields: self.max_lifetime_source_fields,
            max_matches: self.max_lifetime_source_matches, max_scan_steps: self.max_lifetime_source_scan_steps }
    }
    fn check(self, per_attempt: GroundingBudget, summary_scan_steps: u64, attempts: u64)
        -> Result<(), Failure> {
        let fits = |amount: u64, cap: u64| amount.checked_mul(attempts).is_some_and(|total| total <= cap);
        let fields = u64::try_from(per_attempt.max_fields).map_err(|_| Failure::usage("lifetime_document_verification"))?;
        let matches = u64::try_from(per_attempt.max_matches).map_err(|_| Failure::usage("lifetime_document_verification"))?;
        let field_cap = u64::try_from(self.max_lifetime_source_fields).map_err(|_| Failure::usage("lifetime_document_verification"))?;
        let match_cap = u64::try_from(self.max_lifetime_source_matches).map_err(|_| Failure::usage("lifetime_document_verification"))?;
        if attempts == 0 || !fits(fields, field_cap) || !fits(matches, match_cap)
            || !fits(per_attempt.max_scan_steps, self.max_lifetime_source_scan_steps)
            || !fits(summary_scan_steps, self.max_lifetime_summary_scan_steps) {
            return Err(Failure::usage("lifetime_document_verification"));
        }
        Ok(())
    }
}

pub(super) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let tasks = [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer]
            .into_iter().zip(source_map::definitions()).map(|(kind, command)| {
                let command = LifetimeArgs::augment_args(StorageArgs::augment_args(command.name(kind.name())
                    .long_about("Retain and resume COMPLETE original documents with one fixed INT4 source task. Supply the COMPLETE original {id,text,task_args?} NDJSON population on every invocation, including resume. text is the original source document; answer uses one separate exact --question file for the entire job. Task arguments may only be absent, null or an empty object. Each document commits atomically after every map and selected reduction pass; individual chunks are not checkpoints. NER/keyphrases default to independent chunk results; keyphrases --reduce-keyphrases preserves the complete exact candidate union and applies final global ranking once. Ranked phrases retain original-coordinate evidence without native token transcripts; their independent lifetime scan-work cap is authenticated. Summarize defaults to deterministic cited-bullet selection, and answer defaults to independent question-aware chunk answers. Explicit synthesis/hierarchy flags use the same native pipelines as the single-document map commands. Requires --store-results, protected existing --job-dir and --key-file, exact --job-id, and immutable --limits. Five native work axes, masks, independent verification, original question/options and all reduction/hierarchy limits are authenticated and never renewed on resume. Every allowed synthesis pass is reserved before discovery; failed attempts remain debited. Stdout contains metadata only. --materialize publishes verified retained results after every original document commits. Hierarchies are explicitly lossy, structural citations do not establish semantic support, and inference failure is never successful abstention. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
                    .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population, also on resume; '-' reads stdin"))
                    .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for a complete NDJSON record and decoded source; immutable stored input ceiling must fit"))
                    .mut_arg("max_map_result_bytes", |arg| arg.help("Raw map/reduction frontier cap; selected complete-document output must fit immutable JobLimits.max_result_bytes"))
                    .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for this invocation; durable work and attempts never renew"))))
                    .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-document results in the protected spool"));
                let command = if kind == Kind::Keyphrases { KeyphraseLifetimeArgs::augment_args(command) }
                    else { command };
                let command = if kind == Kind::Summarize {
                    command.arg(clap::Arg::new("map_only").long("map-only").action(clap::ArgAction::SetTrue)
                        .conflicts_with_all(["synthesize_summary", "max_summary_bullets", "max_summary_unique_bullets",
                            "max_summary_citations", "max_summary_evidence_spans", "max_summary_value_bytes", "max_summary_scan_steps"])
                        .help("Retain independent cited chunk summaries without deterministic selection or neural synthesis"))
                } else { command };
                if name == "resume" {
                    command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                        .action(clap::ArgAction::SetTrue)
                        .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
                } else { command }
            });
        clap::Command::new(name).about(about).subcommand_required(true).subcommands(tasks)
    };
    clap::Command::new("document-job")
        .about("Start or resume retained whole-document extraction, summaries and question answering")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained whole-document INT4 job"))
        .subcommand(operation("resume", "Authenticate the original population and run pending documents"))
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
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0
            || self.lifetime.max_lifetime_source_fields == 0 || self.lifetime.max_lifetime_source_matches == 0
            || self.lifetime.max_lifetime_source_scan_steps == 0 {
            return Err(Failure::usage("explicit_document_job_storage"));
        }
        let map = self.map.validate_document_job(args, self.map_only)?;
        if (self.map.kind == Kind::Keyphrases) != self.keyphrase_lifetime.is_some()
            || self.keyphrase_lifetime.is_some_and(|lifetime|
                lifetime.max_lifetime_keyphrase_scan_work == Some(0)
                || (map.keyphrases.is_none() && lifetime.max_lifetime_keyphrase_scan_work.is_some())) {
            return Err(Failure::usage("lifetime_keyphrase_scan_work"));
        }
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > map.common.memory_bytes) {
                return Err(Failure::usage("document_job_memory"));
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
        let work = JobWork::from_int4(map.mapping.max_model_work, map.mapping.max_mask_visits);
        if !work.fits(job.max_work) { return Err(Failure::usage("lifetime_document_work_limits")); }
        self.lifetime.check(self.map.document_verification(map)?,
            map.summary.map_or(0, |summary| summary.aggregation.max_scan_steps), job.max_attempts)?;
        if map.keyphrases.is_some_and(|limits| limits.aggregation.max_scan_work.checked_mul(job.max_attempts)
            .is_none_or(|total| total > self.keyphrase_scan_cap())) {
            return Err(Failure::usage("lifetime_keyphrase_scan_work"));
        }
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = map.common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Match the host's complete population, journal and serialization
        // owners. The reused map quote already prices the maximum frontier,
        // every permitted pass and the independently retained document output.
        let buffer_bytes = sum(&[
            product(job.max_snapshot_bytes, 2)?, product(job.max_items, 1024)?,
            product(job.max_input_bytes_per_item as u64, 8)?,
            product(job.max_result_bytes as u64, 4)?, 2 * MIB,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes,
        ]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        map.common.minimum_memory_bytes = map.common.minimum_memory_bytes.checked_add(buffer_bytes)
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if map.common.minimum_memory_bytes > map.common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { map, transport, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
    fn mode_name(&self, limits: MapLimits) -> &'static str {
        if limits.question_hierarchy.is_some() { "question_hierarchy" }
        else if limits.question_synthesis.is_some() { "question_synthesis" }
        else if self.map.kind == Kind::Answer { "question_discovery" }
        else if limits.hierarchy.is_some() { "summary_hierarchy" }
        else if limits.synthesis.is_some() { "summary_synthesis" }
        else if limits.summary.is_some() { "summary_selection" }
        else if limits.keyphrases.is_some() { "keyphrase_ranking" }
        else { "source_map" }
    }
    fn keyphrase_scan_cap(&self) -> u64 {
        self.keyphrase_lifetime.and_then(|limits| limits.max_lifetime_keyphrase_scan_work)
            .unwrap_or(536_870_912_000)
    }
}
impl JobCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<(Self, Int4Args), clap::Error> {
        let invalid = |message| clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, message);
        let (operation, matches) = matches.subcommand().ok_or_else(|| invalid("document-job operation required"))?;
        let (name, matches) = matches.subcommand().ok_or_else(|| invalid("document-job task required"))?;
        let kind = Kind::named(name).ok_or_else(|| invalid("document-job task refused"))?;
        let mode = match operation {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(invalid("document-job operation refused")),
        };
        let args = JobArgs { map: MapArgs::from_matches(kind, matches)?,
            map_only: kind == Kind::Summarize && matches.get_flag("map_only"),
            storage: StorageArgs::from_arg_matches(matches)?, lifetime: LifetimeArgs::from_arg_matches(matches)?,
            keyphrase_lifetime: if kind == Kind::Keyphrases {
                Some(KeyphraseLifetimeArgs::from_arg_matches(matches)?)
            } else { None } };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name(); let task = self.args.map.kind.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, task, status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        let map = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, map, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, map, input, output); Err(Failure::usage("owned_document_job_profile_unavailable")) }
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
