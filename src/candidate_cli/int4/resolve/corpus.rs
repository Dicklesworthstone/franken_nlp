//! Raw entity epochs under one fixed policy and one resident INT4 invocation.
use super::*;
use crate::{
    candidate_cli::{batch::IO_BUFFER_BYTES, stream::Failure},
    corpus::{entities_int4::Int4EntityConfig, entity_stream::{EntityStreamLimits, ENTITY_STREAM_PROTOCOL}},
    native_engine::decode::DecodeCancellationKind,
    tasks::ner::NerOptions,
    validation::grounded_fields::GroundingBudget,
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

const POLICY_BYTES: usize = 16 * 1024;
const ENVELOPE_BYTES: usize = 16 * 1024;

#[derive(Args, Clone, Eq, PartialEq)]
struct StreamArgs {
    /// Fixed JSON {ner,options}; both policies are explicit and immutable for the run.
    #[arg(long, value_name = "FILE", required = true)]
    policy: PathBuf,
    /// Encoded NDJSON bytes before LF, including escaping and a trailing CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    /// Decoded text bytes per original document; defaults to the epoch input ceiling.
    #[arg(long)]
    max_document_bytes: Option<usize>,
    /// Nonblank records, including exact flush controls, across the entire run.
    #[arg(long, default_value_t = 1000)]
    max_records: u64,
    /// Completed epochs, including explicitly flushed empty epochs.
    #[arg(long, default_value_t = 1000)]
    max_snapshots: u64,
    /// Complete input stream bytes, including blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All canonical corpus envelopes and the mandatory terminal footer.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Source-grounding fields across every epoch; flush never replenishes this.
    #[arg(long, default_value_t = 1_000_000)]
    max_grounding_fields: usize,
    /// Source-grounding matches across every epoch, separate from per-epoch mentions.
    #[arg(long, default_value_t = 1_000_000)]
    max_grounding_matches: usize,
    /// Source-grounding scan work across the entire run.
    #[arg(long, default_value_t = 268_435_456)]
    max_grounding_scan_steps: u64,
}

#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs { resolution: ResolveArgs, stream: StreamArgs }
impl std::fmt::Debug for CorpusArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("EntityCorpusArgs") }
}

#[derive(Clone, Copy)]
pub(in crate::candidate_cli::int4) struct CorpusLimits {
    pub common: Int4Limits,
    transport: EntityStreamLimits,
    io_bytes: u64,
}

// Deliberately no Debug: a policy may contain private taxonomy information.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy { ner: NerOptions, options: ResolveOptions }
impl Policy {
    fn parse(text: &str) -> Result<Self, Failure> {
        if text.len() > POLICY_BYTES { return Err(input_failure("entity_corpus_policy_bytes")); }
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
            max_depth: 8, max_string_bytes: POLICY_BYTES,
        }).map_err(|_| input_failure("entity_corpus_policy_json"))?;
        let policy: Self = serde_json::from_value(value).map_err(|_| input_failure("entity_corpus_policy_shape"))?;
        policy.ner.validate().map_err(|_| input_failure("entity_corpus_ner_policy"))?;
        if policy.options.context_scalars > 2048
            || !(1..=1_000_000).contains(&policy.options.minimum_margin_milli) {
            return Err(input_failure("entity_corpus_resolution_policy"));
        }
        Ok(policy)
    }
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let mut command = StreamArgs::augment_args(super::definition().name("resolve-corpus")
        .about("Discover and resolve raw entity NDJSON epochs with one resident INT4 model")
        .long_about("Read bounded {id,text} NDJSON and exact {\"flush\":true} controls. --discover-entities and --policy FILE are required; the policy is one bounded JSON {ner,options} object using the same NER and resolution options as resolve. At each flush, and at EOF for a nonempty tail, discover mentions in original documents and resolve the complete epoch with both presentation orders and complete-link clustering. --chunked discovers across every original source chunk. Cluster IDs are local to one epoch; no cross-epoch or corpus-wide clustering is claimed. A single resident model, native engine, checkpoint allowance, five-axis work ceiling, total mask ceiling and grounding ledger serve the entire invocation. Flush does not renew those limits. Each corpus event is a completed epoch; whole-run success requires run_complete AND exit 0 after physical native drain. Invalid input, exhausted budgets, failed inference or failed output stop the run without a footer. Earlier completed epochs remain valid. Candidate output is not release qualification, calibration or evidence that every real entity was detected. Blocking IO observes the total deadline cooperatively.")
        .mut_arg("discover_entities", |arg| arg.required(true))
        .mut_arg("input", |arg| arg.help("Raw {id,text} NDJSON file or '-' for stdin; fixed policies come from --policy"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Maximum original ID plus text bytes retained in one complete epoch"))
        .mut_arg("max_documents", |arg| arg.help("Maximum original documents per epoch; flush releases their IDs"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native result per epoch; fixed transport envelope is additional"))
        .mut_arg("max_snapshot_mask_node_visits", |arg| arg.help("One source-mask allowance for the ENTIRE stream (default 1000000000000); never renewed at flush"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for the ENTIRE corpus execution")));
    for name in ["max_forward_positions", "max_projected_logits", "max_attention_pairs",
        "max_dot_products", "max_multiply_accumulates"] {
        command = command.mut_arg(name, |arg| arg.help("Whole-stream native work ceiling, shared by all NER chunks, pairs and epochs"));
    }
    command
}

impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { resolution: ResolveArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli::int4) fn validate(&self, args: &Int4Args) -> Result<CorpusLimits, Failure> {
        if !self.resolution.discovery.discover_entities || self.stream.policy.as_os_str().is_empty()
            || self.stream.policy.as_os_str() == "-" {
            return Err(Failure::usage("entity_corpus_policy_required"));
        }
        let mut common = self.resolution.validate(args).map_err(candidate_failure)?;
        let s = &self.stream;
        let document_bytes = s.max_document_bytes.unwrap_or(args.max_input_bytes);
        if !(1..=100_000).contains(&s.max_records) || !(1..=100_000).contains(&s.max_snapshots)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=args.max_input_bytes).contains(&document_bytes) || s.max_line_bytes < document_bytes
            || !(1..=1_000_000).contains(&s.max_grounding_fields)
            || !(1..=1_000_000).contains(&s.max_grounding_matches)
            || !(1..=1_000_000_000_000).contains(&s.max_grounding_scan_steps) {
            return Err(Failure::usage("entity_corpus_transport_options"));
        }
        let output_line = common.result_bytes.checked_add(ENVELOPE_BYTES)
            .ok_or_else(|| Failure::usage("entity_corpus_size_arithmetic"))?;
        // A full epoch must be deliverable while retaining the reserved footer.
        if s.max_total_output_bytes < output_line as u64 + ENVELOPE_BYTES as u64 {
            return Err(Failure::usage("entity_corpus_output_too_small"));
        }
        let transport = EntityStreamLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: document_bytes,
            max_corpus_bytes: args.max_input_bytes, max_documents: self.resolution.graph.max_documents,
            max_input_bytes: s.max_total_input_bytes, max_records: s.max_records, max_snapshots: s.max_snapshots,
            max_output_line_bytes: output_line, max_output_bytes: s.max_total_output_bytes,
        };
        let io_bytes = IO_BUFFER_BYTES as u64 * 2 + POLICY_BYTES as u64 + ENVELOPE_BYTES as u64;
        let task = self.configuration(args, common, Policy { ner: NerOptions::default(), options: ResolveOptions::default() })?;
        if transport.max_corpus_bytes > task.graph.max_input_bytes {
            return Err(Failure::usage("entity_corpus_original_capacity"));
        }
        #[cfg(feature = "asupersync-runtime")]
        let additional_minimum = runtime::memory_floor(args, self, common, transport, io_bytes, &task)?;
        #[cfg(not(feature = "asupersync-runtime"))]
        let additional_minimum = {
            let _ = task;
            sum(&[common.minimum_memory_bytes, io_bytes, transport.max_line_bytes as u64 * 8,
                transport.max_output_line_bytes as u64 * 8]).map_err(candidate_failure)?
        };
        common.minimum_memory_bytes = common.minimum_memory_bytes.max(additional_minimum);
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
                code: "entity_corpus_memory_floor", cancellation: None });
        }
        Ok(CorpusLimits { common, transport, io_bytes })
    }

    fn configuration(&self, args: &Int4Args, common: Int4Limits, policy: Policy) -> Result<Int4EntityConfig, Failure> {
        let mut task = self.resolution.discovery.configuration(self.resolution, args, common,
            policy.ner, policy.options).map_err(candidate_failure)?;
        task.verification = GroundingBudget { max_fields: self.stream.max_grounding_fields,
            max_matches: self.stream.max_grounding_matches, max_scan_steps: self.stream.max_grounding_scan_steps };
        task.validate().map_err(|_| Failure::usage("entity_corpus_task_options"))?;
        Ok(task)
    }

    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: ENTITY_STREAM_PROTOCOL, schema_version: 1, event: "run_error",
                    scope: "real-source-current-candidate-int4", evidence: "non_authoritative", status: "incomplete",
                    code: failure.code, exit_code: failure.exit, cancellation: failure.cancellation,
                    completed_epochs_may_exist: true, cluster_scope: "epoch_local" };
                if publish(&report, ENVELOPE_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }

    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(Failure::usage("entity_corpus_profile_unavailable")) }
    }
}

fn input_failure(code: &'static str) -> Failure {
    Failure { exit: ErrorCode::InputDecodeOrParse, code, cancellation: None }
}
fn candidate_failure(error: CandidateError) -> Failure {
    let (exit, code) = match error {
        CandidateError::Arguments => (ErrorCode::Usage, "entity_corpus_options"),
        CandidateError::Input => (ErrorCode::InputDecodeOrParse, "entity_corpus_input"),
        CandidateError::Planning => (ErrorCode::SchemaOrRecipeCompile, "entity_corpus_plan_refused"),
        CandidateError::Memory | CandidateError::Runtime => (ErrorCode::AdmissionOrResourceLimit, "entity_corpus_admission"),
        CandidateError::Identity => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "entity_corpus_model_identity"),
        CandidateError::Timeout => (ErrorCode::BudgetOrTimeout, "entity_corpus_invocation_deadline"),
        _ => (ErrorCode::Generic, "entity_corpus_setup_or_delivery"),
    };
    Failure { exit, code, cancellation: (error == CandidateError::Timeout).then_some(DecodeCancellationKind::Deadline) }
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, event: &'static str, scope: &'static str,
    evidence: &'static str, status: &'static str, code: &'static str, exit_code: ErrorCode,
    cancellation: Option<DecodeCancellationKind>, completed_epochs_may_exist: bool, cluster_scope: &'static str,
}
