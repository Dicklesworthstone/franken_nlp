//! Exact-schema extraction over an ordered stream of complete original sources.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        extract::grouped_int4::long::stream::memory::{Int4ExtractionDocumentMemoryLimits,
            Int4ExtractionDocumentMemoryRequirement}},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
    native_engine::decode::DecodeStepControl,
    tasks::extract::object_reconcile::RootObjectLimits,
};
use super::object::ReconciliationArgs;

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Whole-run nonempty records, including malformed records and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Every input byte, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Maximum NDJSON bytes before LF, including JSON syntax and a trailing CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    /// Nonrenewable grammar-mask allowance across every admitted document attempt.
    #[arg(long, default_value_t = 128_000_000_000_000)]
    max_run_mask_node_visits: u64,
    /// Whole-run original-coordinate field-verification allowance, including failed attempts.
    #[arg(long, default_value_t = 4_096_000)]
    max_run_extraction_fields: usize,
    /// Whole-run original-coordinate evidence-span allowance, including failed attempts.
    #[arg(long, default_value_t = 16_384_000)]
    max_run_extraction_evidence_spans: usize,
    /// Whole-run independent verification scan allowance, including failed attempts.
    #[arg(long, default_value_t = 67_108_864_000)]
    max_run_extraction_scan_steps: u64,
}
impl StreamArgs {
    fn verification(self) -> GroundingBudget {
        GroundingBudget { max_fields: self.max_run_extraction_fields,
            max_matches: self.max_run_extraction_evidence_spans,
            max_scan_steps: self.max_run_extraction_scan_steps }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs {
    document: MapArgs,
    stream: StreamArgs,
    reconciliation: ReconciliationArgs,
}
impl std::fmt::Debug for CorpusArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("ExtractionCorpusArgs") }
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    ReconciliationArgs::augment_definition(StreamArgs::augment_args(super::corpus_definition().name("extract-corpus")
        .about("Extract one exact schema from every chunk of ordered complete documents")
        .long_about(concat!(
            "Stream strict {id,text,task_args?} NDJSON and exact {\"flush\":true} controls through one resident grouped-INT4 model. ",
            "text is the COMPLETE original source. --schema fixes one exact raw schema; task_args must be absent, null or an empty object. ",
            "Preflight every real chunk before that document's first native forward. By default, each chunk independently satisfies the complete schema. ",
            "--reconcile-root-object instead derives a separately authenticated partial root schema and combines complementary properties into one original-schema-valid object. ",
            "Only root required fields are relaxed for chunks; nested values stay atomic. Equal values retain all origins; conflicts and missing final required properties fail the document. ",
            "This mode conflicts with --root-array-union. --max-object-* bounds the complete reducer and adds schema/reconciliation memory before input or model IO. ",
            "--source-membership retains chunk evidence and verifies reconciled fields against the complete original source using the remaining document verification allowance. ",
            "No partial document succeeds. Documents execute sequentially; --cohort-chunks shares native layers within a document and --max-selected-rows N evaluates every legal grammar row, including accepting EOS. ",
            "Oversized legal sets refuse without pruning or fallback. All five native work flags bound the ENTIRE RUN. ",
            "--max-run-mask-node-visits and --max-run-extraction-* bound mask and source-verification work across every admitted attempt. ",
            "Failed attempts, cohorts and flushes never renew these allowances. Every completed event carries public candidate provenance. ",
            "Require run_complete AND exit 0 for whole-run success; earlier completed records remain valid after later failure. ",
            "Source conversion and independently pinned snapshots share this route. Optional snapshot saving follows fixed schema/model/envelope admission and may remain after a later row fails. ",
            "No retained job, input spool, per-row overrides, semantic correctness claim, prompt prefill, cross-document cohorts, sampling or repair retry. ",
            "Blocking IO observes the end-to-end deadline cooperatively."))
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; each text is a complete original document"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded UTF-8 bytes of each COMPLETE original document"))
        .mut_arg("max_map_result_bytes", |arg| arg.help("Complete semantic document result cap; transport/provenance framing is additional"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Per-chunk grammar-mask allowance, including EOS; document/run caps also apply"))
        .mut_arg("max_total_mask_node_visits", |arg| arg.help("Complete-document grammar-mask cap; --max-run-mask-node-visits bounds the whole stream"))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoint allowance for the ENTIRE stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream native forward ceiling across all documents and chunks"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream projected-logit ceiling under the selected head strategy"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention-pair ceiling"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream grouped native dot-product ceiling"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling"))))
}

#[derive(Clone, Copy, Debug)]
pub(in crate::candidate_cli::int4) struct CorpusLimits {
    common: Int4Limits,
    document: MapLimits,
    transport: TransportLimits,
    memory: Int4ExtractionDocumentMemoryRequirement,
    max_run_mask_visits: u64,
    max_run_verification: GroundingBudget,
    output_bytes: u64,
    io_bytes: u64,
    cli_preparation_bytes: u64,
    root_object: Option<RootObjectLimits>,
    object_preparation_bytes: u64,
}

impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { document: MapArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)?,
            reconciliation: ReconciliationArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli::int4) fn validate(&self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        let document = self.document.validate_corpus(args)?;
        let mut common = document.common;
        let root_object = self.reconciliation.limits(self.document.mapping)?;
        let encoding = self.document.encoding.policy()?;
        if root_object.is_some() && encoding != crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback {
            return Err(CandidateError::Arguments);
        }
        let object_preparation_bytes = self.reconciliation.workspace_bytes(&self.document, document)?;
        let s = self.stream;
        let verification = s.verification();
        let per_document = document.mapping.verification;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_mask_node_visits)
            || s.max_run_mask_node_visits < document.mapping.mapping.max_mask_visits
            || !(1..=1_000_000_000).contains(&verification.max_fields)
            || !(1..=1_000_000_000).contains(&verification.max_matches)
            || !(1..=1_000_000_000_000_000_000).contains(&verification.max_scan_steps)
            || verification.max_fields < per_document.max_fields
            || verification.max_matches < per_document.max_matches
            || verification.max_scan_steps < per_document.max_scan_steps {
            return Err(CandidateError::Arguments);
        }
        let framing = s.max_requests.checked_add(4).and_then(|count| count.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let work = document.mapping.mapping.max_model_work;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: document.max_result_bytes.checked_add(FRAME_ALLOWANCE as usize)
                .ok_or(CandidateError::Arguments)?,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            2 * IO_BUFFER_BYTES as u64])?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes, io_bytes])?;
        let budget = self.document.extraction.budget(args, common);
        let memory = Int4ExtractionDocumentMemoryLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, preparation_reserve_bytes: common.preparation_bytes,
            io_reserve_bytes: io_bytes, reduction_reserve_bytes: document.reduction_reserve_bytes,
            cohort_chunks: document.cohort_chunks }
            .requirements_with_root_object_and_encoding(args.group_width, budget, document.mapping, transport,
                self.document.extraction.compiler(), root_object, &encoding)
            .map_err(|_| CandidateError::Arguments)?;
        let execution = sum(&[memory.native.kv_bytes, memory.workspace_bytes, memory.input_bytes,
            memory.document_workspace_bytes, memory.output_reservation_bytes])?;
        // The separate original/partial-schema charge remains owned during
        // model loading and optional snapshot saving. It is relinquished before
        // the hosted invocation takes the full object-map frontier charge.
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
        let preparation_peak = sum(&[loading.max(saving), object_preparation_bytes])?;
        common.minimum_memory_bytes = sum(&[
            grouped_memory_floor(args, common, execution.max(preparation_peak))?, io_bytes])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, document, transport, memory,
            max_run_mask_visits: s.max_run_mask_node_visits, max_run_verification: verification,
            output_bytes: s.max_total_output_bytes, io_bytes, cli_preparation_bytes,
            root_object, object_preparation_bytes })
    }

    fn load_defaults(&self, schema: String, budget: TaskBudget) -> Result<ExtractionBatchArgs, CandidateError> {
        let request = self.document.request(schema, budget)?;
        // Check the declaration without inventing a document for verbatim
        // enum feasibility. The native map binds it to each exact real chunk.
        let declaration = crate::grammar::runtime::JsonValidationSchema::compile(
            &request.schema, self.document.extraction.compiler()).map_err(|_| CandidateError::Planning)?;
        if declaration.requires_verbatim_source()
            != (request.grounding == crate::batch::extract::ExtractionBatchGrounding::SourceMembership) {
            return Err(CandidateError::Planning);
        }
        Ok(request)
    }

    fn load_defaults_with_control<C: DecodeStepControl>(&self, schema: String,
        budget: TaskBudget, control: &mut C) -> Result<ExtractionBatchArgs, CandidateError> {
        if self.reconciliation.limits(self.document.mapping)?.is_some() {
            self.reconciliation.request(&self.document, schema, budget, control)
        } else {
            self.load_defaults(schema, budget)
        }
    }

    pub(in crate::candidate_cli::int4) fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }

    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate extract-corpus: {}", diagnostic(error));
                error.exit_code()
            }
        }
    }
}

fn diagnostic(error: CandidateError) -> &'static str {
    match error {
        CandidateError::Timeout => "cooperative corpus deadline expired; earlier completed frames may exist",
        CandidateError::Execution => "native corpus execution failed or was cancelled; earlier completed frames may exist",
        _ => error.message(),
    }
}
