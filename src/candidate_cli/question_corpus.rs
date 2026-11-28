//! Per-record questions over complete original documents on resident INT8.
//! Shared document policy is also consumed by the authenticated durable route.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork, source::quantized::question::{Int8QuestionDocumentBatchConfig,
        memory::{Int8QuestionDocumentMemoryLimits, Int8QuestionDocumentMemoryRequirement}}},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}, stream::Failure},
    native_engine::{portable_int8::ProjectionWork,
        strict_int8::{Int8Work, prefill::Int8PrefillLimits}},
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
    validation::grounded_fields::GroundingBudget,
};

/// Fixed policy for each whole source/question pair. Population work ceilings
/// are supplied separately, and records cannot replace this schema or policy.
#[derive(Args)]
pub(in crate::candidate_cli) struct QuestionDocumentArgs {
    #[command(flatten)]
    pub host: source::SourceHostArgs,
    #[command(flatten)]
    pub prefill: source::prefill::SourcePrefillArgs,
    /// Complete AnswerOptions JSON file fixed for every question; defaults only when omitted.
    #[arg(long, value_name = "FILE")]
    pub options: Option<PathBuf>,
    /// Exact decoded question bytes per record, independently of complete source bytes.
    #[arg(long, default_value_t = 16_384)]
    pub max_question_bytes: usize,
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Original source bytes per candidate chunk; actual question/manifest context fit tightens this.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete answer document, including every independent answer and source range.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_live_value_bytes: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_value_bytes: usize,
    #[arg(long, default_value_t = 64)]
    reduction_reserve_mib: u64,
    /// Complete-document mask allowance, separate from per-chunk and whole-run limits.
    #[arg(long, default_value_t = 64_000_000_000)]
    max_document_mask_node_visits: u64,
    #[arg(long, default_value_t = 4096)]
    max_qa_citations: usize,
    #[arg(long, default_value_t = 16_384)]
    max_qa_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_qa_scan_steps: u64,
    #[arg(long, default_value_t = 262_144)]
    max_document_forward_positions: u64,
    #[arg(long, default_value_t = 10_000_000_000)]
    max_document_projected_logits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_document_attention_pairs: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_document_dot_products: u64,
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_document_multiply_accumulates: u64,
    #[arg(skip)]
    encoding: input_encoding::InputEncodingArgs,
    #[command(flatten)]
    encoding_work: jobs::EncodingWorkArgs,
    #[arg(skip)]
    compute: compute::Args,
}

pub(in crate::candidate_cli) struct QuestionDocumentLimits {
    pub common: CandidateArgs,
    pub shared: Limits,
    pub task: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int8SourceMapLimits,
    pub verification: GroundingBudget,
    pub max_question_bytes: usize,
    pub encoding: UntrustedEncoding,
    pub encoding_work: Option<SafeBpeWorkLimits>,
    pub prefill: Option<Int8PrefillLimits>,
    pub reduction_reserve_bytes: u64,
}

/// Compose these fixed document options into either a live corpus or a durable
/// job definition. The caller supplies its own input and population arguments.
pub(in crate::candidate_cli) fn document_definition(command: clap::Command) -> clap::Command {
    compute::definition(input_encoding::definition(QuestionDocumentArgs::augment_args(command)))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Maximum decoded COMPLETE original source bytes per record; question has its own bound"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Per-chunk mask visits, including the terminal EOS"))
        .mut_arg("max_checkpoints", |arg| arg.help("Original cooperative checkpoint ceiling across the complete population"))
        .mut_arg("compute_threads", |arg| arg.help("One persistent INT8 projection team, 1..=64, across every record and chunk; composes with --prefill-rows"))
        .mut_arg("max_total_tokenizer_input_bytes", |arg| arg.help("Bytes encoded across ONE complete original source/question preparation and reconstruction, including failed probes; safe-bpe-v1 only"))
        .mut_arg("max_total_tokenizer_output_tokens", |arg| arg.help("Encoded tokens across ONE complete original source/question preparation and reconstruction; safe-bpe-v1 only"))
        .mut_arg("max_total_tokenizer_pair_checks", |arg| arg.help("BPE pair checks across ONE complete original source/question preparation and reconstruction; safe-bpe-v1 only"))
        .mut_arg("max_total_tokenizer_pair_bytes", |arg| arg.help("Examined BPE pair bytes across ONE complete original source/question preparation and reconstruction; safe-bpe-v1 only"))
}

impl QuestionDocumentArgs {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut args = Self::from_arg_matches(matches)?;
        args.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        args.compute = compute::Args::from_matches(matches)?;
        Ok(args)
    }

    pub(in crate::candidate_cli) fn admit(&self, input: PathBuf) -> Result<QuestionDocumentLimits, CandidateError> {
        let (mut common, mut shared) = self.host.common(input)?;
        let prefill = self.prefill.limits()?;
        common.compute = self.compute;
        shared.compute = self.compute.admit_prefill(prefill)?;
        if self.options.as_ref().is_some_and(|p| p.as_os_str().is_empty() || p.as_os_str() == "-")
            || !(4..=MAX_INPUT_BYTES).contains(&self.host.max_input_bytes)
            || !(1..=MAX_INPUT_BYTES).contains(&self.max_question_bytes)
            || !(1..=256).contains(&self.max_chunks)
            || !(1..=1_000_000).contains(&self.max_tokenizer_calls)
            || self.max_chunk_bytes.is_some_and(|bytes| !(4..=MAX_INPUT_BYTES).contains(&bytes))
            || [self.max_map_result_bytes, self.max_live_value_bytes, self.max_total_value_bytes]
                .iter().any(|&bytes| !(1..=64 * 1024 * 1024).contains(&bytes))
            || self.host.max_result_bytes > self.max_map_result_bytes
            || self.max_map_result_bytes > self.max_live_value_bytes
            || self.max_map_result_bytes > self.max_total_value_bytes
            || self.max_document_mask_node_visits < self.host.max_mask_node_visits
            || self.max_document_mask_node_visits > 1_000_000_000_000_000
            || self.max_document_forward_positions > 16 * 1024 * 1024 {
            return Err(CandidateError::Arguments);
        }
        let task = self.host.task_budget(shared);
        let mut planning = self.host.planning();
        // Native per-prompt bytes include question and passage manifest. The
        // complete source envelope remains independently bounded by chunks.
        planning.max_input_bytes = planning.max_input_bytes.max(self.max_question_bytes
            .checked_add(4096).ok_or(CandidateError::Arguments)?);
        planning.validate().map_err(|_| CandidateError::Arguments)?;
        let chunk_bytes = self.max_chunk_bytes.unwrap_or(self.host.context_tokens)
            .min(self.host.max_input_bytes);
        let chunks = ChunkLimits { max_input_bytes: self.host.max_input_bytes,
            max_chunk_bytes: chunk_bytes, max_chunk_tokens: self.host.context_tokens,
            context_tokens: self.host.context_tokens, reserved_tokens: self.host.max_new_tokens,
            max_chunks: self.max_chunks, max_tokenizer_calls: self.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let work = Int8Work { forward_positions: self.max_document_forward_positions,
            projected_logits: self.max_document_projected_logits, attention_pairs: self.max_document_attention_pairs,
            projections: ProjectionWork { dot_products: self.max_document_dot_products,
                multiply_accumulates: self.max_document_multiply_accumulates } };
        validate_work(work)?;
        let mapping = Int8SourceMapLimits { chunks,
            reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8, max_reduction_levels: 8,
                max_task_calls: 1024, max_value_bytes: self.max_map_result_bytes,
                max_live_value_bytes: self.max_live_value_bytes, max_total_value_bytes: self.max_total_value_bytes,
                max_result_bytes: self.max_map_result_bytes },
            max_model_work: work, mask_limits: self.host.masks(), mask_visits_per_chunk: self.host.max_mask_node_visits,
            max_mask_visits: self.max_document_mask_node_visits };
        mapping.validate(task, planning).map_err(|_| CandidateError::Arguments)?;
        let verification = GroundingBudget { max_fields: self.max_qa_citations,
            max_matches: self.max_qa_evidence_spans, max_scan_steps: self.max_qa_scan_steps };
        if !(1..=1_000_000).contains(&verification.max_fields)
            || !(1..=1_000_000).contains(&verification.max_matches)
            || !(1..=1_000_000_000_000).contains(&verification.max_scan_steps) {
            return Err(CandidateError::Arguments);
        }
        let per_call_bytes = chunk_bytes.checked_add(self.max_question_bytes)
            .and_then(|bytes| bytes.checked_add(4096)).ok_or(CandidateError::Arguments)?
            .min(planning.max_input_bytes);
        let encoding = self.encoding.policy(per_call_bytes, per_call_bytes)?;
        let encoding_work = self.encoding_work.for_document(&encoding).map_err(|_| CandidateError::Arguments)?;
        let encoding_workspace = match (&encoding, encoding_work) {
            (UntrustedEncoding::ByteFallback, None) => 0,
            (UntrustedEncoding::SafeBpeV1(call), Some(total)) => {
                if total.remaining_call(SafeBpeWork::default(), *call).ok() != Some(*call) {
                    return Err(CandidateError::Arguments);
                }
                call.requirements().map_err(|_| CandidateError::Arguments)?.workspace_bytes
                    .checked_add(1024).ok_or(CandidateError::Arguments)?
            }
            _ => return Err(CandidateError::Arguments),
        };
        let per_chunk = sum(&[self.host.context_tokens as u64 * 32,
            u64::from(task.max_grammar_states) * 512, source::OPTIONS_BYTES as u64 * 2])?;
        let floor = sum(&[mul(per_chunk, self.max_chunks as u64)?, 256 * MIB,
            mul(self.host.max_input_bytes as u64, 64)?,
            mul(self.max_map_result_bytes as u64 + 4096, 2)?, encoding_workspace])?;
        let reduction_reserve_bytes = self.reduction_reserve_mib.checked_mul(MIB)
            .filter(|&bytes| bytes != 0 && bytes <= shared.memory_bytes).ok_or(CandidateError::Arguments)?;
        if shared.preparation_bytes < floor { return Err(CandidateError::Arguments); }
        Ok(QuestionDocumentLimits { common, shared, task, planning, mapping, verification,
            max_question_bytes: self.max_question_bytes, encoding, encoding_work, prefill, reduction_reserve_bytes })
    }

    pub(in crate::candidate_cli) fn configuration(&self, limits: &QuestionDocumentLimits, options: AnswerOptions,
        max_model_work: Int8Work, max_mask_visits: u64, max_verification: GroundingBudget)
        -> Result<Int8QuestionDocumentBatchConfig, CandidateError> {
        options.validate().map_err(|_| CandidateError::Arguments)?;
        validate_work(max_model_work)?;
        let config = Int8QuestionDocumentBatchConfig { options, budget: limits.task, planning: limits.planning,
            mapping: limits.mapping, verification: limits.verification, max_question_bytes: limits.max_question_bytes,
            max_model_work, max_mask_visits, max_verification };
        config.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(config)
    }
}

pub(in crate::candidate_cli) fn parse_options(raw: Option<&str>) -> Result<AnswerOptions, CandidateError> {
    let options = source::options::<AnswerOptions>(raw)?;
    options.validate().map_err(|_| CandidateError::Arguments)?;
    options.schema_source().map_err(|_| CandidateError::Planning)?;
    Ok(options)
}

pub(in crate::candidate_cli) fn document_memory_limits(limits: &QuestionDocumentLimits, io_bytes: u64)
    -> Int8QuestionDocumentMemoryLimits {
    Int8QuestionDocumentMemoryLimits { context_tokens: limits.common.context_tokens,
        allocator_reserve_bytes: 64 * MIB, preparation_reserve_bytes: limits.shared.preparation_bytes,
        io_reserve_bytes: io_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes }
}

#[derive(Args)]
struct StreamArgs {
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    /// Nonrenewable native work across the entire stream, including failed attempts.
    #[arg(long, default_value_t = 16_777_216)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000_000)]
    max_attention_pairs: u64,
    #[arg(long, default_value_t = 100_000_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000_000)]
    max_multiply_accumulates: u64,
    #[arg(long, default_value_t = 128_000_000_000_000)]
    max_run_mask_node_visits: u64,
    #[arg(long, default_value_t = 4_096_000)]
    max_run_qa_citations: usize,
    #[arg(long, default_value_t = 16_384_000)]
    max_run_qa_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864_000)]
    max_run_qa_scan_steps: u64,
}

pub(crate) struct QuestionCorpusCommand {
    pub(in crate::candidate_cli) input: PathBuf,
    pub(in crate::candidate_cli) document: QuestionDocumentArgs,
    stream: StreamArgs,
}
pub(in crate::candidate_cli) struct Admitted {
    pub document: QuestionDocumentLimits,
    pub config: Int8QuestionDocumentBatchConfig,
    pub transport: BatchLimits,
    pub memory: Int8QuestionDocumentMemoryRequirement,
    pub output_bytes: u64,
    pub io_bytes: u64,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    StreamArgs::augment_args(document_definition(clap::Command::new("answer-corpus")
        .about("Answer each record's question across its complete original source on resident INT8")
        .arg(clap::Arg::new("input").default_value("-").value_parser(clap::value_parser!(PathBuf))
            .help("NDJSON file or '-' for stdin; strict {id,text,task_args:{question}} records"))))
        .long_about("Stream complete original documents with an exact private question on each record through one resident INT8 model. Input is strict {id,text,task_args:{question}} NDJSON; task_args contains only question. Exact {\"flush\":true} controls reset duplicate-ID tracking only. Every nonblank source chunk sees that record's unchanged question. Results retain all independent passage answers, abstentions and whitespace ranges, with byte/scalar citations in the original document. All native chunks and complete output are admitted before inference. A failed chunk emits no partial document result. --options fixes AnswerOptions for the population. --max-document-* limits each complete document; the five --max-* native work flags and --max-run-* mask/QA verification flags bound the entire invocation, including failed attempts. Compact input shares one tokenizer ledger across a record's question, sizing probes, complete preparation and exact reconstruction; the next independent record receives the same fixed per-document allowance. --prefill-rows and --compute-threads compose on one process-owned native engine; terminal completion follows physical worker join. Earlier completed item frames remain valid after later failure. Require run_complete and exit 0 for whole-stream success. Exact citations establish source membership; answers remain uncalibrated and are not merged into a synthesized global answer.")
}

impl QuestionCorpusCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { input: matches.get_one::<PathBuf>("input").cloned().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingRequiredArgument, "input required"))?,
            document: QuestionDocumentArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }
    pub(in crate::candidate_cli) fn admit(&self) -> Result<Admitted, CandidateError> {
        self.admit_options(AnswerOptions::default())
    }
    pub(in crate::candidate_cli) fn admit_options(&self, options: AnswerOptions) -> Result<Admitted, CandidateError> {
        let document = self.document.admit(self.input.clone())?;
        let s = &self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || s.max_line_bytes < document.common.max_input_bytes || s.max_line_bytes < document.max_question_bytes
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_mask_node_visits)
            || !(1..=1_000_000_000).contains(&s.max_run_qa_citations)
            || !(1..=1_000_000_000).contains(&s.max_run_qa_evidence_spans)
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_qa_scan_steps)
            || s.max_forward_positions > 16 * 1024 * 1024 {
            return Err(CandidateError::Arguments);
        }
        let work = Int8Work { forward_positions: s.max_forward_positions, projected_logits: s.max_projected_logits,
            attention_pairs: s.max_attention_pairs, projections: ProjectionWork {
                dot_products: s.max_dot_products, multiply_accumulates: s.max_multiply_accumulates } };
        let config = self.document.configuration(&document, options, work, s.max_run_mask_node_visits,
            GroundingBudget { max_fields: s.max_run_qa_citations, max_matches: s.max_run_qa_evidence_spans,
                max_scan_steps: s.max_run_qa_scan_steps })?;
        let framing = s.max_requests.checked_add(4).and_then(|count| count.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let output_line = config.mapping.reduction.max_result_bytes.checked_add(FRAME_ALLOWANCE as usize)
            .ok_or(CandidateError::Arguments)?;
        let transport = BatchLimits { max_line_bytes: s.max_line_bytes,
            max_document_bytes: document.common.max_input_bytes, max_id_bytes: 128,
            max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: output_line,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits } };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[output_line as u64, FRAME_ALLOWANCE, 2 * IO_BUFFER_BYTES as u64])?;
        let memory = document_memory_limits(&document, io_bytes).requirements(config, transport)
            .map_err(|_| CandidateError::Arguments)?;
        let admitted = Admitted { document, config, transport, memory,
            output_bytes: s.max_total_output_bytes, io_bytes };
        validate_peak(&admitted)?;
        Ok(admitted)
    }
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        let result = self.admit().map_err(|_| Failure::usage("int8_question_corpus_options"))
            .and_then(|admitted| {
                #[cfg(feature = "asupersync-runtime")]
                { runtime::question_corpus::execute(self, admitted, input, output) }
                #[cfg(not(feature = "asupersync-runtime"))]
                { let _ = (self, admitted, input, output);
                    Err(Failure::usage("int8_question_corpus_profile_unavailable")) }
            });
        match result { Ok(()) => ExitCode::SUCCESS, Err(failure) => report_failure(failure, diagnostics) }
    }
}

pub(in crate::candidate_cli) fn validate_peak(admitted: &Admitted) -> Result<u64, CandidateError> {
    let d = &admitted.document; let m = admitted.memory;
    if d.shared.compute != d.common.compute.admit_prefill(d.prefill)? { return Err(CandidateError::Arguments); }
    let prefill = d.prefill.map(Int8PrefillLimits::validate).transpose()
        .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
    let loading = sum(&[d.shared.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[d.shared.weight_bytes, 192 * MIB, m.native.kv_bytes, m.workspace_bytes,
        m.input_bytes, m.document_workspace_bytes, m.output_reservation_bytes,
        d.shared.compute.workspace_bytes, prefill, 65_536])?;
    let peak = sum(&[d.shared.preparation_bytes, loading.max(execution)])?;
    if peak > d.shared.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(peak)
}
fn validate_work(work: Int8Work) -> Result<(), CandidateError> {
    if work.forward_positions == 0
        || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0 {
        return Err(CandidateError::Arguments);
    }
    Ok(())
}
fn sum(values: &[u64]) -> Result<u64, CandidateError> {
    values.iter().try_fold(0_u64, |total, bytes| total.checked_add(*bytes).ok_or(CandidateError::Arguments))
}
fn mul(a: u64, b: u64) -> Result<u64, CandidateError> { a.checked_mul(b).ok_or(CandidateError::Arguments) }

pub(in crate::candidate_cli) fn report_failure(failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    match failure.cancellation {
        Some(cause) => { let _ = writeln!(diagnostics, "fnlp candidate answer-corpus: {} ({cause:?})", failure.code); }
        None => { let _ = writeln!(diagnostics, "fnlp candidate answer-corpus: {}", failure.code); }
    }
    failure.exit.as_process_exit()
}

#[cfg(test)] mod tests;
