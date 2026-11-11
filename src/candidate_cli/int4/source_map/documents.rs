//! Ordered complete-document source tasks with one fixed resident configuration.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        source::grouped_int4::document::Int4SourceDocumentBatchLimits},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
};

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
    /// Nonrenewable grammar-mask allowance across all admitted document attempts.
    #[arg(long, default_value_t = 128_000_000_000_000)]
    max_run_mask_node_visits: u64,
    /// Independent original-source verification fields across the entire stream.
    #[arg(long, default_value_t = 4_096_000)]
    max_run_source_fields: usize,
    /// Independent original-source evidence occurrences across the entire stream.
    #[arg(long, default_value_t = 16_384_000)]
    max_run_source_evidence_spans: usize,
    /// Nonrenewable source-verification scan work, including failed attempts.
    #[arg(long, default_value_t = 67_108_864_000)]
    max_run_source_scan_steps: u64,
    /// Independent exact-summary reducer scan work across admitted documents.
    #[arg(long, default_value_t = 536_870_912_000)]
    max_run_summary_scan_steps: u64,
    /// Independent ranked-keyphrase reducer scan work across admitted documents.
    #[arg(long, default_value_t = 536_870_912_000)]
    max_run_keyphrase_scan_work: u64,
}

#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs { document: MapArgs, stream: StreamArgs }
impl std::fmt::Debug for CorpusArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("SourceDocumentCorpusArgs") }
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    clap::Command::new("source-corpus")
        .about("Stream complete documents through a fixed NER, keyphrase, summary or question pipeline")
        .subcommand_required(true)
        .subcommands(["ner", "keyphrases", "summarize", "answer"].into_iter()
            .zip(all_definitions()).map(|(name, command)| command.name(name)))
}
pub(in crate::candidate_cli::int4) fn definitions() -> Vec<clap::Command> {
    // Per-record questions have their own answer-corpus contract. The fixed
    // question and full hierarchy route is source-corpus answer --question.
    all_definitions().into_iter().filter(|command| command.get_name() != "answer-corpus").collect()
}
fn all_definitions() -> Vec<clap::Command> {
    ["ner-corpus", "keyphrases-corpus", "summarize-corpus", "answer-corpus"]
        .into_iter().zip(super::document_job_definitions()).map(|(name, command)| {
            StreamArgs::augment_args(command.name(name)
                .about("Stream complete original documents through one resident source-task pipeline")
                .long_about("Process strict {id,text,task_args?} NDJSON and exact {\"flush\":true} controls with one resident grouped-INT4 model. text is the COMPLETE original document. --options fixes the per-chunk task options; source-corpus answer also requires one exact --question FILE for the entire stream. task_args must be absent, null or an empty object. Every document is losslessly chunked and fully preflighted before its first native forward. NER and default keyphrases retain independent source-grounded chunk results. --reduce-keyphrases preserves the complete candidate union and applies global ranking once. Summaries default to an exact cited-bullet union; --synthesize-summary and --hierarchical-summary select the existing explicit neural pipelines. Answer defaults to question-aware chunk answers; --synthesize-answer and --hierarchical-answer select their evidence-based synthesis pipelines. Every selected pass belongs to one atomic document result. --cohort-chunks shares native layers within each document. Independently, --cohort-hierarchy-groups shares native layers across independent groups in each explicitly enabled hierarchy level. All five native work flags bound the ENTIRE stream. Separate --max-run-mask-node-visits, --max-run-source-* and reducer scan ceilings remain nonrenewable across documents, failures and flush controls. All allowed synthesis passes are reserved before discovery. Completed events include public candidate provenance and the selected mode/schedules. Require run_complete AND exit 0 for whole-run success; earlier completed documents remain valid after later failure. Source conversion and independently pinned snapshots use the same route. Optional snapshot saving follows fixed task/model/envelope admission and may remain after a later row fails. Hierarchies are explicitly lossy; source membership does not establish semantic entailment. No retained job, input spool, per-row options/question overrides, selected-row projection, prompt prefill, cross-document cohorts, sampling or repair retry. Blocking IO observes one end-to-end deadline cooperatively.")
                .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; each text is a complete original document"))
                .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded UTF-8 bytes of each COMPLETE original document"))
                .mut_arg("max_map_result_bytes", |arg| arg.help("Complete semantic document result cap; stream/provenance framing is additional"))
                .mut_arg("max_total_mask_node_visits", |arg| arg.help("Complete-document mask cap; --max-run-mask-node-visits separately bounds the stream"))
                .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoints for the ENTIRE stream"))
                .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream native forward ceiling across all documents and passes"))
                .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream full-vocabulary projection ceiling"))
                .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention-pair ceiling"))
                .mut_arg("max_dot_products", |arg| arg.help("Whole-stream grouped native dot-product ceiling"))
                .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling")))
        }).collect()
}

#[derive(Clone, Copy, Debug)]
pub(in crate::candidate_cli::int4) struct CorpusLimits {
    common: Int4Limits,
    document: MapLimits,
    transport: TransportLimits,
    run: Int4SourceDocumentBatchLimits,
    execution_bytes: u64,
    retained_defaults_bytes: u64,
    output_bytes: u64,
    io_bytes: u64,
    cli_preparation_bytes: u64,
}

impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(kind: Kind, matches: &clap::ArgMatches)
        -> Result<Self, clap::Error> {
        Ok(Self { document: MapArgs::from_matches(kind, matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli::int4) fn validate(&self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        let document = self.document.validate_corpus(args)?;
        let mut common = document.common;
        let s = self.stream;
        let verification = GroundingBudget { max_fields: s.max_run_source_fields,
            max_matches: s.max_run_source_evidence_spans, max_scan_steps: s.max_run_source_scan_steps };
        let per_document = self.document.document_verification(document)?;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_mask_node_visits)
            || s.max_run_mask_node_visits < document.mapping.max_mask_visits
            || !(1..=1_000_000_000).contains(&verification.max_fields)
            || !(1..=1_000_000_000).contains(&verification.max_matches)
            || !(1..=1_000_000_000_000_000_000).contains(&verification.max_scan_steps)
            || verification.max_fields < per_document.max_fields || verification.max_matches < per_document.max_matches
            || verification.max_scan_steps < per_document.max_scan_steps
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_summary_scan_steps)
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_keyphrase_scan_work)
            || s.max_run_summary_scan_steps < document.summary.map_or(0, |limits| limits.aggregation.max_scan_steps)
            || s.max_run_keyphrase_scan_work < document.keyphrases.map_or(0, |limits| limits.aggregation.max_scan_work) {
            return Err(CandidateError::Arguments);
        }
        let run = Int4SourceDocumentBatchLimits { max_model_work: document.mapping.max_model_work,
            max_mask_visits: s.max_run_mask_node_visits, max_verification: verification,
            max_summary_scan_steps: s.max_run_summary_scan_steps, max_keyphrase_scan_work: s.max_run_keyphrase_scan_work };
        let framing = s.max_requests.checked_add(4).and_then(|count| count.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: document.max_result_bytes.checked_add(FRAME_ALLOWANCE as usize)
                .ok_or(CandidateError::Arguments)?,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: run.max_model_work.forward_positions,
                projected_logits: run.max_model_work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE, 2 * IO_BUFFER_BYTES as u64])?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes, io_bytes])?;
        // Price retained private defaults before opening any settings file.
        // Serde's growing vector and the bounded question reader may each
        // retain capacity beyond their used length. The exact factory quote
        // is checked against this bound before model acquisition.
        let retained_defaults_bytes = match self.document.kind {
            Kind::Ner => times(shared::OPTIONS_BYTES as u64,
                2 * std::mem::size_of::<crate::tasks::ner::EntityType>() as u64)?,
            Kind::Answer => times(question::QUESTION_BYTES as u64 + 1, 2)?,
            _ => 0,
        };
        let input_bytes = sum(&[times(transport.max_line_bytes as u64, 8)?,
            times(transport.max_output_line_bytes as u64, 4)?, times(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, common.preparation_bytes, io_bytes,
            document.mapping.chunks.max_input_bytes as u64, times(document.mapping.chunks.max_chunks as u64, 32768)?,
            retained_defaults_bytes, document.hierarchy_preparation_bytes])?;
        let passes = document.hierarchy.map(|limits| limits.max_passes as u64)
            .or(document.question_hierarchy.map(|limits| limits.max_passes as u64))
            .unwrap_or(u64::from(document.synthesis.is_some() || document.question_synthesis.is_some()));
        let tokens = if document.summary.is_some() || document.keyphrases.is_some() { 0 }
            else if self.document.kind == Kind::Answer {
                // QA discovery and flat synthesis retain text/citations;
                // only explicit hierarchy passes retain token transcripts.
                times(document.question_hierarchy.map_or(0, |limits| limits.max_passes as u64),
                    args.max_new_tokens as u64)?
            }
            else { times(sum(&[document.mapping.chunks.max_chunks as u64, passes])?, args.max_new_tokens as u64)? };
        let output_reservation_bytes = sum(&[times(document.max_result_bytes as u64, 4)?, times(tokens, 8)?, 4096])?;
        let native = document.cohort.map_or(common.native, |quote| quote.native);
        let execution_bytes = sum(&[native.kv_bytes, native.rope_bytes, native.scratch_payload_bound,
            ALLOCATOR_BYTES, document.cohort.map_or(0, |quote| quote.workspace_bytes),
            input_bytes, document.workspace_bytes, output_reservation_bytes])?;
        common.minimum_memory_bytes = sum(&[grouped_memory_floor(args, common, execution_bytes)?, io_bytes])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, document, transport, run, execution_bytes, retained_defaults_bytes,
            output_bytes: s.max_total_output_bytes, io_bytes, cli_preparation_bytes })
    }

    fn mode_name(&self, limits: MapLimits) -> &'static str {
        if limits.question_hierarchy.is_some() { "question_hierarchy" }
        else if limits.question_synthesis.is_some() { "question_synthesis" }
        else if self.document.kind == Kind::Answer { "question_discovery" }
        else if limits.hierarchy.is_some() { "summary_hierarchy" }
        else if limits.synthesis.is_some() { "summary_synthesis" }
        else if limits.summary.is_some() { "summary_selection" }
        else if limits.keyphrases.is_some() { "keyphrase_ranking" }
        else { "source_map" }
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
        let task = self.document.kind.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let message = match error {
                    CandidateError::Timeout => "cooperative corpus deadline expired; earlier completed frames may exist",
                    CandidateError::Execution => "native corpus execution failed or was cancelled; earlier completed frames may exist",
                    _ => error.message(),
                };
                let _ = writeln!(diagnostics, "fnlp candidate source-corpus {task}: {message}");
                error.exit_code()
            }
        }
    }
}
