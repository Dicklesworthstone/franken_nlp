//! Complete-document source tasks with finite grouped-INT4 map/reduction work.
use super::*;
use crate::{
    candidate_cli::source::{self as shared, Kind},
    corpus::summarize::CorpusSummaryLimits,
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::ProjectionWork, strict_int4::Int4Work},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourceTaskRequest, grouped_int4::long::{Int4SourceMapLimits,
            SourceMapTask, summary::Int4SummaryLimits}}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan},
};

mod question;
mod synthesis;
#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MapArgs {
    pub kind: Kind,
    pub source: source::SourceArgs,
    mapping: MappingArgs,
    summary: Option<SummaryArgs>,
    question: Option<question::QuestionArgs>,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct MappingArgs {
    /// Refuse a document requiring more chunks; never truncate its source.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional byte ceiling; exact typed task/template context fit also applies.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete document result, separate from each chunk's --max-result-bytes.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_live_value_bytes: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_value_bytes: usize,
    /// Modeled reduction/evidence allocation headroom, not an RSS guarantee.
    #[arg(long, default_value_t = 64)]
    reduction_reserve_mib: u64,
    /// Nonrenewable complete-document mask work, across every chunk.
    #[arg(long, default_value_t = 128_000_000_000)]
    max_total_mask_node_visits: u64,
    #[arg(long, default_value_t = 262_144)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 10_000_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_attention_pairs: u64,
    /// Actual per-group native dot products.
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_multiply_accumulates: u64,
    /// Independently verified source fields across the complete document.
    #[arg(long, default_value_t = 4096)]
    max_source_fields: usize,
    /// Original-coordinate occurrence spans across every chunk.
    #[arg(long, default_value_t = 16384)]
    max_source_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_source_scan_steps: u64,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct SummaryArgs {
    /// Final global bullet cap, applied once after all candidates are merged.
    /// Per-chunk summary options remain controlled by --options.
    #[arg(long, default_value_t = 12, conflicts_with = "synthesize_summary")]
    max_summary_bullets: usize,
    #[arg(long, default_value_t = 4096, conflicts_with = "synthesize_summary")]
    max_summary_unique_bullets: usize,
    #[arg(long, default_value_t = 16384, conflicts_with = "synthesize_summary")]
    max_summary_citations: usize,
    #[arg(long, default_value_t = 65536, conflicts_with = "synthesize_summary")]
    max_summary_evidence_spans: usize,
    /// Both canonical and conservatively charged stored aggregate values.
    #[arg(long, default_value_t = 4_194_304, conflicts_with = "synthesize_summary")]
    max_summary_value_bytes: usize,
    /// Independent reducer reverification work, never reset between chunks.
    #[arg(long, default_value_t = 536_870_912, conflicts_with = "synthesize_summary")]
    max_summary_scan_steps: u64,
    #[command(flatten)]
    synthesis: synthesis::SynthesisArgs,
}
pub(super) fn definitions() -> Vec<clap::Command> {
    [(Kind::Ner, "ner-map"), (Kind::Keyphrases, "keyphrases-map"),
        (Kind::Summarize, "summarize-map"), (Kind::Answer, "answer-map")]
        .into_iter().map(|(kind, name)| {
            let about = match kind {
                Kind::Ner => "Find entities throughout a complete document with original-coordinate source evidence",
                Kind::Keyphrases => "Extract keyphrases from every chunk of a complete document",
                Kind::Summarize => "Summarize every chunk with an exact cited union or explicit neural synthesis",
                Kind::Answer => "Ask one question of every chunk and preserve all original-source evidence",
            };
            let command = MappingArgs::augment_args(source::definition(kind).name(name).about(about)
                .long_about("Losslessly partition one complete UTF-8 document with the pinned source encoder and the exact typed task/schema/template capacity. Preflight and admit ALL chunks before the first native forward. NER and keyphrases retain independent chunk results and exact original-document occurrence spans; local keyphrase ranks are not a global ranking. summarize-map returns one deterministic exact-bullet/evidence union, preserving every candidate through intermediate reduction and applying --max-summary-bullets only at the complete final root. Citation membership does not prove semantic entailment; there is no neural hierarchical synthesis or contradiction reconciliation. --options supplies the selected task's complete per-chunk options. Native work, mask visits, source verification and reducer verification never renew per chunk. A failed chunk or insufficient budget produces no partial document result. Source conversion and digest-pinned snapshot reload share the same checked path. Blocking IO observes the end-to-end deadline cooperatively.")
                .mut_arg("input", |arg| arg.help("One complete UTF-8 document; '-' reads stdin; not NDJSON"))
                .mut_arg("max_input_bytes", |arg| arg.default_value("1048576").help("Maximum bytes of the COMPLETE original document"))
                .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
                .mut_arg("max_mask_node_visits", |arg| arg.help("Mask allowance per chunk; --max-total-mask-node-visits bounds the whole document"))
                .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for hosted whole-document preparation, maps and reduction")));
            match kind {
                Kind::Summarize => SummaryArgs::augment_args(command.long_about("Summarize every chunk of one complete UTF-8 document. The default keeps an exact cited bullet union and applies --max-summary-bullets only at the final root. --synthesize-summary instead adds one native pass over ALL independently verified verbatim source quotes from the complete map. Generated map bullets never become factual source evidence; no quote is ranked away to fit. The complete evidence must fit its byte, segment, token and context ceilings or the command returns no result. Every final citation lifts back to all evidence-reachable original-document byte and Unicode-scalar occurrences. Discovery plus synthesis shares one five-axis native work ceiling, mask allowance and independent verification budget. Structural source membership does not establish entailment, full-document equivalence or recall. Source conversion and digest-pinned snapshot reload share this checked path; optional snapshot save follows complete-map and reserved-synthesis preflight, and survives later evidence-fit or task failure. --options remains the complete per-map SummaryOptions JSON; --synthesis-bullets optionally changes only the final neural bullet cap. No hierarchical compression or contradiction reconciliation is implied.")),
                Kind::Answer => question::definition(command),
                _ => command,
            }
        }).collect()
}
#[derive(Clone, Copy, Debug)]
pub(super) struct MapLimits {
    pub common: Int4Limits,
    pub mapping: Int4SourceMapLimits,
    pub summary: Option<Int4SummaryLimits>,
    pub synthesis: Option<crate::tasks::source_planning::grouped_int4::long::summary::synthesis::SummarySynthesisLimits>,
    pub question_synthesis: Option<crate::tasks::source_planning::grouped_int4::long::question::synthesis::QuestionSynthesisLimits>,
    pub reduction_reserve_bytes: u64,
    pub max_result_bytes: usize,
    pub workspace_bytes: u64,
}
impl MapArgs {
    fn planning(&self, args: &Int4Args) -> crate::tasks::source_planning::SourcePlanningLimits {
        let planning = self.source.planning(args);
        self.question.as_ref().map_or(planning, |question| question.planning(planning))
    }
    pub(super) fn from_matches(kind: Kind, matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { kind, source: source::SourceArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?,
            summary: if kind == Kind::Summarize { Some(SummaryArgs::from_arg_matches(matches)?) } else { None },
            question: if kind == Kind::Answer { Some(question::QuestionArgs::from_arg_matches(matches)?) } else { None } })
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        let m = self.mapping;
        let mut common = self.source.validate(args)?;
        if (self.kind == Kind::Answer) != self.question.is_some()
            || (self.kind == Kind::Summarize) != self.summary.is_some()
            || args.max_input_bytes < 4 || !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || [m.max_map_result_bytes, m.max_live_value_bytes, m.max_total_value_bytes]
                .iter().any(|&n| !(1..=64 * 1024 * 1024).contains(&n))
            || m.max_map_result_bytes > m.max_live_value_bytes || m.max_map_result_bytes > m.max_total_value_bytes
            || common.result_bytes > m.max_map_result_bytes
            || m.max_total_mask_node_visits < self.source.mask_visits()
            || m.max_total_mask_node_visits > 1_000_000_000_000_000
            || !(1..=16 * 1024 * 1024).contains(&m.max_forward_positions)
            || m.max_projected_logits < NANBEIGE_VOCAB_SIZE as u64 || m.max_attention_pairs == 0
            || m.max_dot_products == 0 || m.max_multiply_accumulates == 0
            || !(1..=1_000_000).contains(&m.max_source_fields)
            || !(1..=1_000_000).contains(&m.max_source_evidence_spans)
            || !(1..=1_000_000_000_000).contains(&m.max_source_scan_steps) {
            return Err(CandidateError::Arguments);
        }
        let reduction_reserve_bytes = mib(m.reduction_reserve_mib)?;
        let mapping = self.mapping_limits(args)?;
        if let Some(question) = &self.question { question.validate()?; }
        let synthesis = match self.summary {
            Some(s) => s.synthesis.limits(self.planning(args))?,
            None => None,
        };
        let question_synthesis = match &self.question {
            Some(question) => question.synthesis_limits(self.planning(args))?,
            None => None,
        };
        let summary = self.summary.filter(|s| !s.synthesis.synthesize_summary)
            .map(|s| s.limits(mapping)).transpose()?;
        if let Some(summary) = summary {
            // In addition to the host's full map/frontier payload, price the
            // independent summary verifier and aggregate owners explicitly.
            let aggregation = summary.aggregation;
            let floor = sum(&[times(aggregation.max_value_bytes as u64, 8)?,
                times(aggregation.max_unique_bullets as u64, 512)?,
                times(aggregation.max_citations as u64, 512)?,
                times(times(aggregation.max_evidence_spans as u64,
                    std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?])?;
            if reduction_reserve_bytes < floor { return Err(CandidateError::Arguments); }
        }
        let budget = self.source.budget(args, common);
        // Retain every planned prompt, grammar and evidence configuration
        // until complete execution finishes. Actual host planning is separate.
        let per_chunk = sum(&[args.context_tokens as u64 * 32,
            u64::from(budget.max_grammar_states) * 512, shared::OPTIONS_BYTES as u64 * 2])?;
        let question_preparation = if self.question.is_some() {
            times(question::QUESTION_BYTES as u64,
                (m.max_chunks as u64 + 1 + u64::from(question_synthesis.is_some())) * 4)?
        } else { 0 };
        let preparation_passes = m.max_chunks as u64
            + u64::from(synthesis.is_some() || question_synthesis.is_some());
        let preparation_floor = sum(&[times(per_chunk, preparation_passes)?, 256 * MIB,
            args.max_input_bytes as u64 * 64, (m.max_map_result_bytes as u64 + 4096) * 2,
            question_preparation])?;
        if common.preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        let workspace_bytes = match &self.question {
            Some(question) => match question_synthesis {
                Some(synthesis) => question::synthesis::workspace_bytes(budget, mapping, m.max_chunks,
                    reduction_reserve_bytes, question.verification()?, synthesis)?,
                None => question::workspace_bytes(budget, mapping, m.max_chunks,
                    reduction_reserve_bytes, question.verification()?)?,
            },
            None => match synthesis {
                Some(synthesis) => synthesis::workspace_bytes(budget, mapping, m.max_chunks,
                    reduction_reserve_bytes, synthesis)?,
                None => workspace_bytes(budget, mapping, m.max_chunks, reduction_reserve_bytes)?,
            },
        };
        let tokens = if summary.is_some() { 0 }
            else { times(preparation_passes, u64::from(budget.max_output_tokens))? };
        let output = sum(&[m.max_map_result_bytes as u64 * 4, times(tokens, 8)?, 4096])?;
        let question_input = if self.question.is_some() { question::QUESTION_BYTES as u64 + 1 } else { 0 };
        let input = sum(&[args.max_input_bytes as u64 + 1, question_input, common.preparation_bytes])?;
        // CLI preparation stays alive through publication while the host owns
        // a separately charged source/preparation capture and complete output.
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, input, workspace_bytes, output])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, mapping, summary, synthesis, question_synthesis, reduction_reserve_bytes,
            max_result_bytes: m.max_map_result_bytes, workspace_bytes })
    }
    fn mapping_limits(&self, args: &Int4Args) -> Result<Int4SourceMapLimits, CandidateError> {
        let m = self.mapping;
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.context_tokens).min(args.max_input_bytes),
            max_chunk_tokens: args.context_tokens, context_tokens: args.context_tokens,
            reserved_tokens: args.max_new_tokens, max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        // This preliminary ceiling is tightened by the core's EXACT scaffold
        // capacity, independently for the selected task and typed options.
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        Ok(Int4SourceMapLimits {
            chunks, reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8,
                max_reduction_levels: 8, max_task_calls: 1024, max_value_bytes: m.max_map_result_bytes,
                max_live_value_bytes: m.max_live_value_bytes, max_total_value_bytes: m.max_total_value_bytes,
                max_result_bytes: m.max_map_result_bytes },
            max_model_work: Int4Work { forward_positions: m.max_forward_positions,
                projected_logits: m.max_projected_logits, attention_pairs: m.max_attention_pairs,
                projections: ProjectionWork { dot_products: m.max_dot_products,
                    multiply_accumulates: m.max_multiply_accumulates } },
            mask_limits: self.source.masks(), mask_visits_per_chunk: self.source.mask_visits(),
            max_mask_visits: m.max_total_mask_node_visits,
            verification: GroundingBudget { max_fields: m.max_source_fields,
                max_matches: m.max_source_evidence_spans, max_scan_steps: m.max_source_scan_steps },
        })
    }
    fn request(&self, text: String, options: Option<&str>, budget: TaskBudget, max_input_bytes: usize)
        -> Result<(String, SourceMapTask), CandidateError> {
        if text.is_empty() || self.kind == Kind::Answer { return Err(CandidateError::Input); }
        // Use the existing rejecting typed-options parser on the actual source.
        // No placeholder input, normalization or whole-document clone.
        match shared::request(self.kind, text, options, budget, max_input_bytes)? {
            SourceTaskRequest::Ner { document, options, .. } => Ok((document, SourceMapTask::Ner(options))),
            SourceTaskRequest::Keyphrases { document, options, .. } => Ok((document, SourceMapTask::Keyphrases(options))),
            SourceTaskRequest::Summarize { document, options, .. } => Ok((document, SourceMapTask::Summarize(options))),
            SourceTaskRequest::Answer { .. } => Err(CandidateError::Arguments),
        }
    }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
impl SummaryArgs {
    fn limits(self, mapping: Int4SourceMapLimits) -> Result<Int4SummaryLimits, CandidateError> {
        let limits = Int4SummaryLimits {
            aggregation: CorpusSummaryLimits { max_unique_bullets: self.max_summary_unique_bullets,
                max_citations: self.max_summary_citations, max_evidence_spans: self.max_summary_evidence_spans,
                max_value_bytes: self.max_summary_value_bytes, max_scan_steps: self.max_summary_scan_steps },
            max_bullets: self.max_summary_bullets, max_result_bytes: mapping.reduction.max_result_bytes,
        };
        if self.max_summary_scan_steps > 1_000_000_000_000 { return Err(CandidateError::Arguments); }
        limits.validate(mapping).map_err(|_| CandidateError::Arguments)?;
        Ok(limits)
    }
}
fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
fn workspace_bytes(budget: TaskBudget, limits: Int4SourceMapLimits, chunks: usize, reserve: u64)
    -> Result<u64, CandidateError> {
    if chunks == 0 || chunks > limits.chunks.max_chunks || chunks > 256 || reserve == 0 {
        return Err(CandidateError::Arguments);
    }
    let native = times(budget.max_output_bytes, chunks as u64)?;
    let staged = times(sum(&[native, limits.reduction.max_live_value_bytes as u64])?, 4)?;
    let tokens = times(times(u64::from(budget.max_output_tokens), chunks as u64)?, 8)?;
    let fields = times(limits.verification.max_fields as u64, std::mem::size_of::<SourceFieldEvidence>() as u64)?;
    let spans = times(times(limits.verification.max_matches as u64,
        std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
    sum(&[staged, tokens, fields, spans, reserve])
}
