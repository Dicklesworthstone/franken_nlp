//! Complete-document INT4 redaction with independent original/transformed maps.
use super::*;
pub(in crate::candidate_cli::int4) mod corpus;
pub(in crate::candidate_cli::int4) mod jobs;
use crate::{
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::ProjectionWork, strict_int4::Int4Work},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits}, ner::NerOptions,
        redact::grouped_int4::long::{Int4LongRedactionConfig, INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES},
        source_planning::grouped_int4::long::{Int4SourceMapLimits,
            cohort::{Int4SourceMapCohortMemoryLimits, Int4SourceMapCohortRequirement}}},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct MappingFlags {
    /// Complete NER chunks per original/transformed stage; never truncate.
    #[arg(long, default_value_t = 64)]
    max_ner_chunks: usize,
    /// Additional byte ceiling; exact pinned prompt/schema capacity still applies.
    #[arg(long)]
    max_ner_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_ner_tokenizer_calls: usize,
    /// Complete map bytes per stage; live/cumulative bounds are 2x/4x this.
    #[arg(long, default_value_t = 16_777_216)]
    max_ner_map_bytes: usize,
    /// Whole-operation mask ceiling across BOTH stages and every chunk.
    #[arg(long, default_value_t = 256_000_000_000)]
    max_total_mask_node_visits: u64,
    #[arg(long, default_value_t = 262_144)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 10_000_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_attention_pairs: u64,
    /// Actual group-dependent dot products across BOTH stages.
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    max_multiply_accumulates: u64,
    /// Complete-operation field ceiling for EACH independent verification pool.
    #[arg(long, default_value_t = 4096)]
    max_source_fields: usize,
    #[arg(long, default_value_t = 16384)]
    max_source_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864)]
    max_source_scan_steps: u64,
}
#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct DocumentArgs {
    redaction: RedactArgs,
    mapping: MappingFlags,
    cohort_chunks: Option<usize>,
}
impl std::fmt::Debug for DocumentArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("DocumentRedactArgs") }
}
#[derive(Clone, Copy)]
pub(in crate::candidate_cli::int4) struct DocumentLimits {
    pub common: Int4Limits,
    mapping: Int4SourceMapLimits,
    edit_bytes: u64,
    workspace_bytes: u64,
    cohort: Option<Int4SourceMapCohortRequirement>,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    serial_definition().arg(clap::Arg::new("cohort_chunks").long("cohort-chunks")
        .value_parser(clap::value_parser!(usize)).value_name("N")
        .help("Run up to N independent NER chunks together in BOTH stages (1..=64); admit the full window, never shrink or fall back"))
}
// Corpus and retained jobs continue to own serial document engines. Do not
// inherit an accepted flag whose selected physical schedule they cannot run.
fn serial_definition() -> clap::Command {
    MappingFlags::augment_args(super::definition().name("redact-document")
        .about("Redact a complete document with whole-source rules and native INT4 NER chunks")
        .long_about("Losslessly partition one complete UTF-8 document for source-grounded native INT4 NER while deterministic rules scan the WHOLE text, including patterns crossing neural chunk boundaries. Preflight every original chunk before the first forward. Independently rescan and lift all NER coordinates before one transactional edit. By default re-tokenize and re-partition the actual transformed bytes, then rerun the same NER/rule union. Both stages share five native work counters, mask work and independent verification pools; no allowance renews per chunk. Serial execution is the default. Explicit --cohort-chunks runs independent NER chunks in native cohorts in both stages, admitting the full declared window including the possible transformed-text peak; it never changes per-chunk KV authority or silently reduces the selected width. The result keeps separate physical cohort counters around the unchanged semantic redaction. Any failed chunk or residual detection publishes no redacted result. NER chunk boundaries may split entities; a clean declared-union scan does not establish recall or anonymity. Same mask/placeholder/full256-HMAC key contract as redact, with commitment checked before document IO. Source conversion and independently pinned snapshots share the actual-model path. No automatic download or backend fallback. Blocking IO observes the deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("One complete UTF-8 document; '-' reads stdin unless stdin supplies the private key"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576").help("Maximum bytes of the COMPLETE original document"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Per-chunk mask allowance; --max-total-mask-node-visits covers BOTH stages"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for complete hosted planning, detection, editing and verification")))
}
impl DocumentArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { redaction: RedactArgs::from_matches(matches)?, mapping: MappingFlags::from_arg_matches(matches)?,
            cohort_chunks: matches.try_get_one::<usize>("cohort_chunks").ok().flatten().copied() })
    }
    pub(in crate::candidate_cli::int4) fn validate(&self, args: &Int4Args) -> Result<DocumentLimits, CandidateError> {
        let short = self.redaction.validate(args)?;
        let mut common = short.common;
        let m = self.mapping;
        if args.max_input_bytes < 4 || !(1..=256).contains(&m.max_ner_chunks)
            || m.max_ner_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || !(1..=1_000_000).contains(&m.max_ner_tokenizer_calls)
            || !(1..=16 * 1024 * 1024).contains(&m.max_ner_map_bytes)
            || common.result_bytes > m.max_ner_map_bytes
            || m.max_total_mask_node_visits < self.redaction.source.mask_visits()
            || m.max_total_mask_node_visits > 1_000_000_000_000_000
            || !(1..=16 * 1024 * 1024).contains(&m.max_forward_positions)
            || m.max_projected_logits < NANBEIGE_VOCAB_SIZE as u64 || m.max_attention_pairs == 0
            || m.max_dot_products == 0 || m.max_multiply_accumulates == 0
            || !(1..=1_000_000).contains(&m.max_source_fields)
            || !(1..=1_000_000).contains(&m.max_source_evidence_spans)
            || !(1..=1_000_000_000_000).contains(&m.max_source_scan_steps)
            || self.cohort_chunks.is_some_and(|chunks| !(1..=64).contains(&chunks) || chunks > m.max_ner_chunks) {
            return Err(CandidateError::Arguments);
        }
        let budget = self.redaction.source.budget(args, common);
        let per_chunk = sum(&[args.context_tokens as u64 * 32, u64::from(budget.max_grammar_states) * 512,
            crate::candidate_cli::source::OPTIONS_BYTES as u64 * 2])?;
        let preparation_floor = sum(&[times(per_chunk, m.max_ner_chunks as u64)?, 256 * MIB,
            times(args.max_input_bytes.max(common.result_bytes) as u64, 64)?,
            times(m.max_ner_map_bytes as u64 + 4096, 2)?,
            if self.cohort_chunks.is_some() { 2 * INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES } else { 0 }])?;
        if common.preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        let mapping = self.mapping_limits(args, common.result_bytes)?;
        let workspace_bytes = workspace_bytes(budget, mapping, short.edit_bytes)?;
        // Replace the short detector's temporary quote, retaining its exact
        // source capture, second preparation, secret metadata and native base.
        let short_workspace = sum(&[budget.max_output_bytes * 4,
            u64::from(budget.max_output_tokens) * 8, short.edit_bytes])?;
        common.minimum_memory_bytes = common.minimum_memory_bytes.checked_sub(short_workspace)
            .and_then(|n| n.checked_add(workspace_bytes)).ok_or(CandidateError::Arguments)?;
        let cohort = self.cohort_chunks.map(|chunks| {
            Int4SourceMapCohortMemoryLimits { max_chunks: chunks, context_tokens: args.context_tokens,
                max_output_tokens_per_chunk: u64::from(budget.max_output_tokens),
                max_result_bytes_per_chunk: budget.max_output_bytes }.requirements(args.group_width)
                .map_err(|_| CandidateError::Arguments)
        }).transpose()?;
        if let Some(required) = cohort {
            let original = sum(&[common.native.kv_bytes, common.native.rope_bytes, common.native.scratch_payload_bound])?;
            let grouped = sum(&[required.native.kv_bytes, required.native.rope_bytes, required.native.scratch_payload_bound])?;
            let additional_native = grouped.checked_sub(original).ok_or(CandidateError::Arguments)?;
            // Both document stages reuse this complete declared peak. The
            // cohort workspace already owns its transient result; only the
            // final wrapper adds to the hosted output's 4x payload charge.
            common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, additional_native,
                required.workspace_bytes, 4 * INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES])?;
        }
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(DocumentLimits { common, mapping, edit_bytes: short.edit_bytes, workspace_bytes, cohort })
    }
    fn verification(&self) -> GroundingBudget {
        GroundingBudget { max_fields: self.mapping.max_source_fields,
            max_matches: self.mapping.max_source_evidence_spans, max_scan_steps: self.mapping.max_source_scan_steps }
    }
    fn mapping_limits(&self, args: &Int4Args, result_bytes: usize) -> Result<Int4SourceMapLimits, CandidateError> {
        let m = self.mapping;
        let max_input = args.max_input_bytes.max(result_bytes);
        let chunks = ChunkLimits { max_input_bytes: max_input,
            max_chunk_bytes: m.max_ner_chunk_bytes.unwrap_or(args.context_tokens).min(max_input),
            max_chunk_tokens: args.context_tokens, context_tokens: args.context_tokens,
            reserved_tokens: args.max_new_tokens, max_chunks: m.max_ner_chunks,
            max_tokenizer_calls: m.max_ner_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        Ok(Int4SourceMapLimits { chunks,
            reduction: ExecutionLimits { map_batch_chunks: self.cohort_chunks.unwrap_or(1), reduce_fan_in: 8, max_reduction_levels: 8,
                max_task_calls: 1024, max_value_bytes: m.max_ner_map_bytes,
                max_live_value_bytes: m.max_ner_map_bytes * 2, max_total_value_bytes: m.max_ner_map_bytes * 4,
                max_result_bytes: m.max_ner_map_bytes },
            max_model_work: Int4Work { forward_positions: m.max_forward_positions,
                projected_logits: m.max_projected_logits, attention_pairs: m.max_attention_pairs,
                projections: ProjectionWork { dot_products: m.max_dot_products,
                    multiply_accumulates: m.max_multiply_accumulates } },
            mask_limits: self.redaction.source.masks(), mask_visits_per_chunk: self.redaction.source.mask_visits(),
            max_mask_visits: m.max_total_mask_node_visits, verification: self.verification() })
    }
    fn request(&self, args: &Int4Args, limits: DocumentLimits) -> RedactionRequest {
        let mut request = self.redaction.request(args, RedactLimits { common: limits.common, edit_bytes: limits.edit_bytes });
        request.grounding_budget = self.verification();
        request
    }
    fn detector(&self, args: &Int4Args, limits: DocumentLimits, ner: NerOptions) -> Int4LongRedactionConfig {
        let mut planning = self.redaction.source.planning(args);
        planning.max_input_bytes = args.max_input_bytes.max(limits.common.result_bytes);
        Int4LongRedactionConfig { ner, per_chunk: self.redaction.source.budget(args, limits.common),
            planning, mapping: limits.mapping, max_result_bytes: limits.common.result_bytes as u64 }
    }
    pub(in crate::candidate_cli::int4) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits, edit: u64) -> Result<u64, CandidateError> {
    let count = mapping.chunks.max_chunks;
    if !(1..=256).contains(&count) || edit == 0 || mapping.reduction.max_live_value_bytes == 0 {
        return Err(CandidateError::Arguments);
    }
    let native = times(budget.max_output_bytes, count as u64)?;
    let staged = times(sum(&[native, mapping.reduction.max_live_value_bytes as u64])?, 4)?;
    let tokens = times(times(u64::from(budget.max_output_tokens), count as u64)?, 8)?;
    let fields = times(mapping.verification.max_fields as u64, std::mem::size_of::<SourceFieldEvidence>() as u64)?;
    let spans = times(times(mapping.verification.max_matches as u64,
        std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
    sum(&[staged, tokens, fields, spans, edit])
}
