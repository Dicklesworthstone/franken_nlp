//! One long original document, losslessly partitioned before native map/merge.
//! No linguistic-boundary, global synthesis, global ranking or recall claims.
use super::*;
use serde::de::DeserializeOwned;
use crate::{native_engine::{portable_int8::ProjectionWork, strict_int8::Int8Work},
    tasks::{BuiltInTask, mapreduce::{ChunkLimits, ExecutionLimits},
        ner::NerOptions, keyphrases::KeyphraseOptions, summarize::SummaryOptions,
        source_planning::quantized::{capacity::Int8SourceMapCapacity,
            long::{Int8SourceMapLimits, SourceMapTask}}}};

const MAX_MAP_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKS: usize = 256;

#[derive(Args)]
pub(crate) struct MapCommand {
    /// One exact UTF-8 document, not NDJSON; '-' reads stdin.
    #[arg(default_value = "-")]
    pub(super) input: PathBuf,
    #[arg(long, value_parser = ["ner", "keyphrases", "summarize"])]
    pub(super) task: String,
    /// Shared context/model resources. Output-token and result limits apply
    /// to each chunk; input bytes apply to the complete original document.
    #[command(flatten)]
    pub(super) host: source::SourceHostArgs,
    /// Complete typed options for the selected task; never prompt instructions.
    #[arg(long, value_name = "FILE")]
    pub(super) options: Option<PathBuf>,
    /// Fail rather than silently truncating the document at this count.
    #[arg(long, default_value_t = 64)]
    pub(super) max_chunks: usize,
    /// Optional additional byte ceiling; actual context fit is derived from
    /// the pinned task/template/schema, then checked by the real source encoder.
    #[arg(long)]
    pub(super) max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    pub(super) max_tokenizer_calls: usize,
    /// Complete map/merge result, not a per-chunk result ceiling.
    #[arg(long, default_value_t = 16_777_216)]
    pub(super) max_map_result_bytes: usize,
    #[arg(long, default_value_t = 33_554_432)]
    pub(super) max_live_value_bytes: usize,
    #[arg(long, default_value_t = 67_108_864)]
    pub(super) max_total_value_bytes: usize,
    /// Additional modeled reduction/allocator headroom, not an RSS guarantee.
    #[arg(long, default_value_t = 64)]
    pub(super) reduction_reserve_mib: u64,
    /// Whole-document allowance; never renewed by another chunk or merge.
    #[arg(long, default_value_t = 64_000_000_000)]
    pub(super) max_total_mask_node_visits: u64,
    #[arg(long, default_value_t = 262_144)]
    pub(super) max_forward_positions: u64,
    #[arg(long, default_value_t = 10_000_000_000)]
    pub(super) max_projected_logits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    pub(super) max_attention_pairs: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    pub(super) max_dot_products: u64,
    #[arg(long, default_value_t = 10_000_000_000_000_000)]
    pub(super) max_multiply_accumulates: u64,
}

pub(super) fn definition() -> clap::Command {
    MapCommand::augment_args(clap::Command::new("map")
        .about("Process a long document with source-aligned, independent native chunk results")
        .long_about("Losslessly partition one UTF-8 document using the actual pinned task scaffold and source encoder, then run NER, keyphrases or cited summaries on one resident candidate model. Output retains ordered independent chunk results and original-document coordinates. This is not a global synthesized summary, global entity census, global ranking, overlapping-window analysis or single-context-equivalence claim. No partial success is published."))
}

impl MapCommand {
    pub(super) fn kind(&self) -> Result<BuiltInTask, CandidateError> {
        match self.task.as_str() { "ner" => Ok(BuiltInTask::Ner), "keyphrases" => Ok(BuiltInTask::Keyphrases),
            "summarize" => Ok(BuiltInTask::Summarize), _ => Err(CandidateError::Arguments) }
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        self.kind()?;
        let (args, limits) = self.host.common(self.input.clone())?;
        if self.host.max_input_bytes < 4 || !(1..=MAX_CHUNKS).contains(&self.max_chunks)
            || !(1..=1_000_000).contains(&self.max_tokenizer_calls)
            || self.max_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || [self.max_map_result_bytes, self.max_live_value_bytes, self.max_total_value_bytes]
                .iter().any(|&n| !(1..=MAX_MAP_BYTES).contains(&n))
            || self.max_map_result_bytes > self.max_live_value_bytes
            || self.host.max_result_bytes > self.max_map_result_bytes
            || self.max_map_result_bytes > self.max_total_value_bytes
            || self.reduction_reserve_mib == 0
            || self.max_total_mask_node_visits < self.host.max_mask_node_visits
            || self.max_total_mask_node_visits > 1_000_000_000_000_000
            || self.max_forward_positions == 0 || self.max_forward_positions > 16 * 1024 * 1024
            || self.max_projected_logits == 0 || self.max_attention_pairs == 0
            || self.max_dot_products == 0 || self.max_multiply_accumulates == 0
            || self.options.as_ref().is_some_and(|p| p.as_os_str().is_empty() || p.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        let reduction = self.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        if reduction > limits.memory_bytes { return Err(CandidateError::Arguments); }
        // The host retains ALL prepared prompts/grammars before any forward.
        // Price that aggregate, plus complete-result staging, not one chunk's
        // tokenizer and a byte count masquerading as a whole-document charge.
        let per_chunk = (self.host.context_tokens as u64).checked_mul(32)
            .and_then(|n| n.checked_add(u64::from(self.host.max_grammar_states).checked_mul(512)?))
            .ok_or(CandidateError::Arguments)?;
        let floor = per_chunk.checked_mul(self.max_chunks as u64)
            .and_then(|n| n.checked_add(256 * MIB))
            .and_then(|n| n.checked_add(self.host.max_input_bytes as u64 * 64))
            .and_then(|n| n.checked_add((self.max_map_result_bytes as u64 + 4096) * 2))
            .ok_or(CandidateError::Arguments)?;
        if limits.preparation_bytes < floor { return Err(CandidateError::Arguments); }
        Ok((args, limits))
    }
    pub(super) fn map_task(&self, raw: Option<&str>) -> Result<SourceMapTask, CandidateError> {
        let task = match self.kind()? {
            BuiltInTask::Ner => {
                let options = options::<NerOptions>(raw)?;
                options.schema_source().map_err(|_| CandidateError::Planning)?;
                SourceMapTask::Ner(options)
            }
            BuiltInTask::Keyphrases => {
                let options = options::<KeyphraseOptions>(raw)?;
                options.schema_source().map_err(|_| CandidateError::Planning)?;
                SourceMapTask::Keyphrases(options)
            }
            BuiltInTask::Summarize => {
                let options = options::<SummaryOptions>(raw)?;
                options.schema_source().map_err(|_| CandidateError::Planning)?;
                SourceMapTask::Summarize(options)
            }
            _ => return Err(CandidateError::Arguments),
        };
        Ok(task)
    }
    pub(super) fn work_ceiling(&self) -> Int8Work {
        Int8Work { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits,
            attention_pairs: self.max_attention_pairs, projections: ProjectionWork {
                dot_products: self.max_dot_products, multiply_accumulates: self.max_multiply_accumulates } }
    }
    pub(super) fn admit_work(&self, work: Int8Work, masks: u64) -> Result<(), CandidateError> {
        let max = self.work_ceiling();
        if work.forward_positions > max.forward_positions || work.projected_logits > max.projected_logits
            || work.attention_pairs > max.attention_pairs || !work.projections.fits(max.projections)
            || masks > self.max_total_mask_node_visits { return Err(CandidateError::Planning); }
        Ok(())
    }
    pub(super) fn mapping(&self, capacity: Int8SourceMapCapacity) -> Result<Int8SourceMapLimits, CandidateError> {
        // Four bytes leave room for one UTF-8 scalar. The actual source count
        // still decides fit, and ChunkPlan preserves scalar and CRLF boundaries.
        if capacity.max_source_tokens() < 4 { return Err(CandidateError::Planning); }
        let bytes = self.max_chunk_bytes.unwrap_or(capacity.max_source_tokens());
        let chunks = capacity.constrain_chunks(ChunkLimits {
            max_input_bytes: self.host.max_input_bytes,
            max_chunk_bytes: bytes.min(self.host.max_input_bytes.max(4)),
            max_chunk_tokens: capacity.max_source_tokens(), context_tokens: self.host.context_tokens,
            reserved_tokens: capacity.reserved_tokens(), max_chunks: self.max_chunks,
            max_tokenizer_calls: self.max_tokenizer_calls,
        }).map_err(|_| CandidateError::Planning)?;
        Ok(Int8SourceMapLimits { chunks,
            reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8,
                max_reduction_levels: 8, max_task_calls: 1024,
                max_value_bytes: self.max_map_result_bytes, max_live_value_bytes: self.max_live_value_bytes,
                max_total_value_bytes: self.max_total_value_bytes, max_result_bytes: self.max_map_result_bytes },
            max_model_work: self.work_ceiling(), mask_limits: self.host.masks(),
            mask_visits_per_chunk: self.host.max_mask_node_visits, max_mask_visits: self.max_total_mask_node_visits })
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (args, limits) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::map_tasks::execute(self, args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

fn options<T: DeserializeOwned + Default>(raw: Option<&str>) -> Result<T, CandidateError> {
    let Some(raw) = raw else { return Ok(T::default()); };
    if raw.len() > source::OPTIONS_BYTES { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(raw, canonjson::ParseLimits {
        max_depth: 8, max_string_bytes: source::OPTIONS_BYTES,
    }).map_err(|_| CandidateError::Input)?;
    serde_json::from_value(value).map_err(|_| CandidateError::Input)
}

#[cfg(test)] pub(super) mod tests;
