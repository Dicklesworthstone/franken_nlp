//! Complete-document exclusive and independent multi-label classification.
use super::*;
use crate::tasks::{ir::TaskBudget, mapreduce::ChunkLimits,
    classify::{ClassificationAggregationPolicy, ClassificationLimits,
        grouped_int4::long::{Int4ClassificationMapLimits,
            memory::{Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement}}}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MapArgs {
    scoring: classify::ScoringArgs,
    mapping: MappingArgs,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct MappingArgs {
    /// Refuse a document needing more chunks; never truncate the source.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional byte ceiling; every label head also checks its exact prompt fit.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete semantic document result; provenance has a separate fixed allowance.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Complete retained chunk results; never renewed per chunk or label.
    #[arg(long, default_value_t = 67_108_864)]
    max_retained_result_bytes: u64,
    /// Required original-scalar coverage by classified chunks; 1000000 requires all.
    /// Multi-label classification applies this threshold independently to every label.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_classified_coverage_ppm: u32,
}

pub(super) fn definition() -> clap::Command {
    MappingArgs::augment_args(classify::definition().name("classify-map")
        .about("Classify every chunk of a complete document and report an explicit label-weight mixture")
        .long_about("Read the same strict {document, labels, mode?, policy?} JSON object as classify. Losslessly partition the complete original source using the pinned encoder and the exact label/codebook prompt capacity. Preflight every chunk and all five whole-document native work ceilings before the first forward. Preserve complete per-chunk classifications and original byte/scalar lineage. The named aggregation weights chunks by original Unicode scalar count, includes abstained chunks in descriptive mixtures, and publishes a decision only when both the classified-coverage threshold and the request's candidate-weight/margin policy pass. Multi-label yes/no mixtures and coverage are independent for each label; labels never compete in a softmax. Aggregate weights are uncalibrated descriptive chunk mixtures, not full-context model probabilities. All plans, one reusable native cache, retained results and complete publication are memory-admitted before input IO. Source conversion and independently digest-pinned snapshots use the same route. A failed chunk publishes no partial document. Execution is serial; no sampling, thinking, tools, prompt overrides, --prefill-rows or --cohort-heads. Blocking IO observes the complete invocation deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("One complete typed classification JSON object; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Maximum bytes of the complete JSON request, including original document and labels"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native result per chunk; --max-map-result-bytes bounds the document"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("262144")
            .help("Whole-document native forward ceiling across all chunks and label heads"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("1000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for the complete hosted plan, all chunks and reduction")))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MapLimits {
    pub common: Int4Limits,
    mapping: Int4ClassificationMapLimits,
    planning: ClassificationLimits,
    memory: Int4ClassificationMapMemoryRequirement,
    cli_preparation_bytes: u64,
    envelope_bytes: usize,
}
impl MapArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        let mut common = self.scoring.validate(args)?;
        let m = self.mapping;
        if args.max_input_bytes < 4 || !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|bytes| !(4..=MAX_INPUT_BYTES).contains(&bytes))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
            || m.max_map_result_bytes > m.max_retained_result_bytes
            || common.result_bytes as u64 > m.max_retained_result_bytes {
            return Err(CandidateError::Arguments);
        }
        let aggregation = ClassificationAggregationPolicy {
            minimum_classified_coverage_ppm: m.minimum_classified_coverage_ppm };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let budget = self.budget(common);
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.max_input_bytes),
            max_chunk_tokens: common.max_prompt_tokens,
            context_tokens: args.context_tokens, reserved_tokens: budget.max_output_tokens as usize,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = Int4ClassificationMapLimits { chunks, max_model_work: self.scoring.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes, aggregation };
        let planning = self.scoring.classification_limits(args);
        let memory = Int4ClassificationMapMemoryLimits { context_tokens: args.context_tokens,
            preparation_reserve_bytes: common.preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES }
            .requirements(args.group_width, budget, planning, mapping).map_err(|_| CandidateError::Arguments)?;
        let envelope_bytes = memory.max_output_bytes.checked_add(4096)
            .and_then(|bytes| usize::try_from(bytes).ok()).ok_or(CandidateError::Arguments)?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes,
            (envelope_bytes as u64).checked_mul(2).ok_or(CandidateError::Arguments)?])?;
        let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
        let execution = sum(&[memory.native.kv_bytes, memory.input_bytes,
            memory.workspace_bytes, memory.output_reservation_bytes])?;
        common.minimum_memory_bytes = sum(&[cli_preparation_bytes, resident,
            loading.max(saving).max(execution), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(MapLimits { common, mapping, planning, memory, cli_preparation_bytes, envelope_bytes })
    }
    fn budget(self, common: Int4Limits) -> TaskBudget { self.scoring.budget(common) }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
