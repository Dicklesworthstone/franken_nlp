//! Complete-document exclusive and independent multi-label classification.
use super::*;
use crate::{native_engine::strict_int4::cohort::MAX_INT4_SEQUENCES,
    tasks::{ir::TaskBudget, mapreduce::ChunkLimits,
    classify::{ClassificationAggregationPolicy, ClassificationLimits, ClassificationMode, ClassificationRequest,
        grouped_int4::long::{Int4ClassificationMapLimits,
            memory::{Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement}}}}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
pub(super) mod corpus;
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
    /// Complete semantic document result; framing has a separate fixed allowance.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Complete retained native chunk results, never renewed per chunk.
    #[arg(long, default_value_t = 67_108_864)]
    max_retained_result_bytes: u64,
    /// Required original-scalar coverage by classified chunks; 1000000 requires all.
    /// Multi-label classification applies this threshold independently to every label.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_classified_coverage_ppm: u32,
    /// Execute up to N complete chunks together; N times the actual label heads must be <=64.
    #[arg(long, value_name = "N")]
    cohort_chunks: Option<usize>,
}

pub(super) fn definition() -> clap::Command {
    MappingArgs::augment_args(classify::definition().name("classify-map")
        .about("Classify every chunk of a complete document with explicit aggregate decisions")
        .long_about("Read the same strict {document, labels, mode?, policy?} JSON request as classify, then losslessly partition the complete original source using the pinned encoder and every label head's exact prompt capacity. Preflight all chunks and all five whole-document native work ceilings before the first forward. Preserve complete per-chunk full-vocabulary sequence scores and byte/scalar lineage. A separately named, uncalibrated reduction weights candidate distributions by original Unicode scalar counts; it is not a full-context model probability. Exclusive mode mixes the finite label choices; multi_label mode retains independent yes/no decisions and coverage for each label. Abstained chunks remain in descriptive weights, and aggregate decisions require the explicit classified-coverage threshold plus the existing peak/margin policy. Serial execution is the default; --cohort-chunks explicitly shares native decoder layers across complete independent chunks. The requested chunk count times actual heads must be at most 64: exclusive mode has one head, multi_label mode has one head per label. Invalid geometry refuses before model IO, never silently clamping or falling back. All-chunk plans, native workspaces, retained results and complete publication are memory-admitted before input IO. Every future cohort window is preflighted before the first native allocation or an optional snapshot save. Source conversion and independently pinned snapshots share this route. A failed chunk publishes no partial document. No sampling, prompt overrides, thinking, tools, --prefill-rows or --cohort-heads. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("One complete typed classification JSON object; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Maximum bytes of the complete JSON request, including source, labels and policy"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native result per chunk; --max-map-result-bytes bounds the document"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("262144")
            .help("Whole-document native forward ceiling across every chunk and label head"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("1000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for the complete hosted plan, every chunk and reduction")))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MapLimits {
    pub common: Int4Limits,
    pub(super) mapping: Int4ClassificationMapLimits,
    pub(super) planning: ClassificationLimits,
    pub(super) memory: Int4ClassificationMapMemoryRequirement,
    pub(super) cli_preparation_bytes: u64,
    envelope_bytes: usize,
}
impl MapArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<MapLimits, CandidateError> {
        self.validate_with_max_labels(args, 128)
    }
    /// A live corpus may narrow label authority for every default and override.
    /// Existing one-off and durable workflows keep their original ceiling.
    pub(super) fn validate_with_max_labels(self, args: &Int4Args, max_labels: usize)
        -> Result<MapLimits, CandidateError> {
        if !(1..=128).contains(&max_labels) { return Err(CandidateError::Arguments); }
        let mut common = self.scoring.validate(args)?;
        let m = self.mapping;
        if args.max_input_bytes < 4 || !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|bytes| !(4..=MAX_INPUT_BYTES).contains(&bytes))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
            || m.max_map_result_bytes > m.max_retained_result_bytes
            || common.result_bytes as u64 > m.max_retained_result_bytes
            || m.cohort_chunks.is_some_and(|chunks| !(1..=MAX_INT4_SEQUENCES).contains(&chunks)
                || chunks > m.max_chunks) {
            return Err(CandidateError::Arguments);
        }
        let aggregation = ClassificationAggregationPolicy {
            minimum_classified_coverage_ppm: m.minimum_classified_coverage_ppm };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let mut planning = self.scoring.classification_limits(args);
        planning.max_labels = max_labels;
        let budget = self.budget(common, planning.max_labels)?;
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.max_input_bytes),
            max_chunk_tokens: common.max_prompt_tokens,
            context_tokens: args.context_tokens, reserved_tokens: budget.max_output_tokens as usize,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = Int4ClassificationMapLimits { chunks, max_model_work: self.scoring.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes, aggregation };
        // Keep the one-context basis independent of the physical schedule.
        // Unknown input prices the largest valid label/cohort geometry; the
        // actual product is checked separately after strict request parsing.
        let quote = Int4ClassificationMapMemoryLimits { context_tokens: args.context_tokens,
            preparation_reserve_bytes: common.preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES };
        let memory = if let Some(chunks) = m.cohort_chunks {
            let heads = planning.max_labels.min(self.cohort_head_limit()?);
            quote.cohort_requirements(args.group_width, budget, planning, mapping, heads, chunks)
                .map_err(|_| CandidateError::Arguments)?.memory
        } else {
            quote.requirements(args.group_width, budget, planning, mapping)
                .map_err(|_| CandidateError::Arguments)?
        };
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
    fn cohort_head_limit(self) -> Result<usize, CandidateError> {
        match self.mapping.cohort_chunks {
            None => Ok(1),
            Some(chunks) if (1..=MAX_INT4_SEQUENCES).contains(&chunks) => Ok(MAX_INT4_SEQUENCES / chunks),
            _ => Err(CandidateError::Arguments),
        }
    }
    pub(super) fn budget(self, common: Int4Limits, max_heads: usize) -> Result<TaskBudget, CandidateError> {
        let mut budget = self.scoring.budget(common);
        // Freeze logical task authority for all declared independent heads.
        // The physical serial memory quote still reserves just one cache;
        // changing an admitted schedule must not change the task identity.
        budget.max_kv_bytes = common.native.kv_bytes.checked_mul(max_heads as u64)
            .ok_or(CandidateError::Arguments)?;
        Ok(budget)
    }
    fn admit_request(self, request: &ClassificationRequest) -> Result<(), CandidateError> {
        self.admit_labels(request.labels.len(), request.mode)
    }
    pub(super) fn cohort_chunks(self) -> Option<usize> { self.mapping.cohort_chunks }
    pub(super) fn admit_labels(self, labels: usize, mode: ClassificationMode) -> Result<(), CandidateError> {
        let heads = if mode == ClassificationMode::Exclusive { 1 } else { labels };
        if labels == 0 || labels > 128
            || (mode == ClassificationMode::Exclusive && labels < 2)
            || self.mapping.cohort_chunks.is_some_and(|chunks|
                chunks.checked_mul(heads).is_none_or(|count| count == 0 || count > MAX_INT4_SEQUENCES)) {
            return Err(CandidateError::Planning);
        }
        Ok(())
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
