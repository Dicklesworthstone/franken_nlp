//! Complete-source faithfulness with an explicit bounded evidence reduction.
use super::*;
pub(super) mod corpus;
use serde::Deserialize;
use crate::{native_engine::strict_int4::cohort::MAX_INT4_SEQUENCES,
    tasks::{ir::TaskBudget, mapreduce::ChunkLimits,
        judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy, JudgeLimits, JudgeRequest,
            grouped_int4::faithfulness_map::{Int4FaithfulnessMapLimits,
                memory::{Int4FaithfulnessMapMemoryLimits, Int4FaithfulnessMapMemoryRequirement}}}}};

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
    /// Refuse a source needing more chunks; never omit the tail.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional source-byte ceiling; exact prompts also include the entire claim.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete semantic document result; provenance has a fixed additional allowance.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Complete retained chunk results; never renewed per chunk or evidence head.
    #[arg(long, default_value_t = 67_108_864)]
    max_retained_result_bytes: u64,
    /// Required original-scalar coverage by classified chunks; 1000000 requires all.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_classified_coverage_ppm: u32,
    /// Execute up to N whole chunks together; each actual window must fit 64 evidence heads.
    #[arg(long, value_name = "N")]
    cohort_chunks: Option<usize>,
}

pub(super) fn definition() -> clap::Command {
    MappingArgs::augment_args(classify::definition().name("faithfulness-map")
        .about("Judge one claim against every chunk of a complete source and retain original evidence")
        .long_about("Read one strict {mode: faithfulness, source, claim, policy} JSON object. Preserve the complete original source and exact claim, losslessly partitioning only the source with the pinned encoder and claim-aware prompt capacity. Every chunk uses the existing full-vocabulary NLI judgment and complete evidence windows. Preflight every chunk, every future cohort window and all five whole-document native work ceilings before the first forward. The named reduction preserves decisive entailed and contradicted quotes in original byte/scalar coordinates, including conflicting evidence when it abstains. Classified coverage uses original Unicode scalars. A document decision requires the explicit coverage policy and compatible witnesses; no averaged correctness probability or full-context equivalence is claimed. Serial execution uses one native cache. --cohort-chunks explicitly shares native projections across complete independent chunks; every actual ragged window must fit at most 64 simultaneous heads, without resizing or serial fallback. All plans, source/claim ownership, native state, retained evidence and complete publication are memory-admitted before input IO. Exact source conversion and independently pinned snapshots use the same route; optional snapshot saving follows complete model-bound preflight. Any failed chunk yields no partial result. Only faithfulness mode is accepted. No generation, sampling, thinking, tools, prompt overrides, --prefill-rows or --cohort-heads. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("One complete faithfulness JSON object; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Maximum complete JSON request bytes, including both original source and exact claim"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native result per chunk; --max-map-result-bytes bounds the document"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("262144")
            .help("Whole-document native forward ceiling across all chunks and evidence heads"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("1000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative allowance for complete planning, native chunks and evidence reduction")))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MapLimits {
    pub common: Int4Limits,
    mapping: Int4FaithfulnessMapLimits,
    planning: JudgeLimits,
    memory: Int4FaithfulnessMapMemoryRequirement,
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
            || common.result_bytes as u64 > m.max_retained_result_bytes
            || m.cohort_chunks.is_some_and(|chunks| !(1..=MAX_INT4_SEQUENCES).contains(&chunks)
                || chunks > m.max_chunks) {
            return Err(CandidateError::Arguments);
        }
        let aggregation = FaithfulnessAggregationPolicy {
            minimum_classified_coverage_ppm: m.minimum_classified_coverage_ppm };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let budget = self.budget(common)?;
        let planning = self.scoring.judge_limits(budget);
        let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(args.max_input_bytes),
            max_chunk_tokens: common.max_prompt_tokens,
            context_tokens: args.context_tokens, reserved_tokens: budget.max_output_tokens as usize,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = Int4FaithfulnessMapLimits { chunks, max_model_work: self.scoring.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes, aggregation };
        let memory_limits = Int4FaithfulnessMapMemoryLimits { context_tokens: args.context_tokens,
            preparation_reserve_bytes: common.preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES };
        let memory = if let Some(chunks) = m.cohort_chunks {
            // Source partitions may have different evidence-head counts.
            // Price every admissible <=64-head window before input IO; the
            // complete actual geometry is checked without changing its width.
            memory_limits.cohort_requirements(args.group_width, budget, planning, mapping, 32, chunks)
                .map_err(|_| CandidateError::Arguments)?.memory
        } else {
            memory_limits.requirements(args.group_width, budget, planning, mapping)
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
    fn budget(self, common: Int4Limits) -> Result<TaskBudget, CandidateError> {
        let mut budget = self.scoring.budget(common);
        // Full-source plus at most 31 evidence heads in one chunk. Logical
        // task authority stays identical for serial and cohort scheduling.
        budget.max_kv_bytes = common.native.kv_bytes.checked_mul(32).ok_or(CandidateError::Arguments)?;
        Ok(budget)
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

/// Keep the ordinary judge's strict faithfulness wire. Its full-source
/// partition check belongs to each model-bound chunk here, not the original
/// long source. No caller-provided tokens, budgets or execution receipts.
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Faithfulness { source: String, claim: String, policy: FaithfulnessPolicy },
}
fn parse_request(json: &str, budget: TaskBudget, cap: usize) -> Result<JudgeRequest, CandidateError> {
    if cap == 0 || cap > MAX_INPUT_BYTES || json.len() > cap { return Err(CandidateError::Input); }
    budget.validate().map_err(|_| CandidateError::Arguments)?;
    let value = canonjson::parse_str_with_limits(json, canonjson::ParseLimits { max_depth: 8, max_string_bytes: cap })
        .map_err(|_| CandidateError::Input)?;
    let Input::Faithfulness { source, claim, policy } = serde_json::from_value(value)
        .map_err(|_| CandidateError::Input)?;
    if source.is_empty() || claim.is_empty()
        || source.len().checked_add(claim.len()).is_none_or(|bytes| bytes > cap) {
        return Err(CandidateError::Input);
    }
    policy.validate().map_err(|_| CandidateError::Input)?;
    Ok(JudgeRequest::Faithfulness { source, claim, policy, budget })
}
