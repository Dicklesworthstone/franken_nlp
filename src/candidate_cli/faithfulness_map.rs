//! Complete-source INT8 faithfulness using the existing strict judgment wire.
pub(super) mod corpus;
use super::*;
use serde::Deserialize;
use crate::{
    tasks::{ir::TaskBudget, judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy,
        JudgeLimits, JudgeRequest, quantized::faithfulness_map::{
            Int8FaithfulnessMapLimits, MAX_FAITHFULNESS_MAP_HEADS}}, mapreduce::ChunkLimits},
    tokenizer::untrusted::UntrustedEncoding,
    native_engine::strict_int8::prefill::Int8PrefillLimits,
};

#[derive(Args)]
pub(super) struct MappingArgs {
    /// Refuse rather than omit any part of a source needing more chunks.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Source-byte ceiling per chunk; the entire claim must also fit each head.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    /// One partition allowance, including every rejected sizing probe.
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete document result, including all chunks and original evidence.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Cumulative retained chunk results, never renewed per evidence head.
    #[arg(long, default_value_t = 16_777_216)]
    max_retained_result_bytes: u64,
    /// Non-abstaining original-scalar coverage required for a document decision.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_classified_coverage_ppm: u32,
}

pub(crate) struct FaithfulnessMapCommand {
    pub(super) scoring: scored::ScoredArgs,
    pub(super) mapping: MappingArgs,
    pub(super) encoding: input_encoding::DocumentInputEncodingArgs,
    pub(super) compute: compute::Args,
    pub(super) prefill: scored::prefill::ScoringPrefillArgs,
}

pub(super) struct MapLimits {
    pub(super) common: CandidateArgs,
    pub(super) limits: Limits,
    pub(super) task: TaskBudget,
    pub(super) planning: JudgeLimits,
    pub(super) mapping: Int8FaithfulnessMapLimits,
    pub(super) encoding: UntrustedEncoding,
    pub(super) prefill: Option<Int8PrefillLimits>,
    pub(super) envelope_bytes: usize,
}

pub(super) fn definition() -> clap::Command {
    compute::definition(input_encoding::document_definition(scored::prefill::ScoringPrefillArgs::augment_args(MappingArgs::augment_args(scored::ScoredArgs::augment_args(
        clap::Command::new("faithfulness-map")
            .about("Judge a claim against every chunk of a complete source with native INT8")
            .long_about("Read one strict {mode: faithfulness, source, claim, policy} JSON object. Preserve the complete source and unchanged claim, partitioning only the source into nonoverlapping UTF-8 chunks with claim-aware prompt capacity. Plan every chunk and every evidence window before loading weights, then reuse one native INT8 cache. Full-vocabulary candidate scores retain EOS and the existing judgment policy. The shared document reduction preserves original byte/scalar evidence coordinates, including contrary witnesses when it abstains. A document decision requires the declared classified-coverage floor and compatible evidence. Independent chunk judgments do not establish cross-chunk inference, calibrated correctness or a factuality certificate. Any failed chunk, cancellation or exhausted allowance returns no complete document result."))))))
        .mut_arg("input", |arg| arg.help("One complete faithfulness JSON object; '-' reads stdin"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576")
            .help("Complete JSON byte cap, including both original source and exact claim"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_result_bytes", |arg| arg.default_value("65536")
            .help("Complete native result ceiling per chunk; document output has a separate cap"))
        .mut_arg("max_forward_positions", |arg| arg.default_value("262144")
            .help("Whole-document forward allowance across all chunks and evidence heads"))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_attention_pairs", |arg| arg.default_value("1000000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("10000000000000000"))
        .mut_arg("max_multiply_accumulates", |arg| arg.default_value("10000000000000000"))
        .after_long_help("Explicit compact input: --input-encoding safe-bpe-v1 preserves exact source and claim bytes without admitting template-control tokens. One bounded tokenizer ledger covers claim capacity, accepted and rejected source probes, and every final evidence-head plan in each complete preparation. CLI preflight and owned hosted preparation use the same fixed policy; work never renews per chunk or head. Optional --compute-threads reuses one bounded persistent CPU team across every source chunk and evidence head; --prefill-rows bounds prompt staging on the same execution path. Scheduling preserves the exact claim, complete work allowance and source/result bindings. Native work, source bytes, context, tokenizer work, retained chunks and final result have independent ceilings. The complete process memory envelope is checked before input/model IO. The original cooperative deadline spans reading, preparation, loading, inference, evidence reduction and checked publication. Only faithfulness mode is accepted; no sampling, thinking, tools, cohorts, prompt overrides or retained jobs.")
}

impl FaithfulnessMapCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: scored::ScoredArgs::from_arg_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?,
            encoding: input_encoding::DocumentInputEncodingArgs::from_matches(matches)?,
            compute: compute::Args::from_matches(matches)?,
            prefill: scored::prefill::ScoringPrefillArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(&self) -> Result<MapLimits, CandidateError> {
        let (mut common, mut limits) = self.scoring.common()?;
        let prefill = self.prefill.limits()?;
        common.compute = self.compute;
        limits.compute = self.compute.admit_prefill(prefill)?;
        let encoding = self.encoding.policy()?;
        let m = &self.mapping;
        if self.scoring.max_input_bytes < 4 || !(1..=256).contains(&m.max_chunks)
            || !(1..=1_000_000).contains(&m.max_tokenizer_calls)
            || m.max_chunk_bytes.is_some_and(|n| !(4..=self.scoring.max_input_bytes).contains(&n))
            || !(1..=64 * MIB).contains(&m.max_map_result_bytes)
            || !(1..=64 * MIB).contains(&m.max_retained_result_bytes)
            || self.scoring.max_result_bytes as u64 > m.max_retained_result_bytes {
            return Err(CandidateError::Arguments);
        }
        let aggregation = FaithfulnessAggregationPolicy {
            minimum_classified_coverage_ppm: m.minimum_classified_coverage_ppm };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let mut task = self.scoring.budget(limits);
        // The semantic bundle contains a global judgment and up to 31 evidence
        // heads. Physical execution reuses only one admitted native cache.
        task.max_kv_bytes = task.max_kv_bytes.checked_mul(MAX_FAITHFULNESS_MAP_HEADS as u64)
            .ok_or(CandidateError::Arguments)?;
        let chunks = ChunkLimits {
            max_input_bytes: self.scoring.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(self.scoring.max_input_bytes),
            max_chunk_tokens: task.max_input_tokens as usize,
            context_tokens: self.scoring.context_tokens,
            reserved_tokens: task.max_output_tokens as usize,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls,
        };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = Int8FaithfulnessMapLimits { chunks, max_model_work: self.scoring.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes, aggregation };
        let envelope_bytes = usize::try_from(m.max_map_result_bytes).ok()
            .and_then(|n| n.checked_add(4096)).ok_or(CandidateError::Arguments)?;
        let encoder_bytes = self.encoding.workspace_bytes()?;
        let floor = (256 * MIB).checked_add(self.scoring.max_input_bytes as u64 * 32)
            .and_then(|n| n.checked_add((envelope_bytes as u64).checked_mul(2)?))
            .and_then(|n| n.checked_add(encoder_bytes)).ok_or(CandidateError::Arguments)?;
        if floor > limits.preparation_bytes { return Err(CandidateError::Arguments); }
        let admitted = MapLimits { common, limits, task,
            planning: judge::planning_limits(&self.scoring, task), mapping, encoding, prefill, envelope_bytes };
        #[cfg(feature = "asupersync-runtime")]
        runtime::faithfulness_map::admit(&admitted)?;
        Ok(admitted)
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let admitted = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::faithfulness_map::execute(admitted, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (admitted, input, output); Err(CandidateError::Unavailable) }
    }
}

/// This narrow wire intentionally omits the one-context source partition gate:
/// every model-bound chunk must pass that gate after exact claim-aware sizing.
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Faithfulness { source: String, claim: String, policy: FaithfulnessPolicy },
}
pub(super) fn request(json: &str, admitted: &MapLimits) -> Result<JudgeRequest, CandidateError> {
    let cap = admitted.common.max_input_bytes;
    if json.len() > cap { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(json, canonjson::ParseLimits {
        max_depth: 8, max_string_bytes: cap,
    }).map_err(|_| CandidateError::Input)?;
    let Input::Faithfulness { source, claim, policy } = serde_json::from_value(value)
        .map_err(|_| CandidateError::Input)?;
    if source.is_empty() || claim.is_empty()
        || source.len().checked_add(claim.len()).is_none_or(|bytes| bytes > cap) {
        return Err(CandidateError::Input);
    }
    policy.validate().map_err(|_| CandidateError::Input)?;
    Ok(JudgeRequest::Faithfulness { source, claim, policy, budget: admitted.task })
}

#[cfg(test)] pub(super) mod tests;
