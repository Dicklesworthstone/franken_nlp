//! Complete-document INT8 classification with explicit mixture semantics.
//! Shares one-context request parsing and scoring policy; never truncates input.
use super::*;
use crate::{
    tasks::{classify::{ClassificationAggregationPolicy, ClassificationLimits, ClassificationRequest,
        quantized::long::Int8ClassificationMapLimits}, ir::TaskBudget, mapreduce::ChunkLimits},
    tokenizer::untrusted::UntrustedEncoding,
    native_engine::strict_int8::prefill::Int8PrefillLimits,
};

#[derive(Args)]
pub(super) struct MappingArgs {
    /// Refuse rather than truncate a document that needs more chunks.
    #[arg(long, default_value_t = 64)]
    max_chunks: usize,
    /// Additional source-byte cap per chunk; native context still decides fit.
    #[arg(long)]
    max_chunk_bytes: Option<usize>,
    /// One partition-wide ceiling, including rejected sizing probes.
    #[arg(long, default_value_t = 8192)]
    max_tokenizer_calls: usize,
    /// Complete map plus aggregate; --max-result-bytes still caps each chunk.
    #[arg(long, default_value_t = 16_777_216)]
    max_map_result_bytes: u64,
    /// Cumulative complete chunk results retained before final aggregation.
    #[arg(long, default_value_t = 16_777_216)]
    max_retained_result_bytes: u64,
    /// Required non-abstaining original-source coverage for each decision.
    #[arg(long, default_value_t = 1_000_000)]
    minimum_classified_coverage_ppm: u32,
}

pub(crate) struct ClassifyMapCommand {
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
    pub(super) planning: ClassificationLimits,
    pub(super) mapping: Int8ClassificationMapLimits,
    pub(super) encoding: UntrustedEncoding,
    pub(super) prefill: Option<Int8PrefillLimits>,
    pub(super) envelope_bytes: usize,
}

pub(super) fn definition() -> clap::Command {
    compute::definition(input_encoding::document_definition(scored::prefill::ScoringPrefillArgs::augment_args(MappingArgs::augment_args(scored::ScoredArgs::augment_args(
        clap::Command::new("classify-map")
            .about("Classify a complete document with bounded native INT8 chunks and a source-weighted aggregate")
            .long_about("Read one complete {document,labels,mode?,policy?} JSON request. Losslessly partition the exact UTF-8 source into nonoverlapping chunks that fit every classification head. Preflight every chunk and all five native work axes before loading weights. Execute on one resident INT8 model and reuse one native context. Return every complete chunk result with original byte/scalar coordinates, plus the existing source-scalar-weighted classification mixture. Exclusive classification and independent multi-label decisions retain their declared thresholds and full-vocabulary sequence scores. Aggregate weights are descriptive and uncalibrated; chunk mixtures are not single-context equivalents. A failed chunk, exhausted budget or cancellation returns no document result."))))))
        .mut_arg("preparation_mib", |arg| arg.default_value_if("input_encoding",
            clap::builder::ArgPredicate::Equals("safe-bpe-v1".into()), "1024"))
        .after_long_help("Compact input: --input-encoding safe-bpe-v1 preserves exact untrusted source and label bytes. One bounded tokenizer ledger covers capacity sizing, accepted and rejected partition probes, and all final chunk/head plans in each complete preparation. The CLI preflight and later owned hosted preparation each use the same fixed policy with a fresh per-preparation ledger; neither replenishes work per chunk. Tokenizer byte/token ceilings are cumulative and separate from per-chunk context. Compact input changes prompt tokens and carries no parity, quality or speed claim. Optional --compute-threads reuses one bounded persistent CPU team across every chunk and head; --prefill-rows bounds layer-major prompt staging. Both preserve the sealed work ceiling and source binding. One process memory ledger prices retained plans, tokenizer workspace, complete chunk results, native scratch, worker buffers and final output staging before input/model IO. No cohorts, sampling, thinking, tools or retained jobs are accepted by this command.")
}

impl ClassifyMapCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: scored::ScoredArgs::from_arg_matches(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?,
            encoding: input_encoding::DocumentInputEncodingArgs::from_matches(matches)?,
            compute: compute::Args::from_matches(matches)?,
            prefill: scored::prefill::ScoringPrefillArgs::from_arg_matches(matches)? })
    }

    pub(super) fn validate(&self) -> Result<MapLimits, CandidateError> {
        // One-context common arithmetic remains unchanged. Document encoding
        // has distinct cumulative limits and is admitted below, not injected
        // into a per-request token ceiling that would reset between chunks.
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
        let aggregation = ClassificationAggregationPolicy {
            minimum_classified_coverage_ppm: m.minimum_classified_coverage_ppm };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let task = self.scoring.budget(limits);
        let chunks = ChunkLimits {
            max_input_bytes: self.scoring.max_input_bytes,
            max_chunk_bytes: m.max_chunk_bytes.unwrap_or(self.scoring.max_input_bytes),
            max_chunk_tokens: task.max_input_tokens as usize,
            context_tokens: self.scoring.context_tokens,
            reserved_tokens: task.max_output_tokens as usize,
            max_chunks: m.max_chunks, max_tokenizer_calls: m.max_tokenizer_calls,
        };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let mapping = Int8ClassificationMapLimits { chunks, max_model_work: self.scoring.work_ceiling(),
            max_retained_result_bytes: m.max_retained_result_bytes,
            max_result_bytes: m.max_map_result_bytes, aggregation };
        let envelope_bytes = usize::try_from(m.max_map_result_bytes).ok()
            .and_then(|n| n.checked_add(4096)).ok_or(CandidateError::Arguments)?;
        // Staging is held through write AND flush, including the retained
        // native guard. Its complete serialized size can exceed one chunk.
        let encoder_bytes = self.encoding.workspace_bytes()?;
        let floor = (256 * MIB).checked_add(self.scoring.max_input_bytes as u64 * 32)
            .and_then(|n| n.checked_add((envelope_bytes as u64).checked_mul(2)?))
            .and_then(|n| n.checked_add(encoder_bytes)).ok_or(CandidateError::Arguments)?;
        if floor > limits.preparation_bytes { return Err(CandidateError::Arguments); }
        let admitted = MapLimits { common, limits, task, planning: self.scoring.classification_limits(),
            mapping, encoding, prefill, envelope_bytes };
        #[cfg(feature = "asupersync-runtime")]
        runtime::classify_map::admit(&admitted)?;
        Ok(admitted)
    }

    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let limits = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::classify_map::execute(limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (limits, input, output); Err(CandidateError::Unavailable) }
    }
}

pub(super) fn request(json: &str, limits: &MapLimits) -> Result<ClassificationRequest, CandidateError> {
    match scored::request(scored::Kind::Classify, json, limits.task, limits.common.max_input_bytes)? {
        scored::Request::Classify(request) => Ok(request),
        _ => Err(CandidateError::Input),
    }
}

#[cfg(test)] pub(super) mod tests;
