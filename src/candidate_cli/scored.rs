//! Finite-score commands. Input is typed data, never a prompt or work certificate.
use super::*;
use serde::Deserialize;
use crate::{batch::BatchWork,
    native_engine::{lmhead::scoring::ScoringLimits,
        portable_int8::ProjectionWork, strict_int8::Int8Work},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode,
        ClassificationPolicy, ClassificationRequest}, ir::TaskBudget,
        sentiment::{SentimentAxis, SentimentLimits, SentimentPolicy, SentimentRequest}},
};

const MAX_LABELS: usize = 128;
const MAX_LABEL_BYTES: usize = 64 * 1024;
const SCORING_NODES: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kind { Classify, Sentiment }
impl Kind {
    pub(super) fn named(name: &str) -> Option<Self> {
        match name { "classify" => Some(Self::Classify), "sentiment" => Some(Self::Sentiment), _ => None }
    }
}

/// No sampling, generation, grammar-mask, tool or identity overrides. Candidate
/// depth includes EOS. Every work ceiling covers the ENTIRE bundle of heads.
#[derive(Args)]
pub(super) struct ScoredArgs {
    /// JSON request object; '-' reads stdin. Documents and labels stay exact.
    #[arg(default_value = "-")]
    pub input: PathBuf,
    /// Explicit local current-candidate INT8 .fnlpq; no discovery or download.
    #[arg(long, value_name = "FILE")]
    pub model: PathBuf,
    /// Process memory ledger in MiB; not an operating-system RSS limit.
    #[arg(long)]
    pub memory_mib: u64,
    /// Largest live head context; independent heads reuse this one allocation.
    #[arg(long, default_value_t = 2048)]
    pub context_tokens: usize,
    /// Maximum candidate depth INCLUDING EOS, not a generated-text length.
    #[arg(long, default_value_t = 16)]
    pub max_candidate_tokens: usize,
    /// Complete native task result; candidate provenance adds at most 4096 bytes.
    #[arg(long, default_value_t = 1_048_576)]
    pub max_result_bytes: usize,
    #[arg(long, default_value_t = 65_536)]
    pub max_input_bytes: usize,
    #[arg(long, default_value_t = 6144)]
    pub max_weight_mib: u64,
    /// Modeled tokenizer, input, repeated prompts, scorer and staging reserve.
    #[arg(long, default_value_t = 512)]
    pub preparation_mib: u64,
    /// Whole-invocation cooperative deadline. Blocking IO is not preempted.
    #[arg(long, default_value_t = 3600)]
    pub timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000_000)]
    pub max_checkpoints: u64,
    #[arg(long, default_value_t = 131_072)]
    pub max_forward_positions: u64,
    #[arg(long, default_value_t = 100_000_000)]
    pub max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    pub max_attention_pairs: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    pub max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)]
    pub max_multiply_accumulates: u64,
}
impl ScoredArgs {
    pub(super) fn common(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        if !(2..=64).contains(&self.max_candidate_tokens) || self.max_candidate_tokens >= self.context_tokens
            || self.preparation_mib < 512 || self.max_forward_positions == 0
            || self.max_forward_positions > 16 * 1024 * 1024 || self.max_projected_logits == 0
            || self.max_attention_pairs == 0 || self.max_dot_products == 0 || self.max_multiply_accumulates == 0 {
            return Err(CandidateError::Arguments);
        }
        // Reuse only shared host/IO/memory arithmetic, never generation policy.
        let common = CandidateArgs { input: self.input.clone(), model: self.model.clone(), memory_mib: self.memory_mib,
            context_tokens: self.context_tokens, max_new_tokens: self.max_candidate_tokens,
            max_output_bytes: self.max_result_bytes, max_input_bytes: self.max_input_bytes,
            max_weight_mib: self.max_weight_mib, preparation_mib: self.preparation_mib,
            timeout_seconds: self.timeout_seconds, max_checkpoints: self.max_checkpoints,
            seed: None, temperature_milli: None, top_k: None, top_p_ppm: None, logprobs: false };
        let limits = common.validate()?;
        // In addition to the shared floor, price the worst permitted repeated
        // exact prompt/IR copies; user input cannot buy an uncharged head set.
        let prompt_bytes = self.max_forward_positions.checked_mul(32).ok_or(CandidateError::Arguments)?;
        let floor = (256 * MIB).checked_add(prompt_bytes)
            .and_then(|n| n.checked_add(self.max_input_bytes as u64 * 32))
            .and_then(|n| n.checked_add(self.max_result_bytes as u64 * 16))
            .ok_or(CandidateError::Arguments)?;
        if limits.preparation_bytes < floor { return Err(CandidateError::Arguments); }
        Ok((common, limits))
    }
    pub(super) fn budget(&self, limits: Limits) -> TaskBudget {
        TaskBudget { max_input_tokens: (self.context_tokens - self.max_candidate_tokens) as u32,
            max_output_tokens: self.max_candidate_tokens as u32, max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: SCORING_NODES as u32, max_kv_bytes: limits.kv_bytes }
    }
    pub(super) fn work_ceiling(&self) -> Int8Work {
        Int8Work { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits,
            attention_pairs: self.max_attention_pairs, projections: ProjectionWork {
                dot_products: self.max_dot_products, multiply_accumulates: self.max_multiply_accumulates } }
    }
    pub(super) fn admit_plan(&self, context: usize, work: Int8Work) -> Result<(), CandidateError> {
        let cap = self.work_ceiling();
        if context == 0 || context > self.context_tokens || work.forward_positions > cap.forward_positions
            || work.projected_logits > cap.projected_logits || work.attention_pairs > cap.attention_pairs
            || !work.projections.fits(cap.projections) { return Err(CandidateError::Planning); }
        Ok(())
    }
    fn scoring(&self) -> ScoringLimits {
        ScoringLimits { max_candidates: MAX_LABELS, max_total_tokens: SCORING_NODES - 1,
            max_nodes: SCORING_NODES, max_depth: self.max_candidate_tokens,
            max_candidate_id_bytes: 64, max_projected_logits: self.max_projected_logits }
    }
    pub(super) fn classification_limits(&self) -> ClassificationLimits {
        ClassificationLimits { max_labels: MAX_LABELS, max_input_bytes: self.max_input_bytes,
            max_label_id_bytes: 256, max_label_description_bytes: 4096, max_total_label_bytes: MAX_LABEL_BYTES,
            max_context_tokens: self.context_tokens, max_total_prompt_tokens: self.max_forward_positions as usize,
            max_work: BatchWork { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits },
            scoring: self.scoring() }
    }
    pub(super) fn sentiment_limits(&self) -> SentimentLimits {
        SentimentLimits { per_axis: self.scoring(), max_total_prompt_tokens: self.max_forward_positions as usize,
            max_total_candidate_tokens: SCORING_NODES, max_total_nodes: SCORING_NODES + 4,
            max_total_projected_logits: self.max_projected_logits, max_output_bytes: self.max_result_bytes as u64 }
    }
}

pub(crate) struct ScoredCommand { pub(super) kind: Kind, pub(super) args: ScoredArgs }
pub(super) fn definitions() -> Vec<clap::Command> {
    [("classify", "Score every label exactly; exclusive or independent multi-label decisions"),
        ("sentiment", "Score independent affect dimensions; uncalibrated, not psychological measurements")]
        .into_iter().map(|(name, about)| ScoredArgs::augment_args(clap::Command::new(name).about(about))).collect()
}
impl ScoredCommand {
    pub(super) fn from_matches(kind: Kind, matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { kind, args: ScoredArgs::from_arg_matches(matches)? })
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (common, limits) = self.args.common()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::scored_tasks::execute(self, common, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassificationInput {
    document: String,
    labels: Vec<ClassificationLabel>,
    #[serde(default = "exclusive")]
    mode: ClassificationMode,
    #[serde(default)]
    policy: ClassificationPolicy,
}
fn exclusive() -> ClassificationMode { ClassificationMode::Exclusive }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SentimentInput {
    document: String,
    #[serde(default = "all_axes")]
    axes: Vec<SentimentAxis>,
    #[serde(default = "sentiment_policy")]
    policy: SentimentPolicy,
}
fn all_axes() -> Vec<SentimentAxis> { SentimentAxis::ALL.to_vec() }
fn sentiment_policy() -> SentimentPolicy {
    SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }
}

pub(super) enum Request {
    Classify(ClassificationRequest),
    Sentiment { request: SentimentRequest, policy: SentimentPolicy },
}
/// All wire fields are checked before model metadata is opened. The host, not
/// request JSON, supplies TaskBudget, model/identity, EOS and scoring semantics.
pub(super) fn request(kind: Kind, json: &str, budget: TaskBudget, cap: usize) -> Result<Request, CandidateError> {
    if cap == 0 || cap > MAX_INPUT_BYTES || json.len() > cap { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(json, canonjson::ParseLimits {
        max_depth: 8, max_string_bytes: cap,
    }).map_err(|_| CandidateError::Input)?;
    Ok(match kind {
        Kind::Classify => {
            let input: ClassificationInput = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
            if input.document.is_empty() || input.labels.is_empty() || input.labels.len() > MAX_LABELS
                || (input.mode == ClassificationMode::Exclusive && input.labels.len() < 2)
                || input.policy.minimum_candidate_weight_ppm > 1_000_000 || input.policy.minimum_margin_ppm > 1_000_000 {
                return Err(CandidateError::Input);
            }
            let mut ids = std::collections::BTreeSet::new();
            let mut bytes = 0_usize;
            for label in &input.labels {
                bytes = bytes.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
                    .ok_or(CandidateError::Input)?;
                if label.id.trim().is_empty() || label.id.len() > 256 || label.id.chars().any(char::is_control)
                    || label.description.len() > 4096 || bytes > MAX_LABEL_BYTES || !ids.insert(label.id.as_str()) {
                    return Err(CandidateError::Input);
                }
            }
            drop(ids);
            Request::Classify(ClassificationRequest { document: input.document, labels: input.labels,
                mode: input.mode, policy: input.policy, budget })
        }
        Kind::Sentiment => {
            let input: SentimentInput = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
            if input.document.is_empty() || input.axes.is_empty() || input.axes.len() > 4
                || input.policy.minimum_peak_weight_ppm > 1_000_000
                || input.policy.maximum_normalized_entropy_ppm > 1_000_000 {
                return Err(CandidateError::Input);
            }
            let unique: std::collections::BTreeSet<_> = input.axes.iter().copied().collect();
            if unique.len() != input.axes.len() { return Err(CandidateError::Input); }
            Request::Sentiment { request: SentimentRequest { document: input.document, axes: input.axes, budget },
                policy: input.policy }
        }
    })
}

#[cfg(test)] pub(super) mod tests;
