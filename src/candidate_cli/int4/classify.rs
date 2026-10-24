//! Exact finite-label requests over the shared source/snapshot INT4 host.
//! The existing scored CLI parser owns the wire contract; JSON cannot supply
//! model identity, work certificates, EOS, probability space or task budgets.
use super::*;
use crate::{batch::BatchWork,
    native_engine::{lmhead::scoring::ScoringLimits, portable_int8::ProjectionWork,
        strict_int4::Int4Work},
    tasks::{classify::{ClassificationLimits, ClassificationRequest}, ir::TaskBudget}};

// Retain the shared source/load parser, but never accept a free-text control
// on a scored task, even when the explicitly supplied value equals its default.
const TEXT_ONLY: &[&str] = &[
    "max_new_tokens", "max_output_bytes", "seed", "temperature_milli", "top_k", "top_p_ppm", "logprobs",
    "prefill_rows", "min_new_tokens", "stop_suffixes", "banned_token_ids", "repetition_penalty_milli",
    "presence_penalty_milli", "frequency_penalty_milli", "logit_bias",
];

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ScoringArgs {
    /// Maximum candidate depth INCLUDING its scored EOS, not generated text.
    #[arg(long, default_value_t = 16)]
    max_candidate_tokens: usize,
    /// Complete native classification result; candidate provenance is additional.
    #[arg(long, default_value_t = 1_048_576)]
    max_result_bytes: usize,
    /// Whole-bundle forward ceiling; never renewed for individual labels.
    #[arg(long, default_value_t = 131_072)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 100_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    max_attention_pairs: u64,
    /// Counts each matrix's groups, including its partial tail group.
    #[arg(long, default_value_t = 100_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)]
    max_multiply_accumulates: u64,
}

pub(super) fn definition() -> clap::Command {
    let mut command = Int4Args::augment_args(clap::Command::new("classify")
        .about("Score every label exactly; exclusive or independent multi-label INT4 decisions"))
        .mut_arg("input", |arg| arg.help("Bounded classification JSON object; '-' reads stdin"))
        .mut_arg("preparation_mib", |arg| arg.default_value("512"));
    for &id in TEXT_ONLY { command = command.mut_arg(id, |arg| arg.hide(true)); }
    ScoringArgs::augment_args(command)
}
impl ScoringArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if TEXT_ONLY.iter().any(|id| matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine)) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "INT4 finite-candidate tasks do not accept generation or prefill controls"));
        }
        Self::from_arg_matches(matches)
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        if !(2..=64).contains(&self.max_candidate_tokens) || self.max_candidate_tokens >= args.context_tokens
            || !(1..=MAX_CONTENT_BYTES).contains(&self.max_result_bytes)
            || !(1..=16 * 1024 * 1024).contains(&self.max_forward_positions)
            || self.max_projected_logits == 0 || self.max_attention_pairs == 0
            || self.max_dot_products == 0 || self.max_multiply_accumulates == 0 {
            return Err(CandidateError::Arguments);
        }
        // Defend programmatically constructed arguments too. These defaults
        // are only parser storage; no sampler or generation plan is created.
        if args.max_new_tokens != 128 || args.max_output_bytes != 65_536
            || args.options(166_101)? != GenerationOptions::greedy(128, 65_536, 166_101) {
            return Err(CandidateError::Arguments);
        }
        // Bound repeated document/prompt/IR storage by the aggregate forward
        // ceiling, not by input bytes or just the longest live native context.
        let extra = sum(&[128 * MIB, self.max_forward_positions * 32, self.max_result_bytes as u64 * 16])?;
        args.validate_storage(self.max_result_bytes, self.max_candidate_tokens,
            args.context_tokens - self.max_candidate_tokens + 1, 0, extra)
    }
    pub(super) fn budget(self, limits: Int4Limits) -> TaskBudget {
        TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: self.max_candidate_tokens as u32, max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: 4096, max_kv_bytes: limits.native.kv_bytes }
    }
    pub(super) fn classification_limits(self, args: &Int4Args) -> ClassificationLimits {
        ClassificationLimits { max_labels: 128, max_input_bytes: args.max_input_bytes,
            max_label_id_bytes: 256, max_label_description_bytes: 4096, max_total_label_bytes: 64 * 1024,
            max_context_tokens: args.context_tokens, max_total_prompt_tokens: self.max_forward_positions as usize,
            max_work: BatchWork { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits },
            scoring: ScoringLimits { max_candidates: 128, max_total_tokens: 4095, max_nodes: 4096,
                max_depth: self.max_candidate_tokens, max_candidate_id_bytes: 64,
                max_projected_logits: self.max_projected_logits } }
    }
    pub(super) fn sentiment_limits(self, args: &Int4Args) -> crate::tasks::sentiment::SentimentLimits {
        crate::tasks::sentiment::SentimentLimits {
            per_axis: self.classification_limits(args).scoring,
            max_total_prompt_tokens: self.max_forward_positions as usize,
            max_total_candidate_tokens: 4096, max_total_nodes: 4100,
            max_total_projected_logits: self.max_projected_logits,
            max_output_bytes: self.max_result_bytes as u64,
        }
    }
    pub(super) fn work_ceiling(self) -> Int4Work {
        Int4Work { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits,
            attention_pairs: self.max_attention_pairs, projections: ProjectionWork {
                dot_products: self.max_dot_products, multiply_accumulates: self.max_multiply_accumulates } }
    }
    pub(super) fn admit_plan(self, context: usize, capacity: usize, work: Int4Work) -> Result<(), CandidateError> {
        if context == 0 || context > capacity || work.forward_positions > self.max_forward_positions
            || work.projected_logits > self.max_projected_logits || work.attention_pairs > self.max_attention_pairs
            || !work.projections.fits(ProjectionWork { dot_products: self.max_dot_products,
                multiply_accumulates: self.max_multiply_accumulates }) {
            return Err(CandidateError::Planning);
        }
        Ok(())
    }
}

pub(super) fn parse_request(text: &str, budget: TaskBudget, cap: usize) -> Result<ClassificationRequest, CandidateError> {
    use crate::candidate_cli::scored::{self, Kind, Request};
    match scored::request(Kind::Classify, text, budget, cap)? {
        Request::Classify(request) => Ok(request),
        Request::Sentiment { .. } => Err(CandidateError::Input),
    }
}

#[cfg(test)] mod tests;
