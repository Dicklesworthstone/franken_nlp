//! Source-anchored whole-snapshot entity resolution, not lexical alias merging.
use super::*;
use serde::Deserialize;
use crate::{batch::BatchWork,
    corpus::{resolve::{ResolutionDocument, ResolveOptions, ResolveLimits},
        native_resolve::{NativeResolveLimits, quantized::Int8ResolveLimits}},
};
use scored::ScoredArgs;

#[derive(Args)]
pub(crate) struct ResolveCommand {
    #[command(flatten)]
    pub(super) host: ScoredArgs,
    #[arg(long, default_value_t = 256)]
    max_documents: usize,
    #[arg(long, default_value_t = 4096)]
    max_mentions: usize,
    /// Complete candidate-pair ceiling; zero refuses any required comparison.
    #[arg(long, default_value_t = 256)]
    max_pairs: usize,
    #[arg(long, default_value_t = 1_000_000)]
    max_pair_visits: u64,
    #[arg(long, default_value_t = 10_000_000)]
    max_cluster_checks: u64,
    #[arg(long, default_value_t = 268_435_456)]
    max_scan_steps: u64,
    /// Modeled graph/ticket/cluster memory in addition to shared preparation.
    #[arg(long, default_value_t = 64)]
    pub(super) graph_reserve_mib: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub documents: Vec<ResolutionDocument>,
    /// Required explicit lexical/context/margin policy; not calibrated confidence.
    pub options: ResolveOptions,
}

pub(super) fn definition() -> clap::Command {
    ResolveCommand::augment_args(clap::Command::new("resolve")
        .about("Resolve exact source-anchored mentions across a bounded document snapshot")
        .after_help("Input is {documents:[{id,text,mentions:[{entity_type,surface,span}]}],options:{blocking,context_scalars,minimum_margin_milli}}. Original source and byte/scalar offsets are required. Lexical overlap only selects candidates; both model presentation orders must agree, and complete-link clustering refuses missing or conflicting cross-pairs. Scores are uncalibrated and cluster IDs are snapshot-local. This command does not discover missing mentions or truncate an oversized graph."))
}
impl ResolveCommand {
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        let (common, limits) = self.host.common()?;
        if !(1..=4096).contains(&self.max_documents) || !(1..=16_384).contains(&self.max_mentions)
            || self.max_pairs > 65_536 || self.max_pair_visits == 0 || self.max_pair_visits > 1_000_000_000
            || self.max_cluster_checks == 0 || self.max_cluster_checks > 1_000_000_000
            || self.max_scan_steps == 0 || self.max_scan_steps > 1_000_000_000_000 {
            return Err(CandidateError::Arguments);
        }
        let bytes = self.graph_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        let floor = (self.max_documents as u64).checked_mul(128)
            .and_then(|n| n.checked_add(self.max_mentions as u64 * 1024))
            .and_then(|n| n.checked_add(self.max_pairs as u64 * 4096))
            .and_then(|n| n.checked_add(self.host.max_result_bytes as u64 * 16))
            .and_then(|n| n.checked_add(MIB)).ok_or(CandidateError::Arguments)?;
        if bytes < floor || bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok((common, limits))
    }
    pub(super) fn graph(&self) -> ResolveLimits {
        ResolveLimits { max_documents: self.max_documents, max_mentions: self.max_mentions,
            max_input_bytes: self.host.max_input_bytes, max_surface_bytes: 1024,
            max_keys_per_mention: 32, max_block_members: 512, max_candidate_pairs: self.max_pairs,
            max_pair_visits: self.max_pair_visits, max_cluster_checks: self.max_cluster_checks,
            max_scan_steps: self.max_scan_steps, max_result_bytes: self.host.max_result_bytes }
    }
    pub(super) fn scoring(&self, limits: Limits) -> Int8ResolveLimits {
        Int8ResolveLimits { planning: NativeResolveLimits {
            per_head: self.host.budget(limits), max_pairs: self.max_pairs,
            max_context_tokens: self.host.context_tokens,
            max_total_prompt_tokens: self.host.max_forward_positions as usize,
            max_work: BatchWork { forward_positions: self.host.max_forward_positions,
                projected_logits: self.host.max_projected_logits },
            max_retained_guard_bytes: 4 * 1024 * 1024, max_result_bytes: self.host.max_result_bytes,
        }, max_model_work: self.host.work_ceiling() }
    }
    pub(super) fn input(&self, text: &str) -> Result<Input, CandidateError> {
        if text.len() > self.host.max_input_bytes { return Err(CandidateError::Input); }
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
            max_depth: 12, max_string_bytes: self.host.max_input_bytes,
        }).map_err(|_| CandidateError::Input)?;
        let input: Input = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
        if input.documents.is_empty() || input.documents.len() > self.max_documents
            || input.options.context_scalars > 2048 || !(1..=1_000_000).contains(&input.options.minimum_margin_milli) {
            return Err(CandidateError::Input);
        }
        // Exact anchors, duplicate IDs and candidate budgets are validated by
        // ResolutionPlan, under the same preparation control, not reimplemented.
        Ok(input)
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (common, limits) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::resolution::execute(self, common, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

#[cfg(test)] pub(super) mod tests;
