//! Complete source-anchored INT4 entity resolution over an explicit snapshot.
use super::*;
use serde::Deserialize;
use crate::{batch::BatchWork,
    corpus::{resolve::{ResolutionDocument, ResolveOptions, ResolveLimits},
        native_resolve::{NativeResolveLimits, grouped_int4::Int4ResolveLimits}},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct GraphArgs {
    #[arg(long, default_value_t = 256)]
    max_documents: usize,
    #[arg(long, default_value_t = 4096)]
    max_mentions: usize,
    /// Complete lexical candidate-pair ceiling; zero refuses any required comparison.
    #[arg(long, default_value_t = 256)]
    max_pairs: usize,
    #[arg(long, default_value_t = 1_000_000)]
    max_pair_visits: u64,
    #[arg(long, default_value_t = 10_000_000)]
    max_cluster_checks: u64,
    #[arg(long, default_value_t = 268_435_456)]
    max_scan_steps: u64,
    /// Modeled lexical graph, pair ticket and clustering memory; additional to preparation.
    #[arg(long, default_value_t = 64)]
    graph_reserve_mib: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResolveArgs { scoring: classify::ScoringArgs, graph: GraphArgs }

// Deliberately no Debug: original documents and mention surfaces are private.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Population {
    documents: Vec<ResolutionDocument>,
    options: ResolveOptions,
}

pub(super) fn definition() -> clap::Command {
    GraphArgs::augment_args(classify::definition().name("resolve")
        .about("Resolve exact source-anchored mentions across a complete snapshot with native INT4")
        .long_about("Read one bounded JSON object containing original documents, anchored mentions and an explicit lexical/context/margin policy. Lexical overlap selects candidates only. Every pair is scored in both presentation orders with the exact A/B/C-plus-EOS language and full-vocabulary denominators. Complete-link clustering refuses missing, conflicting or uncertain cross-pairs. The whole graph, every source anchor and all five native work axes are admitted before the first forward. Source conversion and independently digest-pinned snapshots use the same actual model identity. Empty candidate graphs allocate no native engine, KV or scratch after model loading. Scores are uncalibrated and cluster IDs are snapshot-local. No automatic entity discovery, generation controls, prefill override, thinking, tools, truncation or backend fallback. Blocking IO observes the total deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("JSON {documents:[{id,text,mentions:[{entity_type,surface,span}]}],options:{blocking,context_scalars,minimum_margin_milli}}; '-' reads stdin"))
        .mut_arg("max_candidate_tokens", |arg| arg.default_value("2").hide(true))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native resolution result; candidate provenance is additional"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-snapshot native forward ceiling, shared by every pair and both presentation orders")))
}
impl ResolveArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if matches.value_source("max_candidate_tokens") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "INT4 resolution uses a fixed label-plus-EOS language"));
        }
        Ok(Self { scoring: classify::ScoringArgs::from_matches(matches)?, graph: GraphArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let mut limits = self.scoring.validate(args)?;
        let g = self.graph;
        if self.scoring.budget(limits).max_output_tokens != 2
            || !(1..=4096).contains(&g.max_documents) || !(1..=16_384).contains(&g.max_mentions)
            || g.max_pairs > 65_536 || !(1..=1_000_000_000).contains(&g.max_pair_visits)
            || !(1..=1_000_000_000).contains(&g.max_cluster_checks)
            || !(1..=1_000_000_000_000).contains(&g.max_scan_steps) {
            return Err(CandidateError::Arguments);
        }
        let graph_bytes = self.graph_bytes()?;
        let graph_floor = sum(&[g.max_documents as u64 * 128, g.max_mentions as u64 * 1024,
            g.max_pairs as u64 * 4096, limits.result_bytes as u64 * 16, MIB])?;
        if graph_bytes < graph_floor { return Err(CandidateError::Arguments); }
        // The CLI keeps its input/planning/publication owner. The host separately
        // retains the moved document capacities, its pinned planner/preparation
        // and complete graph until physical completion. This is a conservative
        // declared reservation quote, not an OS RSS estimate.
        limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, limits.preparation_bytes,
            graph_bytes, self.host_input_bytes(args)?])?;
        if limits.minimum_memory_bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(limits)
    }
    fn graph_bytes(self) -> Result<u64, CandidateError> { mib(self.graph.graph_reserve_mib) }
    fn host_input_bytes(self, args: &Int4Args) -> Result<u64, CandidateError> {
        sum(&[args.max_input_bytes as u64 * 32, self.graph.max_documents as u64 * 128,
            self.graph.max_mentions as u64 * 256])
    }
    fn graph_limits(self, args: &Int4Args, limits: Int4Limits) -> ResolveLimits {
        ResolveLimits { max_documents: self.graph.max_documents, max_mentions: self.graph.max_mentions,
            max_input_bytes: args.max_input_bytes, max_surface_bytes: 1024, max_keys_per_mention: 32,
            max_block_members: 512, max_candidate_pairs: self.graph.max_pairs,
            max_pair_visits: self.graph.max_pair_visits, max_cluster_checks: self.graph.max_cluster_checks,
            max_scan_steps: self.graph.max_scan_steps, max_result_bytes: limits.result_bytes }
    }
    fn scoring_limits(self, args: &Int4Args, limits: Int4Limits) -> Int4ResolveLimits {
        let work = self.scoring.work_ceiling();
        Int4ResolveLimits { planning: NativeResolveLimits {
            per_head: self.scoring.budget(limits), max_pairs: self.graph.max_pairs,
            max_context_tokens: args.context_tokens, max_total_prompt_tokens: work.forward_positions as usize,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
            max_retained_guard_bytes: 4 * 1024 * 1024, max_result_bytes: limits.result_bytes,
        }, max_model_work: work }
    }
    fn parse(self, text: &str, cap: usize) -> Result<Population, CandidateError> {
        if text.len() > cap { return Err(CandidateError::Input); }
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
            max_depth: 12, max_string_bytes: cap,
        }).map_err(|_| CandidateError::Input)?;
        let input: Population = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
        if input.documents.is_empty() || input.documents.len() > self.graph.max_documents
            || input.options.context_scalars > 2048
            || !(1..=1_000_000).contains(&input.options.minimum_margin_milli) {
            return Err(CandidateError::Input);
        }
        // The authoritative ResolutionPlan validates original byte/scalar
        // anchors, duplicate IDs, complete candidate population and scan work.
        Ok(input)
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
