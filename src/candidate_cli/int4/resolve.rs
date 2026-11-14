//! Complete source-anchored INT4 entity resolution over an explicit snapshot.
use super::*;
use serde::Deserialize;
#[cfg(feature = "asupersync-runtime")]
use serde::Serialize;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
use super::source_map::encoding::EncodingLimits;
use crate::{batch::BatchWork,
    corpus::entities_int4::cohort::Int4EntitySchedule,
    corpus::{resolve::{ResolutionDocument, ResolveOptions, ResolveLimits},
        native_resolve::{NativeResolveLimits, grouped_int4::Int4ResolveLimits}},
};
mod discovery;
pub(super) mod corpus;

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CohortArgs {
    /// Complete pairs per native window, 1..=32; both orders share one window.
    #[arg(long, value_name = "N")]
    cohort_pairs: Option<usize>,
    /// NER discovery chunks per native window, 1..=64; may cross document boundaries.
    #[arg(long, value_name = "N", requires = "discover_entities")]
    cohort_chunks: Option<usize>,
}
impl CohortArgs {
    fn enabled(self) -> bool { self.cohort_pairs.is_some() || self.cohort_chunks.is_some() }
    fn validate(self, discover: bool) -> Result<(), CandidateError> {
        if self.cohort_pairs.is_some_and(|n| !(1..=32).contains(&n))
            || self.cohort_chunks.is_some_and(|n| !discover || !(1..=64).contains(&n)) {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    fn schedule(self) -> Int4EntitySchedule {
        Int4EntitySchedule { ner_cohort_chunks: self.cohort_chunks, pair_cohort_pairs: self.cohort_pairs }
    }
    fn output_cap(self, semantic_bytes: usize) -> Result<usize, CandidateError> {
        semantic_bytes.checked_add(if self.enabled() { 8192 } else { 4096 })
            .ok_or(CandidateError::Arguments)
    }
}

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

/// Explicit cumulative preparation work. NER and pair scoring have separate
/// ledgers because the complete pair graph is known only after discovery.
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EncodingArgs {
    #[arg(long)]
    max_total_tokenizer_input_bytes: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_output_tokens: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_checks: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResolveArgs {
    scoring: classify::ScoringArgs, graph: GraphArgs, discovery: discovery::DiscoveryArgs,
    cohorts: CohortArgs, encoding: EncodingArgs,
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
struct EncodingReport {
    policy: UntrustedEncoding,
    complete_phase_limits: SafeBpeWorkLimits,
    resolution_budget_scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_budget_scope: Option<&'static str>,
}

// Deliberately no Debug: original documents and mention surfaces are private.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Population {
    documents: Vec<ResolutionDocument>,
    options: ResolveOptions,
}

pub(super) fn definition() -> clap::Command {
    let command = GraphArgs::augment_args(classify::definition().name("resolve")
        .about("Resolve exact source-anchored mentions across a complete snapshot with native INT4")
        .long_about("Read one bounded JSON object containing original documents, anchored mentions and an explicit lexical/context/margin policy. Add --discover-entities for raw {id,text} documents and optional top-level ner options; add --chunked to discover mentions across every original source chunk. Lexical overlap selects candidates only. Every pair is scored in both presentation orders with the exact A/B/C-plus-EOS language and full-vocabulary denominators. Complete-link clustering refuses missing, conflicting or uncertain cross-pairs. Supplied anchors and their complete graph preflight before model IO. Discovery instead preflights all NER chunks, verifies every recovered occurrence against original text, then plans the complete pair graph under the same remaining five-axis budget. No NER reservation is refunded after early EOS. Source conversion and digest-pinned snapshots use the same actual model identity. Empty supplied-mention graphs allocate no native engine/KV/scratch after model loading; raw nonempty documents always require NER. Scores are uncalibrated, chunk boundaries may miss entities and cluster IDs are snapshot-local. No generation controls, prefill override, thinking, tools, truncation or backend fallback. Blocking IO observes the total deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("JSON {documents:[{id,text,mentions:[{entity_type,surface,span}]}],options:{blocking,context_scalars,minimum_margin_milli}}; '-' reads stdin"))
        .mut_arg("max_candidate_tokens", |arg| arg.default_value("2").hide(true))
        .mut_arg("max_projected_logits", |arg| arg.default_value("10000000000"))
        .mut_arg("max_dot_products", |arg| arg.default_value("100000000000000"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native resolution result; candidate provenance is additional"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-snapshot native forward ceiling, shared by all discovery chunks, pairs and both presentation orders")));
    input_encoding::definition(EncodingArgs::augment_args(compute::definition(
        CohortArgs::augment_args(discovery::DiscoveryArgs::augment_args(command)))))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Safe-BPE pair checks per encoded mention record or source segment"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Safe-BPE pair bytes per encoded mention record or source segment"))
        .mut_arg("max_total_tokenizer_input_bytes", |arg| arg.help("Cumulative input bytes per complete NER snapshot operation or complete pair-graph operation; safe-bpe-v1 only"))
        .mut_arg("max_total_tokenizer_output_tokens", |arg| arg.help("Cumulative encoded tokens per complete NER snapshot operation or complete pair-graph operation; separate from native context"))
        .mut_arg("max_total_tokenizer_pair_checks", |arg| arg.help("Cumulative Safe-BPE pair checks per complete NER snapshot operation or pair-graph operation"))
        .mut_arg("max_total_tokenizer_pair_bytes", |arg| arg.help("Cumulative Safe-BPE examined bytes per complete NER snapshot operation or pair-graph operation"))
        .mut_arg("compute_threads", |arg| arg.help("Coordinator plus persistent INT4 row workers, 1..=64; 1 is serial. One team covers every NER chunk, candidate pair and both presentation orders; resolve-corpus reuses it across all flush epochs. Cannot combine with --cohort-pairs or --cohort-chunks. Worker scratch is admitted before IO"))
        .after_long_help("Native cohorts are explicit and independently selected. --cohort-pairs N executes both presentation orders of up to N complete pairs together (1..=32 pairs, at most 64 heads). --cohort-chunks N executes up to N independent NER discovery chunks together (1..=64); it requires --discover-entities and also applies to unchunked documents. All original sources and the complete pair graph retain their existing semantics. Every window drains before a dependent phase starts. The requested schedule admits its complete native phase peak and simultaneous preparation without enlarging any task's context or replenishing native, mask or grounding budgets. Physical cohort counters are separate from the semantic result. No measured speedup or model parity is implied.\n\n--input-encoding safe-bpe-v1 preserves the exact original source and serialized mention-record bytes while excluding template-control tokens. It supports supplied mentions, NER discovery, original-document chunking, native cohorts and persistent row teams. Per-call byte/token ceilings are min(--max-input-bytes, 1048576). --max-total-tokenizer-input-bytes and --max-total-tokenizer-output-tokens default to eight times --max-input-bytes; total pair checks default to 268435456 and examined pair bytes to 4294967296. One NER ledger covers the complete snapshot's sizing probes, retained plans and execution reconstruction. Pair scoring separately uses one ledger across every pair, preliminary preflight and hosted reconstruction, and reuses encoded records for reverse presentation. Neither a pair, presentation order nor host boundary renews its allowance. These are complete snapshot phase budgets, not lifetime corpus tokenizer counters. Corpus flush starts a new snapshot with the same fixed ceilings. Every workspace is admitted before private input/model IO. Exhaustion rejects without byte fallback, context truncation or partial graph publication. Compact policy and every ceiling bind execution identity and are reported in provenance; different input tokens may change model output and imply no quality, parity or speed claim.")
}
impl ResolveArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if matches.value_source("max_candidate_tokens") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "INT4 resolution uses a fixed label-plus-EOS language"));
        }
        Ok(Self { scoring: classify::ScoringArgs::from_matches(matches)?, graph: GraphArgs::from_arg_matches(matches)?,
            discovery: discovery::DiscoveryArgs::from_arg_matches(matches)?,
            cohorts: CohortArgs::from_arg_matches(matches)?, encoding: EncodingArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        self.cohorts.validate(self.discovery.discover_entities)?;
        // Numerical row teams retain serial prompt/task order. They do not
        // enable prefill or compose with independent native cohort schedules.
        if args.prefill()?.is_some() || self.scoring.cohort_head_limit().is_some()
            || (args.compute.threads() != 1 && self.cohorts.enabled()) {
            return Err(CandidateError::Arguments);
        }
        let encoding_bytes = self.encoding_workspace_bytes(args)?;
        // The shared score quote prices retained prompts, not a single-call
        // text encoder. Entity phases admit their own cumulative workspaces.
        let mut storage = args.clone();
        storage.input_encoding = input_encoding::InputEncodingArgs::default();
        let mut limits = self.scoring.validate_one_off_with_prefill(&storage)?;
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
            g.max_pairs as u64 * 4096, limits.result_bytes as u64 * 16, MIB,
            self.discovery.extra_graph_bytes(self, args)?])?;
        if graph_bytes < graph_floor { return Err(CandidateError::Arguments); }
        self.discovery.check_preparation(self, args, limits)?;
        // The CLI keeps its input/planning/publication owner. The host separately
        // retains the moved document capacities, its pinned planner/preparation
        // and complete graph until physical completion. This is a conservative
        // declared reservation quote, not an OS RSS estimate.
        limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, limits.preparation_bytes,
            graph_bytes, self.host_input_bytes(args)?, self.discovery.temporary_bytes(limits, self.graph.max_documents)?,
            encoding_bytes, encoding_bytes])?;
        #[cfg(feature = "asupersync-runtime")]
        if self.cohorts.enabled() {
            limits.minimum_memory_bytes = limits.minimum_memory_bytes.max(runtime::cohort_memory_floor(self, args, limits)?);
        }
        if limits.minimum_memory_bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(limits)
    }
    fn encoding_limits(self, args: &Int4Args) -> Result<Option<EncodingLimits>, CandidateError> {
        if args.input_encoding.is_default() {
            return if self.encoding == EncodingArgs::default() { Ok(None) } else { Err(CandidateError::Arguments) };
        }
        // Counting probes must be able to encode a candidate before the
        // lossless chunker shrinks it. Actual prompts independently obey KV.
        let bytes = args.max_input_bytes.min(SafeBpeLimits::MAX_INPUT_BYTES);
        let UntrustedEncoding::SafeBpeV1(call) = args.input_encoding.policy(bytes, bytes)?
            else { return Err(CandidateError::Arguments); };
        let default_work = u64::try_from(args.max_input_bytes).ok()
            .and_then(|bytes| bytes.checked_mul(8)).ok_or(CandidateError::Arguments)?;
        let complete = SafeBpeWorkLimits {
            max_input_bytes: self.encoding.max_total_tokenizer_input_bytes.unwrap_or(default_work),
            max_output_tokens: self.encoding.max_total_tokenizer_output_tokens.unwrap_or(default_work),
            max_pair_checks: self.encoding.max_total_tokenizer_pair_checks.unwrap_or(268_435_456),
            max_pair_bytes: self.encoding.max_total_tokenizer_pair_bytes.unwrap_or(4_294_967_296),
        };
        complete.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(EncodingLimits { call, complete }))
    }
    fn encoding_workspace_bytes(self, args: &Int4Args) -> Result<u64, CandidateError> {
        let Some(limits) = self.encoding_limits(args)? else { return Ok(0); };
        let resolution = limits.workspace_bytes()?;
        // Keep both phase reservations even though their tokenization is
        // sequential. Each operation owns its private nonrenewable ledger.
        let source = if self.discovery.discover_entities { limits.workspace_bytes()? } else { 0 };
        sum(&[resolution, source])
    }
    #[cfg(feature = "asupersync-runtime")]
    fn encoding_report(self, args: &Int4Args) -> Result<Option<EncodingReport>, CandidateError> {
        Ok(self.encoding_limits(args)?.map(|limits| EncodingReport {
            policy: limits.policy(), complete_phase_limits: limits.complete,
            resolution_budget_scope: "complete_snapshot_pair_preflight_and_reconstruction",
            discovery_budget_scope: self.discovery.discover_entities
                .then_some("complete_snapshot_discovery_and_reconstruction"),
        }))
    }
    fn graph_bytes(self) -> Result<u64, CandidateError> { mib(self.graph.graph_reserve_mib) }
    fn host_input_bytes(self, args: &Int4Args) -> Result<u64, CandidateError> {
        sum(&[args.max_input_bytes as u64 * 32, self.graph.max_documents as u64 * 128,
            self.graph.max_mentions as u64 * 256, self.discovery.witness_bytes(self)])
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
