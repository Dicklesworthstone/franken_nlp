//! Explicit raw-document mode for resolve. Existing mention input is unchanged.
use super::*;
use crate::{batch::source::SourceMaskBudget,
    corpus::{entities::EntityDocument, entities_int8::Int8EntityConfig},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    tasks::{ir::TaskBudget, ner::NerOptions, source_planning::SourcePlanningLimits},
    validation::grounded_fields::GroundingBudget};
pub(in crate::candidate_cli) mod long;

#[derive(Args)]
pub(in crate::candidate_cli) struct DiscoveryArgs {
    /// Discover source-constrained NER mentions before resolving the raw snapshot.
    #[arg(long)]
    pub discover_entities: bool,
    #[command(flatten)]
    pub long: long::DiscoveryChunkArgs,
    /// NER tokens including EOS per document, or per chunk with --chunked.
    /// This is not the pair-label depth.
    #[arg(long, default_value_t = 256)]
    max_ner_tokens: usize,
    /// NER mask visits per document, or per chunk with --chunked.
    /// The independent whole-snapshot mask ceiling never renews.
    #[arg(long, default_value_t = 1_000_000_000)]
    max_ner_mask_node_visits: u64,
    #[arg(long, default_value_t = 1_000_000_000_000)]
    max_snapshot_mask_node_visits: u64,
    /// Logical original text plus expanded mention bytes, separate from wire input.
    #[arg(long, default_value_t = 16_777_216)]
    max_expanded_bytes: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::candidate_cli) struct RawInput {
    pub documents: Vec<EntityDocument>,
    pub options: ResolveOptions,
    /// Omitted means the existing person/organization/location NER defaults.
    #[serde(default)]
    pub ner: NerOptions,
}
impl DiscoveryArgs {
    pub(super) fn extra_graph_bytes(&self, context: usize) -> Result<u64, CandidateError> {
        let chunks = self.long.extra_graph_bytes(self.discover_entities)?;
        if !self.discover_entities { return Ok(0); }
        if !(1..=1024).contains(&self.max_ner_tokens) || self.max_ner_tokens >= context
            || !(2_000_000..=1_000_000_000_000).contains(&self.max_ner_mask_node_visits)
            || self.max_snapshot_mask_node_visits < self.max_ner_mask_node_visits
            || self.max_snapshot_mask_node_visits > 1_000_000_000_000_000
            || !(1..=64 * 1024 * 1024).contains(&self.max_expanded_bytes) { return Err(CandidateError::Arguments); }
        (self.max_expanded_bytes as u64).checked_mul(2).and_then(|n| n.checked_add(chunks)).ok_or(CandidateError::Arguments)
    }
    pub(in crate::candidate_cli) fn input(&self, command: &ResolveCommand, text: &str) -> Result<RawInput, CandidateError> {
        if !self.discover_entities || text.len() > command.host.max_input_bytes { return Err(CandidateError::Input); }
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
            max_depth: 8, max_string_bytes: command.host.max_input_bytes,
        }).map_err(|_| CandidateError::Input)?;
        let input: RawInput = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
        input.ner.validate().map_err(|_| CandidateError::Input)?;
        if input.documents.len() > command.max_documents || input.options.context_scalars > 2048
            || !(1..=1_000_000).contains(&input.options.minimum_margin_milli) { return Err(CandidateError::Input); }
        let mut ids = std::collections::BTreeSet::new(); let mut bytes = 0_usize;
        for document in &input.documents {
            if document.id.is_empty() || document.id.len() > 256 || document.id.chars().any(char::is_control)
                || !ids.insert(document.id.as_str()) || (self.long.chunked && document.text.is_empty()) {
                return Err(CandidateError::Input);
            }
            bytes = bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len()))
                .filter(|&n| n <= self.max_expanded_bytes).ok_or(CandidateError::Input)?;
        }
        // In chunked mode this is only a lower bound. Exact chunk counts and
        // all mask/native reservations are checked for the snapshot in preflight.
        let masks = (input.documents.len() as u64).checked_mul(self.max_ner_mask_node_visits)
            .ok_or(CandidateError::Input)?;
        if masks > self.max_snapshot_mask_node_visits { return Err(CandidateError::Input); }
        drop(ids);
        Ok(input)
    }
    pub(in crate::candidate_cli) fn config(&self, command: &ResolveCommand, limits: Limits,
        ner: NerOptions, resolution: ResolveOptions) -> Int8EntityConfig {
        let mut graph = command.graph(); graph.max_input_bytes = self.max_expanded_bytes;
        Int8EntityConfig {
            ner, ner_budget: TaskBudget { max_input_tokens: (command.host.context_tokens - self.max_ner_tokens) as u32,
                max_output_tokens: self.max_ner_tokens as u32, max_output_bytes: command.host.max_result_bytes as u64,
                max_grammar_states: 4096, max_kv_bytes: limits.kv_bytes },
            source_planning: SourcePlanningLimits { max_input_bytes: command.host.max_input_bytes,
                max_context_tokens: command.host.context_tokens, max_passages: 1,
                compiler: CompileLimits { max_states: 4096, max_output_bytes: command.host.max_result_bytes, ..CompileLimits::default() },
                source: SourceRuntimeLimits::default() },
            masks: SourceMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
                max_visits_per_item: self.max_ner_mask_node_visits, max_visits_per_run: self.max_snapshot_mask_node_visits },
            resolution, graph, scoring: command.scoring(limits),
            verification: GroundingBudget { max_fields: 1_000_000, max_matches: command.max_mentions,
                max_scan_steps: command.max_scan_steps },
            max_model_work: command.host.work_ceiling(), max_result_bytes: command.host.max_result_bytes,
        }
    }
}

#[cfg(test)] mod tests;
