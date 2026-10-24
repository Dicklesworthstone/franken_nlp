//! Explicit raw original-document discovery under the same resolution command.
use super::*;
use crate::{batch::source::SourceMaskBudget,
    corpus::{entities::EntityDocument, entities_int4::Int4EntityConfig},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    tasks::{ir::TaskBudget, mapreduce::ChunkLimits, ner::NerOptions, source_planning::SourcePlanningLimits},
    validation::grounded_fields::GroundingBudget};

const MASK_STEP: u64 = 2_000_000;
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DiscoveryArgs {
    /// Discover native source-constrained NER mentions before resolving the whole raw snapshot.
    #[arg(long)]
    pub(super) discover_entities: bool,
    /// Partition every original document, run NER on all chunks, then cluster once.
    #[arg(long, requires = "discover_entities")]
    chunked: bool,
    /// NER tokens including EOS per document/chunk (default 256); not pair-label depth.
    #[arg(long, requires = "discover_entities")]
    max_ner_tokens: Option<usize>,
    /// Mask traversal ceiling per NER document/chunk (default 1000000000).
    #[arg(long, requires = "discover_entities")]
    max_ner_mask_node_visits: Option<u64>,
    /// One mask allowance for the whole snapshot (default 1000000000000).
    #[arg(long, requires = "discover_entities")]
    max_snapshot_mask_node_visits: Option<u64>,
    /// Original text plus expanded mention bytes (default 16777216), separate from JSON wire bytes.
    #[arg(long, requires = "discover_entities")]
    max_expanded_bytes: Option<usize>,
    /// Maximum NER chunks per original document (default 64, maximum 256).
    #[arg(long, requires = "chunked")]
    max_ner_chunks: Option<usize>,
    /// Maximum NER chunks across the complete snapshot (default 1024).
    #[arg(long, requires = "chunked")]
    max_snapshot_ner_chunks: Option<usize>,
    /// Chunk byte ceiling (default 4096); actual pinned prompt capacity also applies.
    #[arg(long, requires = "chunked")]
    max_ner_chunk_bytes: Option<usize>,
    /// Pinned tokenizer calls for each complete document partition (default 8192).
    #[arg(long, requires = "chunked")]
    max_ner_tokenizer_calls: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawInput {
    pub(super) documents: Vec<EntityDocument>,
    pub(super) options: ResolveOptions,
    #[serde(default)]
    pub(super) ner: NerOptions,
}
impl DiscoveryArgs {
    fn tokens(self) -> usize { self.max_ner_tokens.unwrap_or(256) }
    fn item_masks(self) -> u64 { self.max_ner_mask_node_visits.unwrap_or(1_000_000_000) }
    fn snapshot_masks(self) -> u64 { self.max_snapshot_mask_node_visits.unwrap_or(1_000_000_000_000) }
    fn expanded_bytes(self) -> usize { self.max_expanded_bytes.unwrap_or(16 * 1024 * 1024) }
    fn per_document_chunks(self) -> usize { self.max_ner_chunks.unwrap_or(64) }
    fn snapshot_chunks(self, command: ResolveArgs) -> usize {
        if self.chunked { self.max_snapshot_ner_chunks.unwrap_or(1024) } else { command.graph.max_documents }
    }
    fn validate(self, args: &Int4Args) -> Result<(), CandidateError> {
        if !self.discover_entities {
            if self.chunked || self.max_ner_tokens.is_some() || self.max_ner_mask_node_visits.is_some()
                || self.max_snapshot_mask_node_visits.is_some() || self.max_expanded_bytes.is_some()
                || self.max_ner_chunks.is_some() || self.max_snapshot_ner_chunks.is_some()
                || self.max_ner_chunk_bytes.is_some() || self.max_ner_tokenizer_calls.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(());
        }
        if !(1..=1024).contains(&self.tokens()) || self.tokens() >= args.context_tokens
            || self.item_masks() < MASK_STEP * self.tokens() as u64 || self.item_masks() > 1_000_000_000_000
            || self.snapshot_masks() < self.item_masks() || self.snapshot_masks() > 1_000_000_000_000_000
            || !(1..=64 * 1024 * 1024).contains(&self.expanded_bytes()) {
            return Err(CandidateError::Arguments);
        }
        if !self.chunked {
            if self.max_ner_chunks.is_some() || self.max_snapshot_ner_chunks.is_some()
                || self.max_ner_chunk_bytes.is_some() || self.max_ner_tokenizer_calls.is_some() {
                return Err(CandidateError::Arguments);
            }
        } else if !(1..=256).contains(&self.per_document_chunks())
            || !(1..=16_384).contains(&self.max_snapshot_ner_chunks.unwrap_or(1024))
            || !(4..=MAX_INPUT_BYTES).contains(&self.max_ner_chunk_bytes.unwrap_or(4096))
            || args.max_input_bytes < 4
            || !(1..=1_000_000).contains(&self.max_ner_tokenizer_calls.unwrap_or(8192)) {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    pub(super) fn extra_graph_bytes(self, command: ResolveArgs, args: &Int4Args) -> Result<u64, CandidateError> {
        self.validate(args)?;
        if !self.discover_entities { return Ok(0); }
        sum(&[self.expanded_bytes() as u64 * 2, self.witness_bytes(command)])
    }
    pub(super) fn witness_bytes(self, command: ResolveArgs) -> u64 {
        if self.discover_entities { self.snapshot_chunks(command) as u64 * 1024 } else { 0 }
    }
    pub(super) fn temporary_bytes(self, limits: Int4Limits) -> Result<u64, CandidateError> {
        if self.discover_entities { sum(&[limits.result_bytes as u64 * 4, self.tokens() as u64 * 8]) }
            else { Ok(0) }
    }
    pub(super) fn check_preparation(self, command: ResolveArgs, args: &Int4Args, limits: Int4Limits)
        -> Result<(), CandidateError> {
        if !self.discover_entities { return Ok(()); }
        // One grammar/source index at a time; both pinned planners, compact
        // witnesses and final publication remain owned by the CLI preparation.
        let minimum = sum(&[256 * MIB, args.max_input_bytes as u64 * 32,
            (limits.result_bytes as u64 + 4096) * 2, limits.result_bytes as u64 * 16,
            4096 * 32, 2 * MIB, self.witness_bytes(command)])?;
        if limits.preparation_bytes < minimum { return Err(CandidateError::Arguments); }
        Ok(())
    }
    pub(super) fn input(self, command: ResolveArgs, args: &Int4Args, text: &str) -> Result<RawInput, CandidateError> {
        if !self.discover_entities || text.len() > args.max_input_bytes { return Err(CandidateError::Input); }
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
            max_depth: 8, max_string_bytes: args.max_input_bytes,
        }).map_err(|_| CandidateError::Input)?;
        let input: RawInput = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
        input.ner.validate().map_err(|_| CandidateError::Input)?;
        if input.documents.len() > command.graph.max_documents || input.options.context_scalars > 2048
            || !(1..=1_000_000).contains(&input.options.minimum_margin_milli) {
            return Err(CandidateError::Input);
        }
        let mut ids = std::collections::BTreeSet::new(); let mut bytes = 0_usize;
        for document in &input.documents {
            if document.id.is_empty() || document.id.len() > 256 || document.id.chars().any(char::is_control)
                || document.text.is_empty() || !ids.insert(document.id.as_str()) {
                return Err(CandidateError::Input);
            }
            bytes = bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len()))
                .filter(|&n| n <= self.expanded_bytes()).ok_or(CandidateError::Input)?;
        }
        // Every nonempty original contributes at least one pass. Exact chunk
        // counts and all native/mask reservations are compiled after model binding.
        let masks = (input.documents.len() as u64).checked_mul(self.item_masks()).ok_or(CandidateError::Input)?;
        if masks > self.snapshot_masks() || input.documents.len() > self.snapshot_chunks(command) {
            return Err(CandidateError::Input);
        }
        drop(ids);
        Ok(input)
    }
    pub(super) fn configuration(self, command: ResolveArgs, args: &Int4Args, limits: Int4Limits,
        ner: NerOptions, resolution: ResolveOptions) -> Result<Int4EntityConfig, CandidateError> {
        self.validate(args)?;
        if !self.discover_entities { return Err(CandidateError::Arguments); }
        let mut graph = command.graph_limits(args, limits); graph.max_input_bytes = self.expanded_bytes();
        let chunks = if self.chunked {
            let chunks = ChunkLimits { max_input_bytes: args.max_input_bytes,
                max_chunk_bytes: self.max_ner_chunk_bytes.unwrap_or(4096).min(args.max_input_bytes),
                max_chunk_tokens: args.context_tokens, context_tokens: args.context_tokens,
                reserved_tokens: self.tokens(), max_chunks: self.per_document_chunks(),
                max_tokenizer_calls: self.max_ner_tokenizer_calls.unwrap_or(8192) };
            chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
            Some(chunks)
        } else { None };
        let config = Int4EntityConfig {
            ner, ner_budget: TaskBudget { max_input_tokens: (args.context_tokens - self.tokens()) as u32,
                max_output_tokens: self.tokens() as u32, max_output_bytes: limits.result_bytes as u64,
                max_grammar_states: 4096, max_kv_bytes: limits.native.kv_bytes },
            source_planning: SourcePlanningLimits { max_input_bytes: args.max_input_bytes,
                max_context_tokens: args.context_tokens, max_passages: 1,
                compiler: CompileLimits { max_states: 4096, max_output_bytes: limits.result_bytes, ..CompileLimits::default() },
                source: SourceRuntimeLimits::default() },
            masks: SourceMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: MASK_STEP as usize,
                    checkpoint_interval_nodes: 256 },
                max_visits_per_item: self.item_masks(), max_visits_per_run: self.snapshot_masks() },
            resolution, graph, scoring: command.scoring_limits(args, limits),
            verification: GroundingBudget { max_fields: 1_000_000, max_matches: command.graph.max_mentions,
                max_scan_steps: command.graph.max_scan_steps },
            max_model_work: command.scoring.work_ceiling(), max_result_bytes: limits.result_bytes,
            chunks, max_snapshot_chunks: self.snapshot_chunks(command),
        };
        config.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests;
