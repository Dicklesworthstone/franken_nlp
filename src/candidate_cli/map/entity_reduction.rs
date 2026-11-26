//! Exact typed-entity union for complete INT8 NER maps.
use super::*;
use crate::tasks::source_planning::quantized::long::entity_reduction::Int8EntityLimits;

#[derive(Args)]
pub(in crate::candidate_cli) struct EntityReductionArgs {
    /// Group identical surface/type pairs across contributing NER chunks.
    /// Keep all original-coordinate occurrences and per-chunk ambiguity.
    #[arg(long)]
    pub reduce_entities: bool,
    /// Maximum native proposals examined, including duplicates (default 16384).
    #[arg(long, requires = "reduce_entities")]
    max_entity_proposals: Option<usize>,
    /// Maximum distinct retained occurrence spans across typed groups (default 65536).
    #[arg(long, requires = "reduce_entities")]
    max_entity_evidence_spans: Option<usize>,
    /// Cumulative proposal bytes and checked occurrence coordinates (default 67108864).
    #[arg(long, requires = "reduce_entities")]
    max_entity_scan_work: Option<u64>,
}
impl EntityReductionArgs {
    pub(in crate::candidate_cli) fn limits(&self, task: &str, max_result_bytes: usize)
        -> Result<Option<Int8EntityLimits>, CandidateError> {
        if !self.reduce_entities { return Ok(None); }
        if task != "ner" { return Err(CandidateError::Arguments); }
        let defaults = Int8EntityLimits::default();
        let limits = Int8EntityLimits {
            max_proposals: self.max_entity_proposals.unwrap_or(defaults.max_proposals),
            max_evidence_spans: self.max_entity_evidence_spans.unwrap_or(defaults.max_evidence_spans),
            max_scan_work: self.max_entity_scan_work.unwrap_or(defaults.max_scan_work),
            max_result_bytes,
        };
        limits.workspace_bytes().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
}
