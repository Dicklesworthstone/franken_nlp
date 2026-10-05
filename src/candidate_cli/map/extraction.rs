//! Whole-document extraction options, separate from fixed source-task options.
use super::*;
use crate::{batch::extract::quantized::long::Int8ExtractionMapLimits,
    candidate_cli::extract as schema, validation::grounded_fields::GroundingBudget};

#[derive(Args)]
pub(in crate::candidate_cli) struct ExtractionArgs {
    /// Exact local JSON schema for --task extract; stdin belongs to the source.
    #[arg(long, value_name = "FILE", required_if_eq("task", "extract"), conflicts_with = "options")]
    schema: Option<PathBuf>,
    /// Enforce verbatim source annotations per chunk, with original coordinates.
    /// This does not establish semantic truth or merge independent JSON objects.
    #[arg(long, requires = "schema")]
    pub source_membership: bool,
    /// Nonrenewable independent verification fields across ALL chunks (default 4096).
    #[arg(long, requires = "schema")]
    max_extraction_fields: Option<usize>,
    /// Original-coordinate occurrence spans across ALL chunks (default 16384).
    #[arg(long, requires = "schema")]
    max_extraction_evidence_spans: Option<usize>,
    /// Nonrenewable independent source verification work (default 67108864).
    #[arg(long, requires = "schema")]
    max_extraction_scan_steps: Option<u64>,
}
impl ExtractionArgs {
    pub(super) fn validate(&self, task: &str) -> Result<(), CandidateError> {
        if task != "extract" {
            if self.schema.is_some() || self.source_membership || self.max_extraction_fields.is_some()
                || self.max_extraction_evidence_spans.is_some() || self.max_extraction_scan_steps.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(());
        }
        schema::check_schema_path(self.path()?)?;
        self.verification()?;
        Ok(())
    }
    pub(in crate::candidate_cli) fn path(&self) -> Result<&std::path::Path, CandidateError> {
        self.schema.as_deref().ok_or(CandidateError::Arguments)
    }
    fn verification(&self) -> Result<GroundingBudget, CandidateError> {
        let defaults = GroundingBudget::default();
        let budget = GroundingBudget {
            max_fields: self.max_extraction_fields.unwrap_or(defaults.max_fields),
            max_matches: self.max_extraction_evidence_spans.unwrap_or(defaults.max_matches),
            max_scan_steps: self.max_extraction_scan_steps.unwrap_or(defaults.max_scan_steps),
        };
        if !(1..=1_000_000).contains(&budget.max_fields) || !(1..=1_000_000).contains(&budget.max_matches)
            || !(1..=1_000_000_000_000).contains(&budget.max_scan_steps) { return Err(CandidateError::Arguments); }
        Ok(budget)
    }
    /// Retained original schemas and schema prompt storage for every prepared
    /// chunk are additional to the generic grammar/context staging allowance.
    pub(super) fn schema_reserve_per_chunk(&self) -> u64 {
        if self.schema.is_some() { schema::SCHEMA_BYTES as u64 * 2 } else { 0 }
    }
}
impl MapCommand {
    /// Requested ceilings only. The actual extraction compiler tightens context
    /// using the exact schema, pinned scaffold and real source token counts.
    pub(in crate::candidate_cli) fn extraction_mapping(&self) -> Result<Int8ExtractionMapLimits, CandidateError> {
        if self.kind()? != BuiltInTask::Extract || self.options.is_some() { return Err(CandidateError::Arguments); }
        self.extraction.validate(&self.task)?;
        let chunks = ChunkLimits { max_input_bytes: self.host.max_input_bytes,
            max_chunk_bytes: self.max_chunk_bytes.unwrap_or(self.host.context_tokens).min(self.host.max_input_bytes),
            max_chunk_tokens: self.host.context_tokens, context_tokens: self.host.context_tokens,
            reserved_tokens: self.host.max_new_tokens, max_chunks: self.max_chunks,
            max_tokenizer_calls: self.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        Ok(Int8ExtractionMapLimits { mapping: self.mapping_limits(chunks), verification: self.extraction.verification()? })
    }
}

#[cfg(test)] mod tests;
