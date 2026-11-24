//! Whole-document extraction options, separate from fixed source-task options.
use super::*;
use crate::{batch::extract::quantized::long::Int8ExtractionMapLimits,
    candidate_cli::extract as schema, native_engine::strict_int8::prefill::Int8PrefillLimits,
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan}};

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
    /// With --task extract: bounded layer-major prompt morsels across every
    /// chunk, 1..=64 rows. Composes with the same persistent compute team.
    /// Extra coordinator and worker storage is admitted before private IO.
    #[arg(long, value_name = "ROWS", requires = "schema")]
    prefill_rows: Option<usize>,
}
impl ExtractionArgs {
    pub(super) fn validate(&self, task: &str) -> Result<(), CandidateError> {
        if task != "extract" {
            if self.schema.is_some() || self.source_membership || self.max_extraction_fields.is_some()
                || self.max_extraction_evidence_spans.is_some() || self.max_extraction_scan_steps.is_some()
                || self.prefill_rows.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(());
        }
        schema::check_schema_path(self.path()?)?;
        self.verification()?;
        self.prefill()?;
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
    pub(in crate::candidate_cli) fn prefill(&self) -> Result<Option<Int8PrefillLimits>, CandidateError> {
        self.prefill_rows.map(|rows| {
            let bytes = Int8PrefillLimits::required_extra_scratch_bytes(rows).map_err(|_| CandidateError::Arguments)?;
            Ok(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes })
        }).transpose()
    }
    pub(super) fn check_schedule_memory(&self, command: &MapCommand, args: &CandidateArgs, limits: Limits)
        -> Result<(), CandidateError> {
        if limits.compute.threads == 1 && self.prefill()?.is_none() { return Ok(()); }
        if self.schedule_peak_bytes(command, args, limits)? > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    fn schedule_peak_bytes(&self, command: &MapCommand, args: &CandidateArgs, limits: Limits)
        -> Result<u64, CandidateError> {
        let native = Int8MemoryRequirement::for_context(args.context_tokens).map_err(|_| CandidateError::Arguments)?;
        let tokens = (command.max_chunks as u64).checked_mul(args.max_new_tokens as u64)
            .ok_or(CandidateError::Arguments)?;
        let base = compute::request_peak_bytes(limits, native, command.max_map_result_bytes as u64,
            tokens, 0, limits.preparation_bytes)?;
        let extra = self.prefill()?.map(Int8PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        // Price the complete retained frontier, original-coordinate scans and
        // a separately transferred input owner, not one chunk multiplied only
        // after allocation. The host independently reserves its actual counts.
        let retained = (command.host.max_result_bytes as u64).checked_mul(command.max_chunks as u64)
            .and_then(|n| n.checked_add(command.max_live_value_bytes as u64))
            .and_then(|n| n.checked_mul(4)).ok_or(CandidateError::Arguments)?;
        let verification = self.verification()?;
        let fields = (verification.max_fields as u64).checked_mul(std::mem::size_of::<SourceFieldEvidence>() as u64)
            .ok_or(CandidateError::Arguments)?;
        let spans = (verification.max_matches as u64).checked_mul(std::mem::size_of::<VerifiedSourceSpan>() as u64)
            .and_then(|n| n.checked_mul(2)).ok_or(CandidateError::Arguments)?;
        let encoding = command.encoding.limits(command)?.map(|limits| limits.workspace_bytes())
            .transpose()?.unwrap_or(0);
        [base, extra, retained, fields, spans, tokens.checked_mul(8).ok_or(CandidateError::Arguments)?,
            command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            (args.max_input_bytes as u64).checked_mul(4).ok_or(CandidateError::Arguments)?,
            2 * schema::SCHEMA_BYTES as u64, encoding]
            .into_iter().try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or(CandidateError::Arguments)
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
